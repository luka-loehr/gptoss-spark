#!/usr/bin/env python3
"""Cross-engine, OpenAI-compatible streaming benchmark for GPT-OSS.

The script intentionally uses only the Python standard library so it can run on
the Spark host without downloading client dependencies. Accepted token counts
come only from the server's final usage object.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Iterable


OUTPUT_FIELDS = ("reasoning_content", "reasoning", "content")


@dataclasses.dataclass(frozen=True)
class RequestSpec:
    sequence: int
    prompt_id: str
    messages: list[dict[str, Any]]
    warmup: bool


def now_ns() -> int:
    return time.monotonic_ns()


def seconds_between(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None or end_ns < start_ns:
        return None
    return (end_ns - start_ns) / 1_000_000_000


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_key_value_file(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
    except OSError:
        pass
    return values


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            token = rest.strip().split()[0]
            values[key] = int(token)
    except (OSError, ValueError, IndexError):
        pass
    return values


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()[-16000:]
    except (OSError, subprocess.SubprocessError):
        return None


def host_snapshot(watchdog_state: pathlib.Path) -> dict[str, Any]:
    meminfo = read_meminfo()
    return {
        "captured_at_epoch_s": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mem_total_kb": meminfo.get("MemTotal"),
        "mem_available_kb": meminfo.get("MemAvailable"),
        "swap_total_kb": meminfo.get("SwapTotal"),
        "swap_free_kb": meminfo.get("SwapFree"),
        "memory_psi": command_output(["cat", "/proc/pressure/memory"]),
        "watchdog": read_key_value_file(watchdog_state) if watchdog_state else {},
        "nvidia_smi": command_output(["nvidia-smi"]),
    }


def load_prompts(path: pathlib.Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("id"), str) or not isinstance(item.get("messages"), list):
                raise ValueError(f"invalid prompt at {path}:{line_number}")
            prompts.append(item)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def iter_sse(response: Any) -> Iterable[tuple[int, str]]:
    data_lines: list[str] = []
    while True:
        raw = response.readline()
        received_ns = now_ns()
        if not raw:
            if data_lines:
                yield received_ns, "\n".join(data_lines)
            return
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield received_ns, "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field == "data":
            data_lines.append(value[1:] if separator and value.startswith(" ") else value)


def stream_request(
    spec: RequestSpec,
    *,
    endpoint: str,
    model: str,
    max_tokens: int,
    reasoning_effort: str | None,
    timeout_seconds: float,
    api_key: str | None,
    min_completion_tokens: int,
    min_output_events: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": spec.messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Connection": "close",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started_ns = now_ns()
    first_output_ns: int | None = None
    last_output_ns: int | None = None
    response_opened_ns: int | None = None
    done_ns: int | None = None
    output_events = 0
    stream_events = 0
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    output_parts: list[str] = []
    channel_characters = {field: 0 for field in OUTPUT_FIELDS}
    channel_characters["tool_calls"] = 0
    error: str | None = None
    http_status: int | None = None

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_opened_ns = now_ns()
            http_status = getattr(response, "status", None)
            for received_ns, data in iter_sse(response):
                stream_events += 1
                if data == "[DONE]":
                    done_ns = received_ns
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    error = f"invalid SSE JSON: {exc}: {data[:200]!r}"
                    break
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    event_had_output = False
                    for field in OUTPUT_FIELDS:
                        value = delta.get(field)
                        if isinstance(value, str) and value:
                            output_parts.append(f"<{field}>{value}")
                            channel_characters[field] += len(value)
                            event_had_output = True
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        rendered = json.dumps(tool_calls, sort_keys=True, separators=(",", ":"))
                        output_parts.append(f"<tool_calls>{rendered}")
                        channel_characters["tool_calls"] += len(rendered)
                        event_had_output = True
                    if event_had_output:
                        output_events += 1
                        if first_output_ns is None:
                            first_output_ns = received_ns
                        last_output_ns = received_ns
            if done_ns is None:
                done_ns = now_ns()
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except OSError:
            body = ""
        error = f"HTTP {exc.code}: {body[:2000]}"
        done_ns = now_ns()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        done_ns = now_ns()

    completed_ns = done_ns or now_ns()
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    if usage:
        try:
            completion_tokens = int(usage["completion_tokens"])
        except (KeyError, TypeError, ValueError):
            pass
        try:
            prompt_tokens = int(usage["prompt_tokens"])
        except (KeyError, TypeError, ValueError):
            pass

    ttft_seconds = seconds_between(started_ns, first_output_ns)
    decode_seconds = seconds_between(first_output_ns, last_output_ns)
    output_span_seconds = seconds_between(started_ns, last_output_ns)
    request_seconds = seconds_between(started_ns, completed_ns)
    decode_tps: float | None = None
    end_to_end_tps: float | None = None
    if completion_tokens is not None and decode_seconds and completion_tokens > 1:
        decode_tps = (completion_tokens - 1) / decode_seconds
    if completion_tokens is not None and output_span_seconds and completion_tokens > 0:
        end_to_end_tps = completion_tokens / output_span_seconds

    valid = (
        error is None
        and completion_tokens is not None
        and completion_tokens >= min_completion_tokens
        and output_events >= min_output_events
        and decode_tps is not None
    )
    invalid_reasons: list[str] = []
    if error:
        invalid_reasons.append(error)
    if completion_tokens is None:
        invalid_reasons.append("missing server usage.completion_tokens")
    elif completion_tokens < min_completion_tokens:
        invalid_reasons.append(
            f"only {completion_tokens} completion tokens; require {min_completion_tokens}"
        )
    if output_events < min_output_events:
        invalid_reasons.append(f"only {output_events} output events; require {min_output_events}")
    if decode_tps is None:
        invalid_reasons.append("decode interval unavailable")

    rendered_output = "".join(output_parts).encode("utf-8")
    return {
        "sequence": spec.sequence,
        "prompt_id": spec.prompt_id,
        "warmup": spec.warmup,
        "valid": valid,
        "invalid_reasons": invalid_reasons,
        "http_status": http_status,
        "finish_reason": finish_reason,
        "usage": usage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": ttft_seconds,
        "decode_seconds": decode_seconds,
        "output_span_seconds": output_span_seconds,
        "request_seconds": request_seconds,
        "decode_tokens_per_second": decode_tps,
        "end_to_end_tokens_per_second": end_to_end_tps,
        "stream_events": stream_events,
        "output_events": output_events,
        "channel_characters": channel_characters,
        "output_bytes": len(rendered_output),
        "output_sha256": hashlib.sha256(rendered_output).hexdigest(),
        "clock": "time.monotonic_ns",
    }


def wait_for_server(models_url: str, timeout_seconds: float, api_key: str | None) -> None:
    deadline = time.monotonic() + timeout_seconds
    headers = {"Accept": "application/json", "Connection": "close"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_error = "not attempted"
    while time.monotonic() < deadline:
        request = urllib.request.Request(models_url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"server did not become ready at {models_url}: {last_error}")


def finite(values: Iterable[float | None]) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [item for item in results if not item["warmup"]]
    valid = [item for item in measured if item["valid"]]
    decode = finite(item["decode_tokens_per_second"] for item in valid)
    end_to_end = finite(item["end_to_end_tokens_per_second"] for item in valid)
    ttft = finite(item["ttft_seconds"] for item in valid)
    return {
        "measured_requests": len(measured),
        "valid_requests": len(valid),
        "all_measured_valid": len(valid) == len(measured) and bool(measured),
        "decode_tokens_per_second": {
            "median": statistics.median(decode) if decode else None,
            "min": min(decode) if decode else None,
            "max": max(decode) if decode else None,
            "p10": percentile(decode, 0.10),
            "p90": percentile(decode, 0.90),
        },
        "end_to_end_tokens_per_second": {
            "median": statistics.median(end_to_end) if end_to_end else None,
            "min": min(end_to_end) if end_to_end else None,
            "max": max(end_to_end) if end_to_end else None,
        },
        "ttft_seconds": {
            "median": statistics.median(ttft) if ttft else None,
            "min": min(ttft) if ttft else None,
            "max": max(ttft) if ttft else None,
        },
        "completion_tokens_total": sum(item["completion_tokens"] or 0 for item in valid),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("bench-result.json"))
    parser.add_argument("--metadata-json", type=pathlib.Path)
    # Optional: a key=value file whose contents get recorded alongside the run
    # (this repo's campaign logged a host memory watchdog there).
    parser.add_argument("--watchdog-state", type=pathlib.Path, default=None)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--ready-timeout-seconds", type=float, default=600)
    parser.add_argument("--min-completion-tokens", type=int, default=256)
    parser.add_argument("--min-output-events", type=int, default=8)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.requests < 1 or args.warmup < 0 or args.concurrency < 1:
        raise ValueError("requests/concurrency must be positive and warmup non-negative")
    prompts = load_prompts(args.prompts)
    base_url = args.base_url.rstrip("/")
    wait_for_server(f"{base_url}/models", args.ready_timeout_seconds, args.api_key)

    metadata: dict[str, Any] = {}
    if args.metadata_json:
        metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    metadata.update({
        "base_url": base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "concurrency": args.concurrency,
        "requested_measured_requests": args.requests,
        "requested_warmups": args.warmup,
        "benchmark_argv": sys.argv,
    })

    specs: list[RequestSpec] = []
    total = args.warmup + args.requests
    for sequence in range(total):
        prompt = prompts[sequence % len(prompts)]
        specs.append(RequestSpec(
            sequence=sequence,
            prompt_id=prompt["id"],
            messages=prompt["messages"],
            warmup=sequence < args.warmup,
        ))

    before = host_snapshot(args.watchdog_state)
    results: list[dict[str, Any]] = []
    print(
        f"benchmark_start model={args.model} warmup={args.warmup} "
        f"requests={args.requests} concurrency={args.concurrency}",
        flush=True,
    )
    lock = threading.Lock()

    def execute(spec: RequestSpec) -> dict[str, Any]:
        item = stream_request(
            spec,
            endpoint=f"{base_url}/chat/completions",
            model=args.model,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout_seconds,
            api_key=args.api_key,
            min_completion_tokens=args.min_completion_tokens,
            min_output_events=args.min_output_events,
        )
        with lock:
            print(
                f"request sequence={item['sequence']} prompt={item['prompt_id']} "
                f"warmup={item['warmup']} valid={item['valid']} "
                f"tokens={item['completion_tokens']} decode_tps={item['decode_tokens_per_second']} "
                f"e2e_tps={item['end_to_end_tokens_per_second']} ttft={item['ttft_seconds']}",
                flush=True,
            )
        return item

    # Warmups remain serial so compilation/cache work cannot overlap measurements.
    for spec in specs[: args.warmup]:
        results.append(execute(spec))

    measured_specs = specs[args.warmup :]
    if args.concurrency == 1:
        for spec in measured_specs:
            results.append(execute(spec))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(execute, spec) for spec in measured_specs]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda item: item["sequence"])
    after = host_snapshot(args.watchdog_state)
    artifact = {
        "schema_version": 1,
        "metadata": metadata,
        "host_before": before,
        "host_after": after,
        "results": results,
        "summary": summarize(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True), flush=True)
    print(f"artifact={args.output}", flush=True)
    return 0 if artifact["summary"]["all_measured_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
