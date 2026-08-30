import json, sys, time, urllib.request

BASE = sys.argv[1]; MODEL = sys.argv[2]; OUT = sys.argv[3]
PROMPTS = [
 "Erkläre den Unterschied zwischen Mitose und Meiose für eine 9. Klasse.",
 "Explain the difference between TCP and UDP, and when to use each.",
 "Schreibe eine kurze Inhaltsangabe von Goethes Faust I (max 150 Wörter).",
 "Write a Python function that merges two sorted lists in O(n).",
 "Was ist 4728 * 391? Rechne Schritt für Schritt.",
 "Nenne die drei Newtonschen Gesetze und je ein Alltagsbeispiel.",
 "Translate to English: Der Klimawandel stellt Schulen vor neue Aufgaben in der Bildung.",
 "A train travels 240 km in 2.5 hours. What is its average speed in m/s?",
 "Erkläre den Fotoeffekt und warum er das Wellenmodell des Lichts in Frage stellte.",
 "Write a SQL query to find the second-highest salary in a table employees(id, name, salary).",
 "Was sind die Hauptursachen des Ersten Weltkriegs? Stichpunkte.",
 "Simplify: (3x^2 - 12)/(x - 2)",
 "Erkläre einer 7. Klasse, wie ein Vulkan entsteht.",
 "What is the time complexity of quicksort in the worst case and why?",
 "Konjugiere das französische Verb aller im Présent.",
 "Ein Auto beschleunigt in 8 s von 0 auf 100 km/h. Berechne die mittlere Beschleunigung.",
 "Explain photosynthesis light and dark reactions briefly.",
 "Schreibe einen höflichen Brief an Eltern über einen anstehenden Wandertag.",
 "Factor: x^2 + 5x + 6, then solve x^2 + 5x + 6 = 0.",
 "Was ist der Unterschied zwischen Wetter und Klima?",
]
results = []
for i, p in enumerate(PROMPTS):
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}],
            "max_tokens": 256, "temperature": 0, "top_p": 1,
            "logprobs": True, "top_logprobs": 5, "reasoning_effort": "low"}
    req = urllib.request.Request(BASE + "/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    content_lp = lp.get("content") or []
    toks = [{"t": x.get("token"), "lp": x.get("logprob"),
             "top": [y.get("token") for y in (x.get("top_logprobs") or [])]} for x in content_lp]
    results.append({"i": i, "prompt": p,
        "content": ch["message"].get("content"),
        "reasoning": (ch["message"].get("reasoning_content") or "")[:2000],
        "tokens": toks, "usage": d.get("usage"), "secs": round(time.time() - t0, 2)})
    print(i, len(toks), "logprob-toks", d["usage"]["completion_tokens"], "completion", results[-1]["secs"], "s", flush=True)
json.dump(results, open(OUT, "w"), ensure_ascii=False)
print("saved", OUT)
