"""Turn a timestamped jev-gpt trace into video/data.json: per step the shown text, the level-1 call, the tree descent of the
top classes, the judge; plus embedding scatters for the visited groups. usage: prep.py trace.jsonl"""
import ast, json, os, re, sys
import numpy as np
from nltk.corpus import wordnet as wn

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import generate as g

rows = [json.loads(l) for l in open(sys.argv[1])]
TASK, MODE = rows[0]["text"], rows[0]["mode"]; g.set_task(TASK); ROOT = g.ROOT
calls = [r for r in rows if r["kind"] in ("next", "judge")]; result = rows[-1]
judges = [c for c in calls if c["kind"] == "judge"]
NAME = {"not": "negation", "term": "task term"}
PUNCT_NAME = {"period": ".", "comma": ",", "question_mark": "?", "exclamation_mark": "!", "colon": ":"}

def opt_text(v): return ast.literal_eval(v.split(": ", 1)[1])
def label(crit, key):
    m = re.match(r"the next word is (.*?) \(for example", crit.get(key, ""))
    s = m.group(1) if m else key
    s = re.sub(r"^(a|an) ", "", re.sub(r"\s*\(.*\)$", "", s)).replace("word like ", "")
    return ", ".join(s.split(", ")[:2]) + " …" if ", " in s else s
def top(c, n=3): return [[label(c["crit"], k), round(c["p"][k], 3)] for k in sorted(c["p"], key=c["p"].get, reverse=True)[:n]]

def path_to(node, w):
    if "words" in node: return [] if w in node["words"] else None
    return next(([k] + p for k, c in node["children"].items() for p in [path_to(c, w)] if p is not None), None)

def has_call(pool, node):
    return any(set(c["p"]) == set(node["children"]) if "children" in node else set(c["p"]) <= set(node["words"]) for c in pool)

def descent(cls, pool, target):
    node, out, path = ROOT[cls], [], [cls]; way = path_to(node, target) or []
    while "children" in node:
        c = next((c for c in pool if set(c["p"]) == set(node["children"])), None)
        if c is None: return out
        pool.remove(c); w = way[len(path) - 1] if len(way) >= len(path) else None
        k = w if w and has_call(pool, node["children"][w]) else max(c["p"], key=c["p"].get); path.append(k)
        if k != w: way = []
        out.append({"level": "group" if all("words" in v for v in node["children"].values()) else "category", "cat": out[-1]["chosen"] if out else NAME.get(cls, cls),
                    "t0": c["t0"], "t1": c["t1"], "chosen": label(c["crit"], k), "p": round(c["p"][k], 3), "n": len(c["p"]), "top": top(c), "key": "/".join(path)})
        node = node["children"][k]
    c = next((c for c in pool if set(c["p"]) <= set(node["words"])), None)
    if c is None: return out
    pool.remove(c); k = target if target in c["p"] else max(c["p"], key=c["p"].get)
    out.append({"level": "word", "t0": c["t0"], "t1": c["t1"], "chosen": k, "p": round(c["p"][k], 3), "n": len(c["p"]), "top": top(c), "key": "/".join(path)})
    return out

answer = result["top"][0][1]
l1s = {c["text"]: c for c in calls if c["kind"] == "next" and "noun" in c["p"] and answer.startswith(c["text"])}
prefixes = sorted(l1s, key=len); steps = []
for i, text in enumerate(prefixes):
    l1 = l1s[text]; nt = prefixes[i + 1] if i + 1 < len(prefixes) else text; piece = nt[len(text):].strip() or None
    pool = [c for c in calls if c["kind"] == "next" and c["text"] == text and c is not l1 and c["t0"] >= l1["t1"]]
    order = sorted(l1["p"], key=l1["p"].get, reverse=True); shown = [k for k in order[:3] if k != "END"]
    cls = next((k for k, v in PUNCT_NAME.items() if v == piece), None) or next((k for k in ROOT if piece and path_to(ROOT[k], piece) is not None), None)
    if cls and cls not in shown: shown = shown[:2] + [cls]
    cols = []
    for k in shown:
        if k in PUNCT_NAME: cols.append({"cls": k, "name": PUNCT_NAME[k], "p": round(l1["p"][k], 3), "calls": [], "word": PUNCT_NAME[k]}); continue
        d = descent(k, pool, piece)
        cols.append({"cls": k, "name": NAME.get(k, k), "p": round(l1["p"][k], 3), "calls": d, "word": d[-1]["chosen"] if d and d[-1]["level"] == "word" else None})
    step = {"text": text, "l1": {"t0": l1["t0"], "t1": l1["t1"], "p": {NAME.get(k, PUNCT_NAME.get(k, k)): round(v, 3) for k, v in l1["p"].items()},
                                 "top": [NAME.get(k, PUNCT_NAME.get(k, k)) for k in shown]}, "cols": cols, "judge": None, "next": nt}
    if i and i - 1 < len(judges) - 1:
        j = judges[i - 1]; ranked = sorted(j["crit"], key=j["p"].get, reverse=True); keys = ranked[:3]
        me = next((k for k in j["crit"] if j["crit"][k].startswith("unfinished") and opt_text(j["crit"][k]) == nt), None)
        if me is not None and me not in keys: keys = sorted(ranked[:2] + [me], key=j["p"].get, reverse=True)
        step["judge"] = {"t0": j["t0"], "t1": j["t1"], "n": len(j["crit"]), "rank": ranked.index(me) + 1 if me is not None else None,
                         "top": [[opt_text(j["crit"][k]), round(j["p"][k], 3), j["crit"][k].startswith("unfinished")] for k in keys]}
    steps.append(step)
final = judges[-1]; ranked = sorted(final["crit"], key=final["p"].get, reverse=True)
final_judge = {"t0": final["t0"], "t1": final["t1"], "n": len(final["crit"]),
               "top": [[opt_text(final["crit"][k]), round(final["p"][k], 3), final["crit"][k].startswith("unfinished")] for k in ranked[:3]]}

# embedding scatters for the visited groups + word origins
emb = np.load("/tmp/jev_tree_emb.npz"); words, E = list(emb["words"]), emb["E"]
rank = {w: i for i, w in enumerate(words)}
scatter, origin = {}, {}
def node_at(key):
    n = ROOT[key.split("/")[0]]
    for k in key.split("/")[1:]: n = n["children"][k]
    return n
for s in steps:
    for c in s["cols"]:
        for call in c["calls"]:
            if call["level"] == "group":
                parent = node_at("/".join(call["key"].split("/")[:-1])); kids = list(parent["children"])
                call["group_idx"] = kids.index(call["key"].split("/")[-1])
            if call["level"] == "group" and call["key"] not in scatter:
                pts = [(w, b) for b, k in enumerate(kids) for w in parent["children"][k]["words"] if w in rank]
                rng = np.random.default_rng(0); pts = [pts[i] for i in rng.permutation(len(pts))[:360]]
                X = E[[rank[w] for w, _ in pts]]; X = X - X.mean(0); pc = np.linalg.svd(X, full_matrices=False)[2][:2]; Y = X @ pc.T
                Y = (Y - Y.min(0)) / (Y.max(0) - Y.min(0))
                scatter[call["key"]] = {"pts": [[round(float(x), 3), round(float(y), 3), b] for (w, b), (x, y) in zip(pts, Y)], "n": len(kids), "total": sum(len(parent["children"][k]["words"]) for k in kids)}
            if call["level"] == "word" and c["cls"] in ("noun", "verb", "adjective", "adverb"):
                for w, _ in call["top"]:
                    origin[w] = "wordnet" if wn.synsets(w) else "jina"
known = sum(1 for w in words if wn.synsets(w))
json.dump({"task": TASK, "mode": MODE, "answer": answer, "t_end": result["t"], "n_calls": len(calls), "classes": [NAME.get(k, k) for k in ROOT],
           "steps": steps, "final": final_judge, "t1s": sorted(c["t1"] for c in calls), "scatter": scatter, "origin": origin, "known": known, "unknown": len(words) - known},
          open(os.path.join(HERE, "data.json"), "w"))
print(len(steps), "steps", len(calls), "calls", f"{result['t']:.1f}s", repr(answer))
for s in steps[:6]: print(f"{s['l1']['t0']:6.1f} {s['text']!r:60} -> {[(c['name'], c['word']) for c in s['cols']]}")
