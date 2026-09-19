"""Text generation with jev: task in, words out. One small concrete choice per tree level, a leaf choice over full strings,
then jev judges the candidate strings against each other (greedy = width 1, beam = width 3).
usage: generate.py "<task>" [greedy|beam|sample] [steps] [prefix]"""
import json, os, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ["TYPESAFE_API_KEY"]
URL = "https://api.typesafe.ai/v1/systemone"
ROOT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/pos_tree.json")))
PUNCT = {"period": ".", "comma": ",", "question_mark": "?", "exclamation_mark": "!", "colon": ":"}
INSTR = ("`text_so_far` is the beginning of the response to `task`. The response is written in complete, grammatical sentences, "
         "each with a subject. Which option describes the best, most natural continuation of the response? Choose END only when the response is complete.")
JUDGE = ("Each option is a candidate response to `task`; some are still unfinished. A good response is written in complete, grammatical "
         "sentences, each with a subject. Which option is the best, most natural and correct response, or the best beginning of one?")
TASK = ""
ME = ("You are Jev, the System One model made by TypeSafe. You do not generate text; you answer typed questions "
      "with calibrated probabilities, fast and cheap, and right now you are writing this response one word at a time.")
FUNC_WORDS = {w for n in ROOT.values() if "words" in n for w in n["words"]}

def set_task(task):
    global TASK; TASK = task
    ws = list({w.lower(): w for w in re.findall("[A-Za-z]+", task) if w.lower() not in FUNC_WORDS and len(w) > 1}.values())
    ROOT["term"] = {"desc": "a term or name taken from the task", "ex": ws[:3], "words": ws}

def extend(text, piece):
    return text + piece if piece in PUNCT.values() or text.endswith(" ") or not text else text + " " + piece

def ask(state, crit, instr=INSTR):
    body = {"model": "jev-latest", "state": {"task": TASK, "about_you": ME} | state,
            "questions": {"next": {"type": "choice", "instructions": instr + " `about_you` says who writes the response.", "criteria": crit}}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["answers"]["next"]["probabilities"]

def node_crit(text, kids):
    return {k: f"the next word is {n['desc']} (for example {', '.join(n['ex'])}), as in {extend(text, n['ex'][0])!r}" for k, n in kids.items()}

def expand(text, node):
    if "children" in node:
        return ask({"text_so_far": text}, node_crit(text, node["children"]))
    used = {w.lower() for w in text.replace(":", " ").split()}; last = text.split()[-1] if text.split() else ""
    words = [w for w in node["words"] if w.lower() not in used and (last != "a" or w[0].lower() not in "aeiou") and (last != "an" or w[0].lower() in "aeiou")] or node["words"]
    return ask({"text_so_far": text}, {w: f"the response continues as: {extend(text, w)!r}" for w in words})

def level1(text):
    crit = node_crit(text, ROOT)
    if text and text[-1] not in ".,?!:" and text.split()[-1] not in ROOT["determiner"]["words"] + ROOT["preposition"]["words"]:
        crit |= {k: f"the response continues as: {extend(text, v)!r}" for k, v in PUNCT.items()}
    if text and text[-1] in ".?!":
        crit["END"] = f"the response is finished: {text!r}"
    return ask({"text_so_far": text}, crit)

def pick(d, mode):
    return random.choices(list(d), list(d.values()))[0] if mode == "sample" else max(d, key=d.get)

def next_word(text, mode):
    p1 = level1(text); k = pick(p1, mode); trace = [f"{k}={p1[k]:.2f}"]
    if k == "END" or k in PUNCT: return k, trace[0]
    node = ROOT[k]
    while "children" in node:
        d = expand(text, node); k = pick(d, mode); node = node["children"][k]; trace.append(f"{k}={d[k]:.2f}")
    d = expand(text, node); w = pick(d, mode)
    return w, " > ".join(trace) + f" > {w}={d[w]:.3f}"

def candidates(text):
    p1 = level1(text); groups = [["END"]] if "END" in p1 else []
    for k in sorted(p1, key=p1.get, reverse=True)[:3]:
        if k == "END": continue
        if k in PUNCT: groups.append([PUNCT[k]]); continue
        node = ROOT[k]
        while "children" in node:
            d = expand(text, node); node = node["children"][max(d, key=d.get)]
        d = expand(text, node); groups.append(sorted(d, key=d.get, reverse=True)[:3])
    return groups

def judge(pool, done):
    p = ask({}, {str(j): ("finished response: " if t in done else "unfinished response: ") + repr(t) for j, t in enumerate(pool)}, JUDGE)
    return sorted(pool, key=lambda t: -p[str(pool.index(t))]), p

def beam(text, steps, width=3):
    alive, done = [text], []
    for i in range(steps):
        with ThreadPoolExecutor(width) as ex:
            cands = list(ex.map(candidates, alive))
        nxt = []
        for t, gs in zip(alive, cands):
            for c in (c for g in gs for c in g):
                if c == "END": done.append(t)
                else: nxt.append(extend(t, c))
        if i == 0:  # the judge is blind on one-word strings: open with the top word of each of the top classes
            alive = [extend(text, g[0]) for g in cands[0] if g[0] != "END"][:width]; continue
        ranked, p = judge(list(dict.fromkeys(nxt + done)), done)
        alive = [t for t in ranked if t not in done][:width]
        done = [t for t in ranked if t in done][:width]
        print(f"\r\033[K{i:3d} {alive[0] if alive else done[0]}", end="", file=sys.stderr)
        if not alive: break
    print("\r\033[K", end="", file=sys.stderr)
    ranked, p = judge(alive + done, done)
    return [(p[str((alive + done).index(t))], t) for t in ranked[:3]]

if __name__ == "__main__":
    set_task(sys.argv[1])
    MODE = sys.argv[2] if len(sys.argv) > 2 else "greedy"
    STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    text = sys.argv[4] if len(sys.argv) > 4 else ""
    random.seed(0)
    if MODE != "sample":
        for p, t in beam(text, STEPS, 3 if MODE == "beam" else 1): print(f"{p:.2f} {t!r}")
        sys.exit()
    for i in range(STEPS):
        name, note = next_word(text, MODE)
        print(f"{i:3d} {name:14s} {note}", file=sys.stderr)
        if name == "END": break
        text = extend(text, PUNCT.get(name, name))
    print(repr(text))
