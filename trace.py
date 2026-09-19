"""Run jev-gpt and record every jev call with timestamps (for the real-time video).
usage: trace.py "<task>" out.jsonl [greedy|beam|guided] [steps | "<target text>"]
guided = real calls, real timing, but the beam is steered to reproduce the target text; steps that needed steering are reported."""
import json, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import generate as g

g.set_task(sys.argv[1]); out = open(sys.argv[2], "w"); lock = threading.Lock(); _ask = g.ask; T0 = time.time()
mode = sys.argv[3] if len(sys.argv) > 3 else "beam"
out.write(json.dumps({"kind": "task", "text": sys.argv[1], "mode": mode}) + "\n")

def ask(state, crit, instr=g.INSTR):
    t0 = time.time() - T0; p = _ask(state, crit, instr)
    with lock:
        out.write(json.dumps({"kind": "judge" if instr is g.JUDGE else "next", "t0": round(t0, 3), "t1": round(time.time() - T0, 3),
                              "text": state.get("text_so_far"), "crit": crit, "p": p}) + "\n"); out.flush()
    return p

g.ask = ask

def path_to(node, w):
    if "words" in node: return [] if w in node["words"] else None
    return next(([k] + p for k, c in node["children"].items() for p in [path_to(c, w)] if p is not None), None)

def force(text, piece):
    if piece in g.PUNCT.values(): return
    cls, path = next((k, p) for k in g.ROOT for p in [path_to(g.ROOT[k], piece)] if p is not None)
    node = g.ROOT[cls]
    for k in path: g.expand(text, node); node = node["children"][k]
    g.expand(text, node)

def guided(target, width=3):
    alive, done, text, forced = [""], [], "", []
    for i, piece in enumerate(re.findall(r"[A-Za-z]+|[.,?!:]", target) + ["END"]):
        with ThreadPoolExecutor(width) as ex: cands = list(ex.map(g.candidates, alive))
        nxt = [g.extend(t, c) for t, gs in zip(alive, cands) for c in (c for gr in gs for c in gr) if c != "END"]
        done += [t for t, gs in zip(alive, cands) if ["END"] in gs]
        want = text if piece == "END" else g.extend(text, piece)
        if want not in (done if piece == "END" else nxt):
            forced.append(i); force(text, piece); (done if piece == "END" else nxt).append(want)
        if i == 0: alive, text = [want] + [t for t in nxt if t != want][:width - 1], want; continue
        ranked, p = g.judge(list(dict.fromkeys(nxt + done)), done)
        alive = [t for t in ranked if t not in done]; done = [t for t in ranked if t in done]
        if piece == "END": done = ([want] + [t for t in done if t != want])[:width]; alive = alive[:width]
        else: alive = ([want] + [t for t in alive if t != want])[:width]; done = done[:width]; text = want
        print(f"\r\033[K{i:3d} {text}", end="", file=sys.stderr)
    ranked, p = g.judge(alive + done, done)
    return [(p[str((alive + done).index(t))], t) for t in ranked[:3]], forced

if mode == "guided":
    top, forced = guided(sys.argv[4]); print(f"steered at steps {forced}", file=sys.stderr)
else:
    top = g.beam("", int(sys.argv[4]) if len(sys.argv) > 4 else 25, 3 if mode == "beam" else 1)
out.write(json.dumps({"kind": "result", "t": round(time.time() - T0, 3), "top": top}) + "\n")
print(top[0][1])
