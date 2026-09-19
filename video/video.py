"""Real-time replay of one jev-gpt run on the real jev clock, portrait for phones: terminal, word-type call, tree descent
of the top classes, Jina embedding groups and the judge. Reads data.json, writes jev-gpt.mp4.
usage: video.py [out.mp4]  |  video.py frame <seconds> out.png"""
import bisect, json, math, multiprocessing, os, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))
W, H = 1080, 1920; L, R = 40, 1040; FPS = 30; T0 = 3.6; HOLD = 5.0
BG, FG, MUTED, DIM, LINE, CARD, TERM = (13, 16, 22), (242, 244, 247), (150, 158, 172), (66, 73, 88), (40, 46, 58), (23, 27, 36), (18, 21, 29)
JEV, JINA = (255, 176, 46), (32, 208, 198)
PAL = [(96, 140, 255), (255, 120, 120), (120, 220, 140), (230, 190, 90), (190, 120, 240), (90, 200, 230), (240, 140, 200), (160, 170, 120)]
INTER, MONO = "/usr/share/fonts/opentype/inter/", "/usr/share/fonts/truetype/liberation2/LiberationMono-"
LOGO = Image.open(os.path.join(HERE, "assets/jina_logo_white.png")).convert("RGBA")
CMD = f'$ jev-gpt "{D["task"]}"'
STEPS, FINAL = D["steps"], D["final"]
SLOTS = D["classes"] + ["END", ".", ",", ":", "?", "!"]
T_END = T0 + FINAL["t1"]; DUR = T_END + HOLD
_fonts = {}

def font(size, w="Regular", kind="text"):
    k = (size, w, kind)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(MONO + ("Bold" if w == "Bold" else "Regular") + ".ttf" if kind == "mono" else f"{INTER}Inter-{w}.otf", size)
    return _fonts[k]

def ease(x): x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)
def seg(t, a, b): return ease((t - a) / (b - a))
def mix(c, a, base=BG): return tuple(int(base[i] + (c[i] - base[i]) * a) for i in range(3))
def pulse(t): return 0.45 + 0.55 * (0.5 + 0.5 * math.sin(t * 7))
def fit(s, f, width):
    if f.getlength(s) <= width: return s
    while s and f.getlength(s + "…") > width: s = s[:-1].rstrip(" ,")
    return s + "…"
def tail(s, f, width):
    if f.getlength(s) <= width: return s
    while s and f.getlength("…" + s) > width: s = s[1:]
    return "…" + s
def wrap(s, f, width):
    lines, cur = [], ""
    for word in s.split(" "):
        t = (cur + " " + word).strip()
        if f.getlength(t) <= width or not cur: cur = t
        else: lines.append(cur); cur = word
    return lines + [cur]
def text(d, xy, s, size=24, fill=FG, w="Regular", kind="text", anchor="la", a=1.0, base=BG):
    if a > 0: d.text(xy, s, font=font(size, w, kind), fill=mix(fill, a, base), anchor=anchor)
def rrect(d, box, r=14, fill=CARD, outline=None, a=1.0, width=3):
    if a > 0: d.rounded_rectangle(box, r, fill=mix(fill, a) if fill else None, outline=mix(outline, a) if outline else None, width=width)
def dot(d, x, y, r, c, a=1.0, base=BG): d.ellipse((x - r, y - r, x + r, y + r), fill=mix(c, a, base))
def logo(img, x, y, width, a=1.0):
    im = LOGO.resize((width, int(width * LOGO.height / LOGO.width)))
    if a < 1: im.putalpha(im.split()[3].point(lambda v: int(v * a)))
    img.paste(im, (x, y), im)
def card(d, box, title, sub=None):
    rrect(d, box, 22, CARD, outline=LINE)
    text(d, (box[0] + 24, box[1] + 20), title, 28, FG, "SemiBold", base=CARD)
    if sub: text(d, (box[0] + 24 + font(28, "SemiBold").getlength(title) + 16, box[1] + 24), sub, 24, MUTED, base=CARD)
def head(d, y, title): text(d, (L, y), title, 24, MUTED, "SemiBold"); return L + font(24, "SemiBold").getlength(title) + 24
def asking(d, x, y, s, t, base=BG, size=26):
    dot(d, x + 8, y + size * 0.58, 7, JEV, pulse(t), base); text(d, (x + 26, y), s, size, MUTED, base=base)

# ---------------------------------------------------------------- run state
def step_at(r):
    i = bisect.bisect_right([s["l1"]["t0"] for s in STEPS], r) - 1
    return STEPS[i] if i >= 0 else None

def shown(r):
    if r >= FINAL["t1"]: return D["answer"], True
    s = step_at(r)
    if s is None: return "", False
    done = s["judge"]["t1"] if s["judge"] else max([c["t1"] for col in s["cols"] for c in col["calls"]] or [0])
    return (s["next"] if r >= done else s["text"]), False

def last_group(r):
    g = None
    for s in STEPS:
        if s["l1"]["t0"] > r: break
        for col in s["cols"]:
            for c in col["calls"]:
                if c["level"] == "group" and c["t0"] <= r: g = c
    return g

# ---------------------------------------------------------------- panels
def terminal(d, t, r):
    rrect(d, (L, 40, R, 400), 26, TERM, outline=LINE)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]): dot(d, 76 + i * 30, 74, 9, c, base=TERM)
    text(d, (540, 74), "terminal", 22, MUTED, "Regular", "mono", "mm", base=TERM)
    f = font(30, "Regular", "mono"); n = int(len(CMD) * seg(t, 0.4, 3.0) + 0.999) if t > 0.4 else 0
    lines = wrap(CMD[:n], f, 940); y = 110
    for i, l in enumerate(lines): text(d, (76, y + i * 40), l, 30, FG, "Regular", "mono", base=TERM)
    if t < T0:
        cx = 76 + f.getlength(lines[-1]) + 6; cy = y + (len(lines) - 1) * 40
        if (t % 1.0) < 0.6: d.rectangle((cx, cy - 2, cx + 17, cy + 33), fill=FG)
        return
    s, done = shown(r); f = font(36, "Regular", "mono"); y = 204
    lines = wrap(s, f, 940) if s else [""]
    for i, l in enumerate(lines): text(d, (76, y + i * 48), l, 36, JEV if done else FG, "Bold" if done else "Regular", "mono", base=TERM)
    cx = 76 + f.getlength(lines[-1]) + 6; cy = y + (len(lines) - 1) * 48
    if done: text(d, (76, y + len(lines) * 48), "$", 36, MUTED, "Regular", "mono", base=TERM)
    elif (t % 1.0) < 0.6: d.rectangle((cx, cy - 2, cx + 20, cy + 40), fill=FG)

def types(d, t, r, s):
    x0 = head(d, 432, "1  WORD TYPE")
    live = s is not None and r < s["l1"]["t1"]
    if live: asking(d, x0, 431, f"jev ranks {len(s['l1']['p'])} word types", t, size=24)
    elif s: text(d, (x0, 432), f"top 3 of {len(s['l1']['p'])} word types go on", 24, MUTED)
    src = (STEPS[STEPS.index(s) - 1] if s is not STEPS[0] else None) if live else s
    p, top = (src["l1"]["p"], src["l1"]["top"]) if src else ({}, [])
    a = 0.35 if live else seg(r, s["l1"]["t1"], s["l1"]["t1"] + 0.25) if s else 0
    for i, name in enumerate(SLOTS):
        x, y = L + (i % 4) * 254, 472 + (i // 4) * 66; box = (x, y, x + 238, y + 56)
        rank = top.index(name) if name in top else -1
        rrect(d, box, 14, CARD)
        if rank == 0: rrect(d, box, 14, JEV, a=a)
        elif rank > 0: rrect(d, box, 14, None, outline=JEV, a=a)
        base = mix(JEV, a, CARD) if rank == 0 else CARD
        fg = BG if rank == 0 and a > 0.5 else FG if rank >= 0 and not live else MUTED if name in p else DIM
        text(d, (x + 16, y + 28), name, 28, fg, "Medium" if rank >= 0 and not live else "Regular", anchor="lm", base=base)
        if name in p: text(d, (x + 222, y + 28), f"{p[name]:.2f}"[1:], 24, fg, "Medium", anchor="rm", a=0.5 if live else 1, base=base)

def chip(d, x, y, s, p, on, a=1.0):
    tw = font(28, "SemiBold").getlength(s) + font(22, "Medium").getlength(p) + 44
    rrect(d, (x, y, x + tw, y + 48), 14, JEV if on else CARD, outline=None if on else JEV, a=a)
    base = mix(JEV, a, CARD) if on else CARD
    text(d, (x + 16, y + 24), s, 28, BG if on else FG, "SemiBold", anchor="lm", a=a, base=base)
    text(d, (x + tw - 16, y + 25), p, 22, BG if on else JEV, "Medium", anchor="rm", a=a, base=base)
    return x + tw

def label(d, x, y, s, p, a):
    s = fit(s, font(28, "Medium"), 230); text(d, (x, y + 24), s, 28, FG, "Medium", anchor="lm", a=a); x += font(28, "Medium").getlength(s) + 8
    text(d, (x, y + 25), p, 22, JEV, "Medium", anchor="lm", a=a); return x + font(22, "Medium").getlength(p)

def chain(d, t, r, col, y, rank):
    x = chip(d, L, y, col["name"], f"{col['p']:.2f}"[1:], rank == 0)
    calls = col["calls"] or [{"level": "word", "t0": 0, "t1": 0, "chosen": col["word"], "p": 1.0, "n": 1, "top": [[col["word"], 1.0]], "punct": True}]
    subs = []
    for c in calls:
        if r < c["t0"]: break
        d.line((x + 4, y + 24, x + 22, y + 24), fill=LINE, width=3); x += 28
        if r < c["t1"]:
            asking(d, x, y + 9, fit(f"jev picks 1 of {c['n']} {dict(category='categories', group='groups', word='words')[c['level']]}", font(26), R - x - 30), t); break
        fa = seg(r, c["t1"], c["t1"] + 0.25); p = f"{c['p']:.2f}"[1:]
        if c["level"] == "word":
            x = chip(d, x, y, c["chosen"], p, True, fa)
            nxt = [f"{w2} " + f"{p2:.2f}"[1:] for w2, p2 in c["top"] if w2 != c["chosen"]][:2]
            subs.append(("punctuation, no lookup" if c.get("punct") else "next: " + " · ".join(nxt) if nxt else f"{c['n']} words", MUTED, fa))
        else:
            x = label(d, x, y, c["chosen"], p, fa)
            if c["level"] == "group": subs += [(f"group {c['group_idx'] + 1} of {c['n']}", MUTED, fa), ("split by jina v5", JINA, fa)]
    sx = L
    for k, (s, c, fa) in enumerate(subs):
        if k: text(d, (sx, y + 58), "·", 24, MUTED, a=fa); sx += 22
        w = "Medium" if c is JINA else "Regular"; text(d, (sx, y + 58), s, 24, c, w, a=fa); sx += font(24, w).getlength(s) + 10

def columns(d, t, r, s):
    x0 = head(d, 824, "2  WORD TREE"); text(d, (x0, 824), "category → group of ~200 → word, one jev call each", 24, MUTED)
    if s is not None and r < s["l1"]["t1"]: s = STEPS[STEPS.index(s) - 1] if s is not STEPS[0] else None
    if s is None: return
    for i, col in enumerate(s["cols"]): chain(d, t, r, col, 868 + i * 100, i)

def jina(img, d, t, r, s):
    y0 = 1186; card(d, (L, y0, R, y0 + 304), "")
    logo(img, 64, y0 + 24, 72); text(d, (150, y0 + 22), "jina-embeddings-v5-text-small", 28, FG, "Medium", base=CARD)
    text(d, (150, y0 + 58), f"built the word tree · {D['known'] + D['unknown']:,} words", 24, MUTED, base=CARD)
    g = last_group(r); sx, sy, sw, sh = 64, y0 + 110, 400, 166
    if g is None:
        for i, (s, c) in enumerate([(f"{D['known']:,} words placed by WordNet", MUTED), (f"{D['unknown']:,} unknown words placed next to their 5 nearest", MUTED),
                                    ("neighbours in v5 embedding space", MUTED), ("groups of ~200 words appear here when jev reaches", DIM), ("a large word category", DIM)]):
            text(d, (64, y0 + 110 + i * 34), s, 24, c, base=CARD)
        return
    sc = D["scatter"][g["key"]]; picked = r >= g["t1"]; fa = seg(r, g["t0"], g["t0"] + 0.4)
    for x, y, b in sc["pts"]:
        hit = picked and b == g["group_idx"]
        dot(d, sx + x * sw, sy + (1 - y) * sh, 5 if hit else 3, JINA if hit else PAL[b % 8], fa * (1.0 if hit else 0.3 if picked else 0.55), CARD)
    cx = 500; text(d, (cx, y0 + 106), fit(g["cat"], font(28, "SemiBold"), 520), 28, FG, "SemiBold", base=CARD)
    text(d, (cx, y0 + 144), f"{sc['total']:,} words · too many for one", 24, MUTED, base=CARD)
    text(d, (cx, y0 + 176), "jev call (max 255)", 24, MUTED, base=CARD)
    text(d, (cx, y0 + 208), f"v5 splits them into {sc['n']} groups", 24, JINA, "Medium", base=CARD)
    if not picked: asking(d, cx, y0 + 240, "jev picks a group", t, CARD, 24)
    else:
        s2 = f"jev picks group {g['group_idx'] + 1}"; text(d, (cx, y0 + 240), s2, 24, FG, "Medium", base=CARD)
        text(d, (cx + font(24, "Medium").getlength(s2) + 10, y0 + 240), f"{g['p']:.2f}"[1:], 24, JEV, "Medium", base=CARD)
    w = s and next((c["chosen"] for col in s["cols"] for c in col["calls"] if c["level"] == "word" and c["t1"] <= r and D["origin"].get(c["chosen"]) == "jina"), None)
    if w: text(d, (cx, y0 + 272), fit(f"“{w}” is unknown to WordNet: placed by v5", font(22, "Medium"), 520), 22, JINA, "Medium", base=CARD)

def judge_rows(d, rows, y, t1, r, cont):
    fa = seg(r, t1, t1 + 0.25); f = font(26, "Regular", "mono")
    for i, (s, p, alive) in enumerate(rows):
        yy = y + i * 80; hit = s == cont
        text(d, (64, yy), tail(s, f, 890), 26, FG if hit else MUTED, "Bold" if hit else "Regular", "mono", a=fa, base=CARD)
        text(d, (1016, yy), f"{p:.2f}"[1:], 26, JEV if hit else MUTED, "Medium", anchor="ra", a=fa, base=CARD)
        d.rounded_rectangle((64, yy + 38, 1016, yy + 45), 3, fill=mix(LINE, fa, CARD))
        d.rounded_rectangle((64, yy + 38, 64 + max(8, 952 * p), yy + 45), 3, fill=mix(JEV if hit else DIM, fa, CARD))
        if not alive: text(d, (1016, yy + 50), "finished", 20, MUTED, anchor="ra", a=fa, base=CARD)

def openers(d, y, r):
    text(d, (64, y), "3 openers, one per word type:", 24, MUTED, base=CARD)
    for i, col in enumerate(STEPS[0]["cols"]):
        c = col["calls"][-1] if col["calls"] else None; fa = seg(r, c["t1"], c["t1"] + 0.25) if c else 1
        text(d, (64, y + 44 + i * 44), col["word"] or "", 28, FG, "Regular", "mono", a=fa, base=CARD)

def judge(d, t, r, s):
    y0 = 1520; card(d, (L, y0, R, y0 + 340), "3  JUDGE", "one jev call ranks the candidate texts")
    if s is None: return
    j = s["judge"]
    if r >= FINAL["t0"] and s is STEPS[-1] and (j is None or r >= j["t1"]): j = FINAL
    if j is None: openers(d, y0 + 76, r); return
    if r < j["t0"]:
        prev = STEPS[STEPS.index(s) - 1]["judge"] if STEPS.index(s) > 0 else None
        if prev: judge_rows(d, prev["top"], y0 + 72, 0, r, s["text"]); text(d, (64, y0 + 306), "the terminal follows the highlighted text", 22, DIM, base=CARD)
        else: openers(d, y0 + 76, r)
        return
    if r < j["t1"]: asking(d, 64, y0 + 84, f"ranking {j['n']} candidate texts", t, CARD); return
    judge_rows(d, j["top"], y0 + 72, j["t1"], r, D["answer"] if j is FINAL else s["next"])
    text(d, (64, y0 + 306), "final answer" if j is FINAL else "the terminal follows the highlighted text", 22, JEV if j is FINAL else DIM, "Medium" if j is FINAL else "Regular", base=CARD)

def footer(d, t, r):
    n = bisect.bisect_right(D["t1s"], r); rr = max(0.0, min(r, FINAL["t1"]))
    text(d, (L, 1872), f"jev · TypeSafe   {n} jev calls   {rr:5.1f} s", 24, MUTED, "Regular", "mono")
    text(d, (R, 1872), "real time, 1×", 24, DIM, "Medium", anchor="ra")
    d.rounded_rectangle((L, 1904, R, 1908), 2, fill=LINE); d.rounded_rectangle((L, 1904, L + 1000 * min(1, t / DUR), 1908), 2, fill=JEV)

def frame(t):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img); r = t - T0; s = step_at(r)
    terminal(d, t, r); types(d, t, r, s); columns(d, t, r, s); jina(img, d, t, r, s); judge(d, t, r, s); footer(d, t, r)
    return img

def render(part):
    a, b, out = part
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(a, b): ff.stdin.write(frame(i / FPS).tobytes())
    ff.stdin.close(); ff.wait()

if __name__ == "__main__":
    if sys.argv[1:2] == ["frame"]: frame(float(sys.argv[2])).save(sys.argv[3]); sys.exit()
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "jev-gpt.mp4"); n = int(DUR * FPS); k = int(os.environ.get("JOBS", 1))
    parts = [(i * n // k, (i + 1) * n // k, f"/tmp/jev-seg{i}.mp4") for i in range(k)]
    with multiprocessing.Pool(k) as pool: pool.map(render, parts)
    open("/tmp/jev-segs.txt", "w").write("".join(f"file '{p}'\n" for _, _, p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "/tmp/jev-segs.txt", "-c", "copy", "-movflags", "+faststart", out], check=True)
    print(out, f"{DUR:.1f}s")
