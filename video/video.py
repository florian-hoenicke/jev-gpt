"""Real-time replay of one jev-gpt run on the real jev clock, portrait for phones: terminal, word tree of the top
classes, Jina embedding groups and the judge. Reads data.json, writes jev-gpt.mp4.
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
def text(d, xy, s, size=32, fill=FG, w="Regular", kind="text", anchor="la", a=1.0, base=BG):
    if a > 0: d.text(xy, s, font=font(size, w, kind), fill=mix(fill, a, base), anchor=anchor)
def rrect(d, box, r=14, fill=CARD, outline=None, a=1.0, width=3):
    if a > 0: d.rounded_rectangle(box, r, fill=mix(fill, a) if fill else None, outline=mix(outline, a) if outline else None, width=width)
def dot(d, x, y, r, c, a=1.0, base=BG): d.ellipse((x - r, y - r, x + r, y + r), fill=mix(c, a, base))
def logo(img, x, y, width, a=1.0):
    im = LOGO.resize((width, int(width * LOGO.height / LOGO.width)))
    if a < 1: im.putalpha(im.split()[3].point(lambda v: int(v * a)))
    img.paste(im, (x, y), im)
def card(d, box, title, sub=None):
    rrect(d, box, 24, CARD, outline=LINE)
    text(d, (box[0] + 24, box[1] + 22), title, 36, FG, "SemiBold", base=CARD)
    if sub: text(d, (box[0] + 24 + font(36, "SemiBold").getlength(title) + 18, box[1] + 27), sub, 32, MUTED, base=CARD)
def head(d, y, title): text(d, (L, y), title, 36, MUTED, "SemiBold"); return L + font(36, "SemiBold").getlength(title) + 24
def asking(d, x, y, s, t, base=BG, size=34):
    dot(d, x + 9, y + size * 0.58, 8, JEV, pulse(t), base); text(d, (x + 30, y), s, size, MUTED, base=base)

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
    rrect(d, (L, 40, R, 462), 28, TERM, outline=LINE)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]): dot(d, 78 + i * 32, 76, 10, c, base=TERM)
    text(d, (540, 76), "terminal", 26, MUTED, "Regular", "mono", "mm", base=TERM)
    f = font(36, "Regular", "mono"); n = int(len(CMD) * seg(t, 0.4, 3.0) + 0.999) if t > 0.4 else 0
    lines = wrap(CMD[:n], f, 940); y = 104
    for i, l in enumerate(lines): text(d, (76, y + i * 46), l, 36, FG if t < T0 else MUTED, "Regular", "mono", base=TERM)
    if t < T0:
        cx = 76 + f.getlength(lines[-1]) + 6; cy = y + (len(lines) - 1) * 46
        if (t % 1.0) < 0.6: d.rectangle((cx, cy - 2, cx + 20, cy + 38), fill=FG)
        return
    s, done = shown(r); f = font(46, "Regular", "mono"); y = 214
    lines = wrap(s, f, 940) if s else [""]
    for i, l in enumerate(lines): text(d, (76, y + i * 58), l, 46, JEV if done else FG, "Bold" if done else "Regular", "mono", base=TERM)
    if not done and (t % 1.0) < 0.6:
        cx = 76 + f.getlength(lines[-1]) + 6; cy = y + (len(lines) - 1) * 58; d.rectangle((cx, cy - 2, cx + 26, cy + 50), fill=FG)

def chip(d, x, y, s, p, on, a=1.0):
    tw = font(40, "SemiBold").getlength(s) + font(30, "Medium").getlength(p) + 52
    rrect(d, (x, y, x + tw, y + 64), 16, JEV if on else CARD, outline=None if on else JEV, a=a)
    base = mix(JEV, a, CARD) if on else CARD
    text(d, (x + 18, y + 32), s, 40, BG if on else FG, "SemiBold", anchor="lm", a=a, base=base)
    text(d, (x + tw - 18, y + 33), p, 30, BG if on else JEV, "Medium", anchor="rm", a=a, base=base)
    return x + tw

def label(d, x, y, s, p, a, width):
    s = fit(s, font(40, "Medium"), width - 80); text(d, (x, y + 32), s, 40, FG, "Medium", anchor="lm", a=a)
    text(d, (x + font(40, "Medium").getlength(s) + 10, y + 33), p, 30, JEV, "Medium", anchor="lm", a=a)

def link(d, x, y): d.line((x + 4, y + 32, x + 24, y + 32), fill=LINE, width=3); return x + 30
def elbow(d, y, x2): d.line((L + 32, y + 66, L + 32, y + 106, x2 - 8, y + 106), fill=LINE, width=3)

def chain(d, t, r, col, y, rank):
    x = chip(d, L, y, col["name"], f"{col['p']:.2f}"[1:], rank == 0); y2 = y + 74; x2 = L + 56
    if not col["calls"]: elbow(d, y, x2); text(d, (x2, y2 + 14), "punctuation, no lookup", 34, MUTED); return
    for c in col["calls"]:
        if r < c["t0"]: break
        top = c["level"] == "category"
        if top: x = link(d, x, y)
        else: elbow(d, y, x2)
        ax, ay = (x, y) if top else (x2, y2)
        if r < c["t1"]:
            what = dict(category="categories", group="groups", word="words")[c["level"]]
            asking(d, ax, ay + 14, fit(f"jev picks 1 of {c['n']} {what}", font(34), R - ax - 40), t); break
        fa = seg(r, c["t1"], c["t1"] + 0.25); p = f"{c['p']:.2f}"[1:]
        if top: label(d, x, y, c["chosen"], p, fa, R - x)
        elif c["level"] == "word":
            xe = chip(d, x2, y2, c["chosen"], p, True, fa)
            nxt = [f"{w2} " + f"{p2:.2f}"[1:] for w2, p2 in c["top"] if w2 != c["chosen"]][:2]
            if nxt: text(d, (xe + 20, y2 + 14), fit("next: " + " · ".join(nxt), font(34), R - xe - 30), 34, MUTED, a=fa)

def tree(d, t, r, s):
    x0 = head(d, 496, "1  WORD TREE")
    if s is not None and r < s["l1"]["t1"]:
        asking(d, x0, 498, f"jev ranks {len(s['l1']['p'])} word types", t)
        s = STEPS[STEPS.index(s) - 1] if s is not STEPS[0] else None
    else: text(d, (x0, 499), "top 3 types → category → word", 34, MUTED)
    if s is None: return
    for i, col in enumerate(s["cols"]): chain(d, t, r, col, 552 + i * 132, i)

def jina(img, d, t, r, s):
    y0 = 980; card(d, (L, y0, R, y0 + 320), "")
    logo(img, 64, y0 + 22, 64); text(d, (144, y0 + 20), "jina-embeddings-v5-text-small", 34, FG, "Medium", base=CARD)
    text(d, (144, y0 + 62), f"built the word tree · {D['known'] + D['unknown']:,} words", 32, MUTED, base=CARD)
    g = last_group(r); sx, sy, sw, sh = 64, y0 + 116, 360, 176
    if g is None:
        for i, (s_, c) in enumerate([(f"{D['known']:,} words placed by WordNet", MUTED), (f"{D['unknown']:,} unknown words placed next to their", MUTED),
                                     ("5 nearest neighbours in v5 space", MUTED), ("groups appear when jev reaches a large category", DIM)]):
            text(d, (64, y0 + 112 + i * 44), s_, 32, c, base=CARD)
        return
    sc = D["scatter"][g["key"]]; picked = r >= g["t1"]; fa = seg(r, g["t0"], g["t0"] + 0.4)
    for x, y, b in sc["pts"]:
        hit = picked and b == g["group_idx"]
        dot(d, sx + x * sw, sy + (1 - y) * sh, 6 if hit else 4, JINA if hit else PAL[b % 8], fa * (1.0 if hit else 0.3 if picked else 0.55), CARD)
    cx = 456; text(d, (cx, y0 + 108), fit(g["cat"], font(34, "SemiBold"), 560), 34, FG, "SemiBold", base=CARD)
    text(d, (cx, y0 + 152), f"{sc['total']:,} words · max 255 per jev call", 32, MUTED, base=CARD)
    text(d, (cx, y0 + 192), f"v5 splits them into {sc['n']} groups", 32, JINA, "Medium", base=CARD)
    if not picked: asking(d, cx, y0 + 232, "jev picks a group", t, CARD, 32)
    else:
        s2 = f"jev picks group {g['group_idx'] + 1}"; text(d, (cx, y0 + 232), s2, 32, FG, "Medium", base=CARD)
        text(d, (cx + font(32, "Medium").getlength(s2) + 12, y0 + 232), f"{g['p']:.2f}"[1:], 32, JEV, "Medium", base=CARD)
    w = s and next((c["chosen"] for col in s["cols"] for c in col["calls"] if c["level"] == "word" and c["t1"] <= r and D["origin"].get(c["chosen"]) == "jina"), None)
    if w: text(d, (cx, y0 + 272), fit(f"“{w}”: not in WordNet, placed by v5", font(30, "Medium"), 560), 30, JINA, "Medium", base=CARD)

def judge_rows(d, rows, y, t1, r, cont):
    fa = seg(r, t1, t1 + 0.25); f = font(34, "Regular", "mono")
    for i, (s, p, alive) in enumerate(rows):
        yy = y + i * 112; hit = s == cont
        text(d, (64, yy), tail(s, f, 860), 34, FG if hit else MUTED, "Bold" if hit else "Regular", "mono", a=fa, base=CARD)
        text(d, (1016, yy), f"{p:.2f}"[1:], 34, JEV if hit else MUTED, "Medium", anchor="ra", a=fa, base=CARD)
        d.rounded_rectangle((64, yy + 50, 1016, yy + 58), 4, fill=mix(LINE, fa, CARD))
        d.rounded_rectangle((64, yy + 50, 64 + max(8, 952 * p), yy + 58), 4, fill=mix(JEV if hit else DIM, fa, CARD))
        if not alive: text(d, (1016, yy + 64), "finished", 26, MUTED, anchor="ra", a=fa, base=CARD)

def openers(d, y, r):
    text(d, (64, y), "3 openers, one per word type:", 34, MUTED, base=CARD)
    for i, col in enumerate(STEPS[0]["cols"]):
        c = col["calls"][-1] if col["calls"] else None; fa = seg(r, c["t1"], c["t1"] + 0.25) if c else 1
        text(d, (64, y + 54 + i * 54), col["word"] or "", 36, FG, "Regular", "mono", a=fa, base=CARD)

def judge(d, t, r, s):
    y0 = 1324; card(d, (L, y0, R, y0 + 522), "2  JUDGE", "one jev call ranks the candidates")
    if s is None: return
    j = s["judge"]
    if r >= FINAL["t0"] and s is STEPS[-1] and (j is None or r >= j["t1"]): j = FINAL
    if j is None: openers(d, y0 + 90, r); return
    if r < j["t0"]:
        prev = STEPS[STEPS.index(s) - 1]["judge"] if STEPS.index(s) > 0 else None
        if prev: judge_rows(d, prev["top"], y0 + 88, 0, r, s["text"]); text(d, (64, y0 + 440), "the terminal follows the highlighted text", 32, DIM, base=CARD)
        else: openers(d, y0 + 90, r)
        return
    if r < j["t1"]: asking(d, 64, y0 + 100, f"ranking {j['n']} candidate texts", t, CARD); return
    judge_rows(d, j["top"], y0 + 88, j["t1"], r, D["answer"] if j is FINAL else s["next"])
    text(d, (64, y0 + 440), "final answer" if j is FINAL else "the terminal follows the highlighted text", 32, JEV if j is FINAL else DIM, "Medium" if j is FINAL else "Regular", base=CARD)

def footer(d, t, r):
    n = bisect.bisect_right(D["t1s"], r); rr = max(0.0, min(r, FINAL["t1"]))
    text(d, (L, 1862), f"jev · TypeSafe   {n} jev calls   {rr:5.1f} s", 32, MUTED, "Regular", "mono")
    text(d, (R, 1862), "real time, 1×", 32, DIM, "Medium", anchor="ra")
    d.rounded_rectangle((L, 1906, R, 1912), 3, fill=LINE); d.rounded_rectangle((L, 1906, L + 1000 * min(1, t / DUR), 1912), 3, fill=JEV)

def frame(t):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img); r = t - T0; s = step_at(r)
    terminal(d, t, r); tree(d, t, r, s); jina(img, d, t, r, s); judge(d, t, r, s); footer(d, t, r)
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
