"""Word tree for jev: part of speech -> WordNet category -> embedding bucket (only when > 255 words) -> words.
Words WordNet does not know take the class of their nearest tagged neighbours in jina-embeddings-v5-text-small space."""
import json, os, re, urllib.request
import numpy as np, wordfreq
from nltk.corpus import wordnet as wn
from concurrent.futures import ThreadPoolExecutor

N, LEAF = 60000, 200
AI = "Jev Claude GPT ChatGPT Gemini Llama OpenAI Anthropic TypeSafe Jina LLM LLMs AI AGI".split()
base = [w for w in wordfreq.top_n_list("en", 120000) if re.fullmatch("[a-z]+", w) and (len(w) > 1 or w in "ai")][:N]
def embed(batch):
    body = {"model": "jina-embeddings-v5-text-small", "task": "text-matching", "input": batch}
    req = urllib.request.Request("https://api.jina.ai/v1/embeddings", json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {os.environ['JINA_API_KEY']}", "Content-Type": "application/json"})
    return [d["embedding"] for d in json.load(urllib.request.urlopen(req))["data"]]
cache = "/tmp/jev_vocab_emb_v5.npy"
if not os.path.exists(cache):
    with ThreadPoolExecutor(8) as ex:
        np.save(cache, np.array(sum(ex.map(embed, [base[i:i + 500] for i in range(0, N, 500)]), []), dtype=np.float32))
keep = [i for i, w in enumerate(base) if w not in {a.lower() for a in AI}]
words = [base[i] for i in keep] + AI
E = np.concatenate([np.load(cache)[keep], np.array(embed(AI), dtype=np.float32)]); E /= np.linalg.norm(E, axis=1, keepdims=True)
np.savez("/tmp/jev_tree_emb.npz", words=words, E=E)
rank = {w: i for i, w in enumerate(words)}

FUNC = {
 "pronoun": ("a pronoun", "i me my mine myself you your yours yourself he him his himself she her hers herself it its itself we us our ours ourselves they them their theirs themselves who whom whose which what that this these those someone somebody something anyone anybody anything everyone everybody everything nobody nothing none one how why"),
 "determiner": ("an article or determiner", "the a an this that these those some any no every each either neither much many more most few little several all both half another other such"),
 "preposition": ("a preposition", "of in to for with on at from by about as into like through after over between out against during without before under around among across behind beyond within upon toward towards near off above below down up inside outside since until per via"),
 "conjunction": ("a conjunction", "and or but so because although though while if unless when whenever where wherever whereas than whether nor yet once"),
 "auxiliary": ("an auxiliary or modal verb", "am is are was were be been being have has had having do does did can could will would shall should may might must"),
 "number": ("a number", "zero one two three four five six seven eight nine ten eleven twelve twenty thirty fifty hundred thousand million billion first second third last next"),
 "interjection": ("an interjection", "oh wow hey haha yes no well okay oops damn please thanks"),
 "not": ("a negation or degree word", "not never very too quite so also just only even still almost really"),
}
DESC = {"noun.Tops": "a very general noun", "noun.act": "an act or activity", "noun.animal": "an animal", "noun.artifact": "a man-made object",
 "noun.attribute": "an attribute or quality", "noun.body": "a body part", "noun.cognition": "a thought, idea or field of knowledge",
 "noun.communication": "a word, message or piece of communication", "noun.event": "an event", "noun.feeling": "a feeling or emotion",
 "noun.food": "a food or drink", "noun.group": "a group or organisation", "noun.location": "a place", "noun.motive": "a motive or goal",
 "noun.object": "a natural object", "noun.person": "a person or role", "noun.phenomenon": "a phenomenon", "noun.plant": "a plant",
 "noun.possession": "money or property", "noun.process": "a process", "noun.quantity": "a quantity or measure", "noun.relation": "a relation",
 "noun.shape": "a shape", "noun.state": "a state or condition", "noun.substance": "a substance or material", "noun.time": "a time or period",
 "verb.body": "a verb of the body (eat, sleep, wear)", "verb.change": "a verb of change (grow, break, fix)", "verb.cognition": "a verb of thinking (know, think, learn)",
 "verb.communication": "a verb of speaking or writing (say, ask, publish)", "verb.competition": "a verb of competing (win, fight, play)",
 "verb.consumption": "a verb of using or consuming (use, spend, eat)", "verb.contact": "a verb of touching or handling (put, hold, cut)",
 "verb.creation": "a verb of making (make, build, write)", "verb.emotion": "a verb of feeling (love, hate, fear)", "verb.motion": "a verb of movement (go, run, walk)",
 "verb.perception": "a verb of perceiving (see, hear, look)", "verb.possession": "a verb of having or giving (have, get, give)",
 "verb.social": "a social verb (help, work, marry)", "verb.stative": "a verb of being or relating (be, seem, mean)", "verb.weather": "a weather verb"}

funcs = set()
for k, (desc, ws) in FUNC.items():
    FUNC[k] = (desc, [w for w in ws.split() if w not in funcs]); funcs |= set(FUNC[k][1])
cats, tag = {}, {}
for w in words:
    if w in funcs: continue
    syns = wn.synsets(w); seen = set()
    cnt = {p: sum(l.count() for syn in syns if syn.pos() == p for l in syn.lemmas() if l.name() == w) for p in "nvasr"}
    for syn in syns:
        p, ln = syn.pos(), syn.lexname()
        if p in seen or cnt[p] < 0.1 * sum(cnt.values()): continue
        seen.add(p)
        key = {"n": ln, "v": ln, "a": "adjective", "s": "adjective", "r": "adverb"}[p]
        cats.setdefault(key, []).append(w)
        tag.setdefault(w, key)
known = np.array([rank[w] for w in tag]); labels = [tag[w] for w in tag]
unknown = [i for i, w in enumerate(words) if w not in tag and w not in funcs]
for i in range(0, len(unknown), 2000):
    idx = unknown[i:i + 2000]
    nn = np.argsort(-(E[idx] @ E[known].T), axis=1)[:, :5]
    for j, row in zip(idx, nn):
        votes = [labels[r] for r in row]; tag[words[j]] = max(set(votes), key=votes.count)
        cats.setdefault(tag[words[j]], []).append(words[j])
print(len(unknown), "words tagged by nearest neighbours; AI names:", {w: tag[w] for w in AI})

def split(idx, k):
    if k == 1: return [idx]
    X = E[idx] - E[idx].mean(0); pc = np.linalg.eigh(X.T @ X)[1][:, -1]
    order = idx[np.argsort(X @ pc)]; kl = k // 2; cut = round(len(idx) * kl / k)
    return split(order[:cut], kl) + split(order[cut:], k - kl)

def leaf_or_buckets(desc, ws):
    ws = sorted(set(ws), key=lambda w: rank.get(w, N))
    if len(ws) <= 255: return {"desc": desc, "words": ws}
    idx = np.array([rank[w] for w in ws if w in rank]); extra = [w for w in ws if w not in rank]
    kids = {}
    for b in split(idx, -(-len(idx) // LEAF)):
        bw = [words[i] for i in sorted(b)]
        kids[bw[0]] = {"desc": f"a word like {', '.join(bw[:4])}", "words": bw}
    if extra: kids[extra[0]] = {"desc": f"a word like {', '.join(extra[:4])}", "words": extra}
    return {"desc": desc, "children": kids}

root = {}
for pos, label in [("noun", "a noun"), ("verb", "a verb")]:
    kids = {k: leaf_or_buckets(DESC[k], ws) for k, ws in cats.items() if k.startswith(pos + ".")}
    root[pos] = {"desc": label, "children": kids}
root["adjective"] = leaf_or_buckets("an adjective", cats["adjective"])
root["adverb"] = leaf_or_buckets("an adverb", cats["adverb"])
for k, (desc, ws) in FUNC.items():
    root[k] = {"desc": desc, "words": ws}

def ex(node, key):
    if "words" in node: ws = [w for w in node["words"] if tag.get(w, key) == key] or node["words"]
    else: ws = [w for k, c in node["children"].items() for w in ex(c, key or k)]
    node["ex"] = sorted(ws, key=lambda w: rank.get(w, N))[:3]
    return node["ex"]
for k, n in root.items(): ex(n, None if k in ("noun", "verb") else k)
json.dump(root, open("data/pos_tree.json", "w"))
def count(n): return len(n["words"]) if "words" in n else sum(count(c) for c in n["children"].values())
print({k: (count(v), len(v.get("children", {}))) for k, v in root.items()})
print(len({w for k in cats for w in cats[k]}), "distinct content words")
