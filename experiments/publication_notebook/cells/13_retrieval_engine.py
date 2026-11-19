from sentence_transformers import SentenceTransformer

EMB_DIR = RETR_DIR / "emb"
EMB_DIR.mkdir(parents=True, exist_ok=True)
ENCODERS = {                                    # same four encoders as the earlier retrieval experiments
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "PubMedBERT": "pritamdeka/S-PubMedBert-MS-MARCO",
    "BGE": "BAAI/bge-base-en-v1.5",
}
VBRI_ENCODER = "BGE"
if TEST_MODE:
    ENCODERS = {"MiniLM": ENCODERS["MiniLM"]}
    VBRI_ENCODER = "MiniLM"
K_GRID = [1, 3, 5, 7, 9, 11, 15]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# answers are used as raw text (stripped), exactly as in the original VBRI experiments
SPLITS = {"bal": RAW["train"], "val": RAW["validation"], "test": RAW["test"]}
if AUG_OK:
    SPLITS["unb"] = RAW_AUG["train"]
META = {s: {"q": np.array([int(v) for v in ds["question_id"]]), "y": np.array([int(v) for v in ds["label"]]),
            "ans": [str(a).strip() for a in ds["answer"]], "question": [str(a).strip() for a in ds["question"]],
            "id": [int(v) for v in ds["id"]]} for s, ds in SPLITS.items()}

def texts_for(split, fmt):
    m = META[split]
    if fmt == "ans":
        return m["ans"]
    return [f"Question: {q}\nAnswer: {a}" for q, a in zip(m["question"], m["ans"])]

_ENC = {}
def get_encoder(name):
    if name not in _ENC:
        _ENC[name] = SentenceTransformer(ENCODERS[name], device=DEVICE)
    return _ENC[name]

def encoder_params_m(name):
    return round(sum(p.numel() for p in get_encoder(name).parameters()) / 1e6, 1)

def embed(enc, fmt, split):
    """Normalised embeddings, cached on disk so later stages can be re-run without re-encoding."""
    p = EMB_DIR / f"{enc}_{fmt}_{split}.npy"
    if p.exists():
        E = np.load(p)
        if E.shape[0] == len(META[split]["y"]):
            return E
    E = get_encoder(enc).encode(texts_for(split, fmt), batch_size=ENCODE_BS, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    np.save(p, E)
    return E

def _topk_rows(S, k):
    k = min(k, S.shape[1])
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    ps = np.take_along_axis(S, part, 1)
    order = np.argsort(-ps, axis=1, kind="stable")
    return np.take_along_axis(part, order, 1), np.take_along_axis(ps, order, 1)

def topk_neighbours(S_full=None, Q=None, T=None, k=5, q_qid=None, t_qid=None, isolated=False):
    """Indices (into the training set) and similarities of the k nearest neighbours, best first.
    isolated=True searches only the training answers of the query's own question (the 9-VDB design)."""
    if not isolated:
        S = S_full if S_full is not None else Q @ T.T
        return _topk_rows(S, k)
    n = Q.shape[0]
    idx = np.zeros((n, k), dtype=int)
    sims = np.zeros((n, k), dtype=np.float32)
    for q in np.unique(q_qid):
        qi, ti = np.where(q_qid == q)[0], np.where(t_qid == q)[0]
        ii, ss = _topk_rows(Q[qi] @ T[ti].T, k)
        idx[qi], sims[qi] = ti[ii], ss
    return idx, sims

def inv_freq_weights(t_y, t_qid=None):
    """Per-training-vector class weight n / (3 * n_class); per question if t_qid is given."""
    w = np.ones(len(t_y))
    groups = [np.arange(len(t_y))] if t_qid is None else [np.where(t_qid == q)[0] for q in np.unique(t_qid)]
    for g in groups:
        cnt = Counter(t_y[g].tolist())
        for c in (0, 1, 2):
            m = g[t_y[g] == c]
            if cnt.get(c):
                w[m] = len(g) / (3 * cnt[c])
    return w

def vote(idx, weights, t_y, vec_w=None):
    """Weighted class scores (n, 3): sum of neighbour weights per class (optionally times a class weight)."""
    lab = t_y[idx]
    ww = weights * (vec_w[idx] if vec_w is not None else 1.0)
    return np.stack([(ww * (lab == c)).sum(1) for c in (0, 1, 2)], axis=1)

def contamination(idx, q_qid, t_qid):
    return (t_qid[idx] != q_qid[:, None]).mean(1) * 100.0

def dense_run(enc, fmt, train, split, k, isolated, class_weighted=False):
    T, Q = embed(enc, fmt, train), embed(enc, fmt, split)
    idx, sims = topk_neighbours(Q=Q, T=T, k=k, q_qid=META[split]["q"], t_qid=META[train]["q"], isolated=isolated)
    vw = inv_freq_weights(META[train]["y"], META[train]["q"] if isolated else None) if class_weighted else None
    sc = vote(idx, np.maximum(sims, 0.0), META[train]["y"], vw)
    return {"pred": sc.argmax(1), "scores": sc, "idx": idx, "sims": sims,
            "contam": contamination(idx, META[split]["q"], META[train]["q"])}

# ------------------------------ lexical baselines ----------------------------
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

def _lex_tok(t):
    return re.sub(r"[^a-z0-9\s]", " ", t.lower()).split()

_LEX = {}
def lexical_scores(method, train, split):
    key = (method, train, split)
    if key not in _LEX:
        tr, qs = META[train]["ans"], META[split]["ans"]
        if method == "BM25":
            bm = BM25Okapi([_lex_tok(a) for a in tr])
            S = np.stack([bm.get_scores(_lex_tok(q)) for q in qs]).astype(np.float64)
        else:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
            Xt = vec.fit_transform(tr)
            S = (vec.transform(qs) @ Xt.T).toarray()
        _LEX[key] = S
    return _LEX[key]

def lexical_run(method, train, split, k):
    S = lexical_scores(method, train, split)
    idx, s = topk_neighbours(S_full=S, k=k)
    if method == "BM25":                       # as before: softmax of BM25 scores inside the top k
        w = np.exp(s - s.max(1, keepdims=True))
        w = w / w.sum(1, keepdims=True)
    else:
        w = np.maximum(s, 0.0)
    sc = vote(idx, w, META[train]["y"])
    return {"pred": sc.argmax(1), "scores": sc, "idx": idx, "sims": s,
            "contam": contamination(idx, META[split]["q"], META[train]["q"])}

# ------------------------------ calibration and safety rule ------------------
def temp_probs(idx, sims, t_y, tau):
    """Temperature-scaled softmax over the top-k similarities, summed per class (as in the VBRI notebook)."""
    e = np.exp((sims - sims.max(1, keepdims=True)) / tau)
    sc = vote(idx, e, t_y)
    return sc / sc.sum(1, keepdims=True)

def calibrated_pred(P, delta, theta):
    """Inconclusive if top-2 gap < delta; else BPD if P(BPD) >= theta; else Normal or Inconclusive."""
    out = np.zeros(len(P), dtype=int)
    srt = -np.sort(-P, axis=1)
    for i, p in enumerate(P):
        if srt[i, 0] - srt[i, 1] < delta:
            out[i] = 2
        elif p[1] >= theta:
            out[i] = 1
        elif p[0] >= p[2]:
            out[i] = 0
        else:
            out[i] = 2
    return out

CSSRS_ATTEMPT_PATTERNS = [                     # C-SSRS 'actual attempt' categories (unchanged from the VBRI notebook)
    r'\b(try|tried|attempt|attempted)\s+suicide\b',
    r'\b(try|tried|attempt|attempted)\s+(to\s+)?(hang|strangle)\b',
    r'\b(try|tried|attempt|attempted)\s+(to\s+)?(overdose|poison)\b',
    r'\b(try|tried|attempt|attempted)\s+(to\s+)?(cut|slit)\s+(my\s+)?(wrist|throat|vein)\b',
    r'\b(try|tried|attempt|attempted)\s+(to\s+)?(jump)\s+(off|from)\b',
    r'\b(swallowed|swallowing|drank)\s+(poison|pills|bleach|chemical)\b',
    r'\b(hang|hanged|hung)\s+myself\b',
    r'\bslit\s+(my\s+)?(wrist|wrists|throat)\b',
]
CSSRS_RE = re.compile("|".join(CSSRS_ATTEMPT_PATTERNS), re.IGNORECASE)

def apply_cssrs(pred, split):
    """Q5 only: if an actual-attempt pattern matches and the prediction is Normal, change it to BPD."""
    out, fired = np.array(pred).copy(), []
    for i, (q, a) in enumerate(zip(META[split]["q"], META[split]["ans"])):
        if q == 5 and out[i] == 0 and CSSRS_RE.search(a):
            out[i] = 1
            fired.append(i)
    return out, fired

def choose_k(run_fn, k_grid=K_GRID):
    """Pick k on the VALIDATION set by macro-F1 (ties -> smaller k)."""
    f1 = {k: summarize(META["val"]["y"].tolist(), run_fn("val", k)["pred"].tolist())["macro_f1"] for k in k_grid}
    best = max(f1, key=lambda k: (round(f1[k], 6), -k))
    return best, f1
