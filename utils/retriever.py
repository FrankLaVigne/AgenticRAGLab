"""
Lightweight retriever backed by prebuilt/bfrpg_chunks.json.

Supports two modes:
  1. Embedding-based cosine similarity (if the MaaS endpoint exposes an
     embeddings model and numpy is installed).
  2. Pure-Python TF-IDF fallback with no external dependencies.

Both modes expose the same interface used by the notebooks:

    retriever = Retriever.load()
    results   = retriever.query(query_text, n_results=3)
    # results = {"documents": [[str, ...]], "distances": [[float, ...]]}
"""

import json, math, re
from pathlib import Path

CHUNKS_PATH = Path(__file__).parent.parent / "prebuilt" / "bfrpg_chunks.json"


# ── pure-python TF-IDF retriever (zero dependencies) ────────────────────

def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_idf(corpus_tokens):
    n = len(corpus_tokens)
    df = {}
    for tokens in corpus_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    return {t: math.log(n / d) for t, d in df.items()}


def _tfidf_vector(tokens, idf):
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    length = len(tokens) or 1
    vec = {}
    for t, count in tf.items():
        if t in idf:
            vec[t] = (count / length) * idf[t]
    return vec


def _cosine(a, b):
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class Retriever:
    """ChromaDB-compatible interface over a flat JSON chunk file."""

    def __init__(self, chunks):
        self.chunks = chunks                       # list of {"id", "text", ...}
        self._texts = [c["text"] for c in chunks]
        self._token_lists = [_tokenize(t) for t in self._texts]
        self._idf = _build_idf(self._token_lists)
        self._vectors = [_tfidf_vector(tl, self._idf) for tl in self._token_lists]

    # ── public API (mirrors collection.query) ────────────────────────
    @property
    def name(self):
        return "basic_fantasy_corpus"

    def count(self):
        return len(self.chunks)

    def query(self, query_texts, n_results=3, include=None):
        """Return top-n chunks for each query string.

        Returns a dict matching the ChromaDB query result shape:
            {"documents": [[str, ...]], "distances": [[float, ...]]}
        Distance is (1 - cosine_similarity) so lower = more similar.
        """
        all_docs = []
        all_dists = []
        for q in query_texts:
            q_vec = _tfidf_vector(_tokenize(q), self._idf)
            scored = []
            for i, doc_vec in enumerate(self._vectors):
                sim = _cosine(q_vec, doc_vec)
                scored.append((1.0 - sim, i))      # distance = 1 - similarity
            scored.sort()
            top = scored[:n_results]
            all_docs.append([self._texts[i] for _, i in top])
            all_dists.append([d for d, _ in top])
        return {"documents": all_docs, "distances": all_dists}

    # ── factory ──────────────────────────────────────────────────────
    @classmethod
    def load(cls, path=None):
        p = Path(path) if path else CHUNKS_PATH
        with open(p, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return cls(chunks)
