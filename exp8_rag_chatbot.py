"""
EXP 8 - RAG chatbot: chunking -> embeddings -> vector index -> top-k retrieval -> prompt -> LLM answer
Run:  python exp8_rag_chatbot.py            (CPU)

Knowledge base: the MOCK documents in DOCS below (made-up facts about a fictional institute!).
  In the exam: put the provided .txt files in ./knowledge_base/ (they replace DOCS automatically)
  and edit QA_SET to match their content.

Models (optional, cache tonight - code falls back to offline options if missing):
  embeddings : sentence-transformers 'all-MiniLM-L6-v2'   (fallback = TF-IDF from scikit-learn)
  generator  : 'google/flan-t5-base'                       (fallback = returns the best retrieved sentence)
"""
import os, re, time, glob
import numpy as np
import pandas as pd

# =====================================================================
# TASK 1 : knowledge base  (source name + text)
# =====================================================================
DOCS = {
    "about.txt": "The Demo Institute of Technology was founded in 1962 and is located in Mumbai. It is an autonomous institute "
                 "affiliated to the University of Mumbai. The campus spans 5 acres and has a central library open from 8 AM to 8 PM.",
    "admissions.txt": "Admission to the B.Tech programme is based on the state entrance exam score and the centralised admission process. "
                      "The total intake for B.Tech is 600 seats. The application window opens in June every year. "
                      "The M.Tech programme admission is based on the GATE score.",
    "departments.txt": "The institute has departments of Computer Engineering, Information Technology, Electronics and Telecommunication, "
                       "and Mechanical Engineering. The Computer Engineering department has 120 seats and runs a Generative AI elective in semester 7. "
                       "The Mechanical Engineering department has 60 seats.",
    "placements.txt": "The placement cell conducts campus drives from August to March. Last year the highest package was 45 LPA and "
                      "the average package was 9 LPA. More than 150 companies visited the campus. Internships are mandatory in the sixth semester.",
    "exam_rules.txt": "A student must have a minimum of 75 percent attendance to appear in the end semester examination. "
                      "The passing criterion is 40 percent in each course. Re-examination forms must be filled within 7 days of result declaration. "
                      "Use of mobile phones during examinations is strictly prohibited.",
    "hostel.txt": "The institute provides separate hostels for boys and girls. The hostel fee is 60000 rupees per year including mess charges. "
                  "Hostel allotment is based on the distance from home and merit. Visitors are allowed between 5 PM and 7 PM only.",
    "library.txt": "The library has more than 50000 books and subscribes to 200 journals. Students can borrow 4 books at a time for 14 days. "
                   "A fine of 2 rupees per day is charged for late returns.",
}
if os.path.isdir("knowledge_base"):                                  # real documents override the mock ones
    DOCS = {os.path.basename(f): open(f, encoding="utf-8").read() for f in glob.glob("knowledge_base/*.txt")}

# QA set: (type, question, expected keyword, gold source)  - keyword None => unanswerable
QA_SET = [
    ("direct", "When was the institute founded?", "1962", "about.txt"),
    ("direct", "How many B.Tech seats are there?", "600", "admissions.txt"),
    ("direct", "What is the highest package offered?", "45", "placements.txt"),
    ("direct", "What is the minimum attendance required for the exam?", "75", "exam_rules.txt"),
    ("direct", "How much is the hostel fee?", "60000", "hostel.txt"),
    ("direct", "How many books can a student borrow from the library?", "4", "library.txt"),
    ("direct", "Which exam score is used for M.Tech admission?", "GATE", "admissions.txt"),
    ("comparison", "Which department has more seats, Computer or Mechanical Engineering?", "120", "departments.txt"),
    ("multi-doc", "What is the fine for late book return and for how many days can I keep a book?", "14", "library.txt"),
    ("reasoning", "If I have 70 percent attendance can I write the end semester exam?", "75", "exam_rules.txt"),
    ("unanswerable", "Who is the current director of the institute?", None, None),
    ("unanswerable", "What is the fee for the MBA programme?", None, None),
    ("unanswerable", "Does the institute have a swimming pool?", None, None),
]

# =====================================================================
# TASK 2 + 3 : clean and chunk  (record format: chunk_id | source | text)
# =====================================================================
def chunk_docs(docs, size=40, overlap=10):
    """split each document into word-chunks of `size` words with `overlap` words shared with the previous chunk"""
    chunks = []
    for src, text in docs.items():
        words = re.sub(r"\s+", " ", text).split()
        for start in range(0, max(1, len(words) - overlap), size - overlap):
            chunks.append({"chunk_id": len(chunks), "source": src, "text": " ".join(words[start:start + size])})
    return chunks

# =====================================================================
# TASK 4 : embeddings  (sentence-transformers if cached, else TF-IDF)
# =====================================================================
try:
    from sentence_transformers import SentenceTransformer
    _st = SentenceTransformer(os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2"))
    EMBED_NAME = "sentence-transformers"
except Exception:
    _st = None; EMBED_NAME = "tfidf"
from sklearn.feature_extraction.text import TfidfVectorizer

class Embedder:
    def __init__(self, texts):
        if _st is None:
            self.tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(texts)
    def __call__(self, texts):
        v = _st.encode(list(texts), normalize_embeddings=True) if _st is not None \
            else self.tfidf.transform(texts).toarray()
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)         # L2 normalise -> dot product = cosine

# =====================================================================
# TASK 5 + 6 : vector index and top-k retrieval (explicit - no hidden RAG chain)
# =====================================================================
class VectorStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.embed = Embedder([c["text"] for c in chunks])
        self.matrix = self.embed([c["text"] for c in chunks])               # (n_chunks, dim)  = the "index"
    def search(self, query, k=3, threshold=None):
        q = self.embed([query])[0]
        scores = self.matrix @ q                                             # cosine similarity with every chunk
        top = np.argsort(-scores)[:k]                                        # indices of the k best
        return [(self.chunks[i], float(scores[i])) for i in top if threshold is None or scores[i] >= threshold]
# (with FAISS:  index = faiss.IndexFlatIP(dim); index.add(matrix); scores, ids = index.search(q[None], k))

# =====================================================================
# TASK 7 + 8 : prompt + generator
# =====================================================================
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    GEN = os.environ.get("GEN_MODEL", "google/flan-t5-base")
    tok = AutoTokenizer.from_pretrained(GEN); llm = AutoModelForSeq2SeqLM.from_pretrained(GEN); USE_LLM = True
except Exception:
    USE_LLM = False

REFUSAL = "The information is not available in the provided documents."

def build_prompt(question, retrieved):
    context = "\n".join(f"[{c['source']}] {c['text']}" for c, _ in retrieved)
    return (f"Answer the question using ONLY the context below. Mention the source file name. "
            f"If the answer is not in the context, say: {REFUSAL}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:")

def generate(question, retrieved):
    if not retrieved:
        return REFUSAL
    if USE_LLM:
        enc = tok(build_prompt(question, retrieved), return_tensors="pt", truncation=True, max_length=512)
        out = llm.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=60)
        return tok.decode(out[0], skip_special_tokens=True).strip()
    # fallback (no LLM): choose the sentence from the retrieved chunks with the most word overlap with the question
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
    words = lambda t: {w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in SW}
    q = words(question)
    sents = [(s, c["source"]) for c, _ in retrieved for s in re.split(r"(?<=[.!?])\s+", c["text"])]
    s, src = max(sents, key=lambda t: len(q & words(t[0])))
    if len(q & words(s)) < 2:                                   # not enough evidence -> refuse
        return REFUSAL
    return f"{s} (source: {src})"

def rag_answer(store, question, k=3, threshold=None):
    t0 = time.time()
    retrieved = store.search(question, k, threshold)
    ans = generate(question, retrieved)
    return ans, retrieved, time.time() - t0

# =====================================================================
# EVALUATION helpers
# =====================================================================
def is_refusal(ans):
    return any(w in ans.lower() for w in ["not available", "cannot", "no information", "unavailable"])

def evaluate(store, k=3, qa=QA_SET, threshold=None, verbose=False):
    rows = []
    for typ, q, key, gold in qa:
        ans, ret, dt = rag_answer(store, q, k, threshold)
        srcs = [c["source"] for c, _ in ret]
        hit = (gold in srcs) if gold else None
        rank = (srcs.index(gold) + 1) if gold in srcs else 0
        if key is None:                                               # unanswerable -> must refuse
            correct = is_refusal(ans); halluc = not correct; faithful = True
        else:
            correct = key.lower() in ans.lower() and not is_refusal(ans)
            ctx = " ".join(c["text"] for c, _ in ret).lower()
            faithful = key.lower() in ctx                              # is the answer supported by the retrieved text?
            halluc = (not faithful) and not is_refusal(ans)
        rows.append(dict(type=typ, q=q, hit=hit, rr=(1 / rank if rank else 0), correct=correct,
                         faithful=faithful, halluc=halluc, time=dt, answer=ans))
        if verbose: print(f"[{typ}] {q}\n   -> {ans}\n   sources={srcs}")
    d = pd.DataFrame(rows); ans_d = d[d.hit.notna()]
    return d, dict(hit_rate=ans_d.hit.astype(float).mean(), mrr=ans_d.rr.mean(), accuracy=d.correct.mean(),
                   faithfulness=d.faithful.mean(), halluc_rate=d.halluc.mean(), avg_time=d.time.mean())

# =====================================================================
# MAIN : run all experiments
# =====================================================================
if __name__ == "__main__":
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 80)
    print(f"Embedder: {EMBED_NAME} | Generator: {'LLM' if USE_LLM else 'extractive fallback'}")
    chunks = chunk_docs(DOCS)
    print(f"{len(DOCS)} documents -> {len(chunks)} chunks. Example record: {chunks[0]}")
    store = VectorStore(chunks)
    print("Index matrix shape:", store.matrix.shape)

    # ---- show retrieval for a few queries (screenshots for the report)
    print("\n=== RETRIEVAL DEMO (top-3 with scores) ===")
    for q in ["What is the highest package offered?", "How much is the hostel fee?", "Who is the director?"]:
        ans, ret, dt = rag_answer(store, q, k=3)
        print(f"\nQ: {q}")
        for c, s in ret: print(f"   {s:.3f}  [{c['source']}] {c['text'][:90]}...")
        print("   ANSWER:", ans)

    # ---- Experiment A : effect of k
    print("\n=== EXPERIMENT A : top-k ===")
    rowsA = []
    for k in (1, 3, 5, 10):
        _, m = evaluate(store, k); rowsA.append({"k": k, **{a: round(b, 3) for a, b in m.items()}})
    print(pd.DataFrame(rowsA).to_string(index=False))

    # ---- Experiment B : retrieval failure (remove the source that contains the answer)
    print("\n=== EXPERIMENT B : remove placements.txt ===")
    store_B = VectorStore(chunk_docs({k: v for k, v in DOCS.items() if k != "placements.txt"}))
    q = "What is the highest package offered?"
    ans, ret, _ = rag_answer(store_B, q, 3)
    print("Q:", q, "\nRetrieved sources:", [c["source"] for c, _ in ret], "\nAnswer:", ans,
          "\n=> refused?" , is_refusal(ans), "(if False the system hallucinated / answered from wrong context)")

    # ---- Experiment C : irrelevant / conflicting document
    print("\n=== EXPERIMENT C : add a conflicting document ===")
    DOCS_C = dict(DOCS, **{"old_notice.txt": "Old notice: the highest package last year was 12 LPA and the hostel fee is 30000 rupees per year."})
    store_C = VectorStore(chunk_docs(DOCS_C))
    for q in ["What is the highest package offered?", "How much is the hostel fee?"]:
        ans, ret, _ = rag_answer(store_C, q, 3)
        print("Q:", q, "\n   sources:", [c["source"] for c, _ in ret], "\n   answer:", ans)

    # ---- Experiment D : RAG vs ICL (ICL = paste the corpus in the prompt, cut at the model limit)
    print("\n=== EXPERIMENT D : RAG vs ICL ===")
    all_text = " ".join(DOCS.values())
    class ICLStore:                                   # same interface, but 'retrieval' = first N words of everything
        def search(self, q, k=3, threshold=None):
            return [({"source": "all_docs", "text": " ".join(all_text.split()[:120])}, 1.0)]   # limited context window
    _, mR = evaluate(store, 3); _, mI = evaluate(ICLStore(), 1)
    print(pd.DataFrame([{"system": "ICL (first 120 words)", **{a: round(b, 3) for a, b in mI.items() if a in ("accuracy", "faithfulness", "halluc_rate")}},
                        {"system": "RAG (k=3)", **{a: round(b, 3) for a, b in mR.items() if a in ("accuracy", "faithfulness", "halluc_rate")}}]).to_string(index=False))

    # ---- Advanced : relevance threshold + two chunk sizes
    print("\n=== ADVANCED : relevance threshold (reject low-confidence retrievals) ===")
    for th in (None, 0.1, 0.2, 0.3):
        _, m = evaluate(store, 3, threshold=th); print(f"threshold={th}: accuracy={m['accuracy']:.2f} halluc={m['halluc_rate']:.2f}")
    print("\n=== ADVANCED : chunk size comparison ===")
    for size in (20, 40, 80):
        st = VectorStore(chunk_docs(DOCS, size=size, overlap=size // 4)); _, m = evaluate(st, 3)
        print(f"chunk_size={size:3d} words -> {len(st.chunks):2d} chunks | hit@3={m['hit_rate']:.2f} accuracy={m['accuracy']:.2f}")

    # ---- full answer log / observation sheet
    print("\n=== OBSERVATION SHEET (k=3) ===")
    d, _ = evaluate(store, 3)
    print(d[["type", "q", "hit", "correct", "faithful", "halluc"]].to_string())
