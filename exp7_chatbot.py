
import os, re, time

DOMAIN_NAME = "E-Commerce Customer Support"
DOMAIN_INFO = """
E-COMMERCE HELP DESK INFORMATION
1. Orders can be cancelled from My Orders before they are shipped.
2. Refunds are processed within 5 to 7 working days to the original payment method.
3. Delivery normally takes 3 to 5 working days.
4. Items can be returned within 30 days of delivery if they are unused.
5. To reset the password, click Forgot Password on the login page.
6. Customer support is available 24x7 through live chat and the toll free number on the Help page.
7. The delivery address can be changed from Account Settings before the order is shipped.
8. Damaged items are replaced for free; upload a photo under Return request in My Orders.
9. Invoices can be downloaded from My Orders using Download Invoice.
10. Premium members get free shipping on all orders above 499 rupees.
"""

# ---------------------------------------------------------------- load the LLM (once)
GEN_MODEL = os.environ.get("GEN_MODEL", "google/flan-t5-base")
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    llm = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    USE_LLM = True
except Exception as e:
    USE_LLM = False
    print("[!] Could not load", GEN_MODEL, "-> using keyword fallback.", type(e).__name__)

# ---------------------------------------------------------------- prompt builder
def create_prompt(question):
    """System instructions + domain information + user question"""
    return f"""You are a helpful {DOMAIN_NAME} assistant.
Answer the question using ONLY the information below.
If the answer is not in the information, reply exactly: "Sorry, that information is not available."

INFORMATION:
{DOMAIN_INFO}

QUESTION: {question}
ANSWER:"""

# ---------------------------------------------------------------- fallback (no model)
STOP = set("the a an is are of to for in on from how what when can i my do does be it and or by with within long take many get new need want should could have has was were this that".split())
def keyword_answer(question):
    """no-LLM fallback: return the fact sharing the most (5-letter-prefix) words with the question"""
    stem = lambda text: {w[:5] for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}
    q = stem(question)
    best, best_score = None, 0
    for line in DOMAIN_INFO.strip().split("\n")[1:]:
        score = len(q & stem(line))
        if score > best_score: best, best_score = line, score
    return re.sub(r"^\d+\.\s*", "", best) if best_score >= 1 else "Sorry, that information is not available."

# ---------------------------------------------------------------- generate the answer
def generate_answer(question):
    if not USE_LLM:
        return keyword_answer(question)
    enc = tokenizer(create_prompt(question), return_tensors="pt", truncation=True, max_length=512)
    out = llm.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=60, num_beams=2)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

# ---------------------------------------------------------------- evaluation (Step 16): 10 questions, 4 categories
TESTS = [  # (category, question, keyword that a correct answer must contain; None = should refuse)
    ("direct", "How long does a refund take?", "5"),
    ("direct", "Can I cancel an order?", "cancel"),
    ("direct", "How do I download my invoice?", "invoice"),
    ("direct", "When is customer support available?", "24"),
    ("paraphrase", "I forgot my login credentials, what should I do?", "forgot"),
    ("paraphrase", "My parcel arrived broken, can I get a new one?", "replace"),
    ("multi-part", "How many days does delivery take and how long is the return window?", "30"),
    ("multi-part", "How can I change my address and until when can I do it?", "shipped"),
    ("out-of-domain", "What is the capital of France?", None),
    ("out-of-domain", "Who is the CEO of Microsoft?", None),
]

if __name__ == "__main__":
    print("LLM in use:" , GEN_MODEL if USE_LLM else "keyword fallback")
    correct = 0
    for cat, q, key in TESTS:
        t0 = time.time(); ans = generate_answer(q); dt = time.time() - t0
        if key is None:
            ok = "not available" in ans.lower() or "sorry" in ans.lower()
        else:
            ok = key.lower() in ans.lower()
        correct += ok
        print(f"[{cat:13}] {'OK ' if ok else 'BAD'} ({dt:.1f}s)  Q: {q}\n{' '*22}A: {ans}")
    print(f"\nAccuracy: {correct}/{len(TESTS)}")
    print("Limitations to mention: hallucination on out-of-domain questions, only knows the 10 facts in the prompt,"
          "\nprompt length is limited (no retrieval) -> that is why Exp 8 uses RAG.")
