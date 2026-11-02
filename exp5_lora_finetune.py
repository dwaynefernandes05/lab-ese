"""
EXP 5 - LoRA fine-tuning of a small pretrained LLM on a domain dataset (customer support) - CPU only
Run:  python exp5_lora_finetune.py

NEEDS (cache tonight while you have internet):
    pip install transformers peft torch
    python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('distilgpt2'); AutoModelForCausalLM.from_pretrained('distilgpt2')"
Change MODEL_NAME below if the exam gives another model (the code picks LoRA target modules automatically).
"""
import os, re, time, random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = os.environ.get("MODEL_NAME", "distilgpt2")
RANKS = [4, 8]                  # add 16 if CPU time permits
EPOCHS, LR, BATCH, MAX_LEN = 10, 1e-3, 8, 64      # LoRA tolerates a high LR; small data -> more epochs
random.seed(0); np.random.seed(0); torch.manual_seed(0)

# =====================================================================
# TASK 1 : dataset  (replace with pd.read_csv(...) if a dataset is provided)
# =====================================================================
topics = {
    "reset password": ("Click on Forgot Password on the login page and follow the instructions sent to your email.",
        ["How can I reset my password?", "I forgot my password, what should I do?", "How do I change my login password?", "Unable to log in, need password help"]),
    "track order": ("Open My Orders and click Track Order to see the live status of your delivery.",
        ["How can I track my order?", "Where is my package?", "How do I check my delivery status?", "I want to know the status of my order"]),
    "refund": ("Refunds are processed within 5 to 7 working days to your original payment method.",
        ["How long does a refund take?", "When will I get my money back?", "What is the refund timeline?", "I returned my item, when is the refund?"]),
    "cancel order": ("Go to My Orders, select the order and click Cancel before it is shipped.",
        ["How do I cancel my order?", "I want to cancel my purchase", "Can I cancel an order I placed?", "How to cancel an order before shipping?"]),
    "change address": ("You can update the delivery address from Account Settings before the order is shipped.",
        ["How do I change my delivery address?", "I entered the wrong address", "Can I edit my shipping address?", "How to update my address for an order?"]),
    "payment failed": ("If payment fails, check your card details and balance and try again. Amount deducted is returned in 3 days.",
        ["My payment failed, what now?", "Money was deducted but order not placed", "Why did my payment not go through?", "Payment error during checkout"]),
    "contact support": ("You can reach our support team 24x7 through live chat or by calling the toll free number on the Help page.",
        ["How can I contact customer support?", "How do I talk to an agent?", "What is the support phone number?", "Is there a live chat available?"]),
    "damaged item": ("Sorry about that. Raise a Return request under My Orders and upload a photo of the damaged item for a free replacement.",
        ["I received a damaged product", "My item arrived broken", "The product I got is defective", "What to do if the item is damaged?"]),
    "delete account": ("Go to Account Settings, choose Privacy and click Delete Account. This action cannot be undone.",
        ["How do I delete my account?", "I want to close my account permanently", "How to remove my profile?", "Can I deactivate my account?"]),
    "invoice": ("Your invoice can be downloaded from My Orders by clicking on Download Invoice next to the order.",
        ["Where can I download my invoice?", "How do I get the bill for my order?", "I need a GST invoice", "How to get the receipt of my purchase?"]),
}
records = [{"instruction": "Answer the customer query.", "input": q, "response": ans}
           for ans, qs in topics.values() for q in qs]
df = pd.DataFrame(records)
df = pd.concat([df, df.iloc[:3]])                      # (demo) duplicates
df.loc[len(df)] = ["Answer the customer query.", None, "x"]   # (demo) missing value
print(df.head(5)); print("Missing:", df.isnull().sum().sum(), "| Duplicates:", df.duplicated().sum())
df = df.dropna().drop_duplicates().sample(frac=1, random_state=0).reset_index(drop=True)   # clean + shuffle
n = len(df); tr, va = int(.8 * n), int(.9 * n)
train_df, val_df, test_df = df[:tr], df[tr:va], df[va:]                  # 80 / 10 / 10
print(f"Split -> train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

def fmt(inp):                                  # prompt part (without answer)
    return f"Instruction:\nAnswer the customer query.\n\nInput:\n{inp}\n\nResponse:\n"

# 5 unseen queries for before/after comparison (+ reference answers for scoring)
unseen = [("My refund has not arrived yet, how long will it take?", topics["refund"][0]),
          ("I forgot the password of my account", topics["reset password"][0]),
          ("Where can I see the location of my parcel?", topics["track order"][0]),
          ("The phone I received has a cracked screen", topics["damaged item"][0]),
          ("I would like to download the bill for my order", topics["invoice"][0])]

# =====================================================================
# TASK 2 : tokenisation
# =====================================================================
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
lens = [len(tok(fmt(r.input) + r.response).input_ids) for r in train_df.itertuples()]
print(f"Token lengths: min={min(lens)} mean={np.mean(lens):.0f} max={max(lens)}  -> MAX_LEN={MAX_LEN} (covers almost all, keeps CPU fast)")

def make_batch(rows):
    """tokenise prompt+answer; labels = -100 on the prompt so loss is computed only on the response"""
    ids, labels = [], []
    for r in rows.itertuples():
        p = tok(fmt(r.input)).input_ids
        a = tok(r.response + tok.eos_token).input_ids
        x = (p + a)[:MAX_LEN]                                  # truncation
        y = ([-100] * len(p) + a)[:MAX_LEN]
        pad = MAX_LEN - len(x)                                 # padding
        ids.append(x + [tok.pad_token_id] * pad); labels.append(y + [-100] * pad)
    ids, labels = torch.tensor(ids), torch.tensor(labels)
    return ids, (ids != tok.pad_token_id) | (labels != -100), labels

# =====================================================================
# helpers: scoring, generation, evaluation
# =====================================================================
def score(pred, ref):
    """auto score 1-5 from unigram F1 with the reference (the lab asks a human 1-5; this mimics it)"""
    p, r = set(re.findall(r"\w+", pred.lower())), set(re.findall(r"\w+", ref.lower()))
    if not p or not r: return 1.0
    inter = len(p & r)
    f1 = 0 if inter == 0 else 2 * inter / (len(p) + len(r))
    return round(1 + 4 * f1, 2)

def generate(model, q):
    model.eval()
    enc = tok(fmt(q), return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=30, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]

def evaluate_queries(model, queries):
    res = [(q, generate(model, q), None) for q, _ in queries]
    return [(q, g, score(g, ref)) for (q, g, _), (_, ref) in zip(res, queries)]

@torch.no_grad()
def eval_loss(model, d):
    model.eval(); ids, att, lab = make_batch(d)
    return model(input_ids=ids, attention_mask=att, labels=lab).loss.item()

# =====================================================================
# TASK 3 : baseline
# =====================================================================
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
baseline = evaluate_queries(base, unseen)
print("\nBASELINE (before fine-tuning)")
for q, g, s in baseline: print(f"  Q: {q}\n  A: {g}\n  score={s}")
base_avg = np.mean([s for _, _, s in baseline]); print("Baseline average score:", round(base_avg, 2))
test_q = [(r.input, r.response) for r in test_df.itertuples()]

# =====================================================================
# TASK 4-7 : apply LoRA, count params, fine-tune, evaluate (for each rank)
# =====================================================================
def target_modules(model):
    names = " ".join(n for n, _ in model.named_modules())
    if "c_attn" in names: return ["c_attn"]                    # GPT-2 family
    return ["q_proj", "v_proj"]                                # LLaMA / OPT / Qwen / SmolLM family

def run_lora(r):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=target_modules(model), task_type="CAUSAL_LM",
                     fan_in_fan_out=("c_attn" in target_modules(model)))
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n=== LoRA r={r} | alpha={2*r} | dropout=0.05 | targets={cfg.target_modules} ===")
    print(f"Total params={total:,}  Trainable={trainable:,}  ({100*trainable/total:.2f}%)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    t0, tr_hist, va_hist = time.time(), [], []
    for ep in range(EPOCHS):
        model.train(); d = train_df.sample(frac=1, random_state=ep); losses = []
        for i in range(0, len(d), BATCH):
            ids, att, lab = make_batch(d[i:i + BATCH])
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        tr_hist.append(np.mean(losses)); va_hist.append(eval_loss(model, val_df))
        print(f"  epoch {ep+1}: train loss={tr_hist[-1]:.3f}  val loss={va_hist[-1]:.3f}")
    secs = time.time() - t0
    after = evaluate_queries(model, unseen)
    test_avg = np.mean([s for _, _, s in evaluate_queries(model, test_q)])
    return dict(r=r, trainable=trainable, pct=100 * trainable / total, time=secs, tr=tr_hist, va=va_hist,
                after=after, unseen_avg=np.mean([s for _, _, s in after]), test_avg=test_avg)

results = [run_lora(r) for r in RANKS]

# ---- Task 7 : before vs after
best = results[-1]
print("\nBEFORE vs AFTER (r=%d)" % best["r"])
for (q, g0, s0), (_, g1, s1) in zip(baseline, best["after"]):
    print(f"Q: {q}\n  before ({s0}): {g0}\n  after  ({s1}): {g1}")
print(f"Average score: baseline {base_avg:.2f} -> fine-tuned {best['unseen_avg']:.2f}")

# ---- loss curves
plt.figure(figsize=(10, 3.5))
for res in results:
    plt.subplot(1, 2, 1); plt.plot(res["tr"], label=f"r={res['r']}"); plt.title("Training loss vs epoch"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(res["va"], label=f"r={res['r']}"); plt.title("Validation loss vs epoch"); plt.legend()
plt.tight_layout(); plt.show()

# ---- High complexity task : rank comparison
table = pd.DataFrame([{"rank": r["r"], "trainable params": r["trainable"], "trainable %": round(r["pct"], 2),
                       "train time (s)": round(r["time"], 1), "train loss": round(r["tr"][-1], 3),
                       "val loss": round(r["va"][-1], 3), "avg score (5 unseen)": round(r["unseen_avg"], 2),
                       "avg score (test split)": round(r["test_avg"], 2)} for r in results])
print("\nLoRA RANK COMPARISON\n", table.to_string(index=False))
print("Overfitting check: train loss << val loss and val loss rising => overfitting.")
