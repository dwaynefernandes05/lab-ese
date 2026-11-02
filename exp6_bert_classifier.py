
import os, re, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MODEL_NAME = os.environ.get("MODEL_NAME", "bert-base-uncased")
random.seed(0); np.random.seed(0); torch.manual_seed(0)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)

# =====================================================================
# TASK 1 : dataset, exploration, 2 preprocessing versions, split
# =====================================================================
def build_dataset(n=400):
    subj = ["movie", "service", "food", "product", "staff", "delivery", "phone", "hotel"]
    pos = ["excellent", "wonderful", "amazing", "great", "fantastic", "superb", "lovely", "brilliant"]
    neg = ["terrible", "awful", "horrible", "poor", "disappointing", "worst", "bad", "useless"]
    tpl = ["The {s} was {a}.", "I thought the {s} was really {a}!", "Honestly, such a {a} {s}.",
           "What a {a} {s}, I would tell everyone.", "The {s} is {a}, nothing else to say."]
    rows = []
    for _ in range(n):
        label = random.randint(0, 1)
        rows.append((random.choice(tpl).format(s=random.choice(subj), a=random.choice(pos if label else neg)), label))
    return pd.DataFrame(rows, columns=["text", "label"])

df = pd.read_csv("data.csv") if os.path.exists("data.csv") else build_dataset()
print(df.sample(5, random_state=1)); print("Missing:", df.isnull().sum().sum(), "| Duplicates:", df.duplicated().sum())
df = df.dropna().drop_duplicates().reset_index(drop=True)
print("Class distribution:\n", df.label.value_counts())

def normalize(t):                                        # Version B : cleaned text
    t = re.sub(r"[^a-z\s]", " ", t.lower())
    return " ".join(w for w in t.split() if w not in ENGLISH_STOP_WORDS)
dfB = df.copy(); dfB["text"] = dfB.text.apply(normalize)  # Version A = df (minimal preprocessing)

N_TRAIN = 160                                            # keep small for CPU
def split(d):
    tr, rest = train_test_split(d, test_size=0.4, random_state=0, stratify=d.label)
    va, te = train_test_split(rest, test_size=0.5, random_state=0, stratify=rest.label)
    return tr[:N_TRAIN], va[:40], te[:60]
trainA, valA, testA = split(df)
trainB, valB, testB = split(dfB)
print(f"Split -> train {len(trainA)} | val {len(valA)} | test {len(testA)}")

# =====================================================================
# TASK 2 : tokenisation analysis
# =====================================================================
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
s = "I really enjoyed this movie."
enc = tok(s, padding="max_length", truncation=True, max_length=16)
print("\nTokens   :", tok.convert_ids_to_tokens(enc["input_ids"]))
print("Input IDs:", enc["input_ids"])
print("Attn mask:", enc["attention_mask"])
for L in (32, 128):
    lens = [len(tok(t, truncation=True, max_length=L)["input_ids"]) for t in df.text[:100]]
    print(f"max_length={L}: avg real tokens={np.mean(lens):.1f}  padded tensor size per batch of 16 = 16x{L}")

# =====================================================================
# MODEL : BERT + selectable classification head
#   linear  : BERT -> Linear
#   dropout : BERT -> Dropout -> Linear
#   dense   : BERT -> Dense -> Dropout -> Linear
# =====================================================================
class BertClassifier(nn.Module):
    def __init__(self, head="linear", n_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        h = self.bert.config.hidden_size
        if head == "linear":    self.head = nn.Linear(h, n_classes)
        elif head == "dropout": self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, n_classes))
        else:                   self.head = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))
    def forward(self, ids, mask):
        cls = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]   # [CLS] vector
        return self.head(cls)                                                         # logits (softmax is inside the loss)

def encode(texts, L):
    e = tok(list(texts), padding="max_length", truncation=True, max_length=L, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]

@torch.no_grad()
def predict(model, texts, L):
    model.eval(); preds = []
    for i in range(0, len(texts), 32):
        ids, m = encode(texts[i:i + 32], L); preds += model(ids, m).argmax(1).tolist()
    return np.array(preds)

def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(acc=accuracy_score(y, p), prec=pr, rec=rc, f1=f1)

# =====================================================================
# TASK 3/4/5/6 : one function that trains any configuration
# =====================================================================
def run(name, train, val, test, L=32, lr=2e-5, bs=16, epochs=3, wd=0.01, head="linear", class_weight=False):
    torch.manual_seed(0)
    model = BertClassifier(head)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = None
    if class_weight:                                          # inverse-frequency weights
        cnt = np.bincount(train.label, minlength=2); w = torch.tensor(len(train) / (2 * cnt), dtype=torch.float)
    lossf = nn.CrossEntropyLoss(weight=w)
    t0, tr_l, va_l = time.time(), [], []
    for ep in range(epochs):
        model.train(); d = train.sample(frac=1, random_state=ep); losses = []
        for i in range(0, len(d), bs):
            b = d[i:i + bs]; ids, m = encode(b.text, L)
            loss = lossf(model(ids, m), torch.tensor(b.label.values))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        tr_l.append(np.mean(losses))
        model.eval()
        with torch.no_grad():
            ids, m = encode(val.text, L); va_l.append(lossf(model(ids, m), torch.tensor(val.label.values)).item())
    secs = time.time() - t0
    p = predict(model, list(test.text), L)
    out = dict(name=name, **metrics(test.label.values, p), time=secs, val_loss=va_l[-1], tr_hist=tr_l, va_hist=va_l,
               cm=confusion_matrix(test.label.values, p), model=model, L=L)
    print(f"[{name}] acc={out['acc']:.3f} f1={out['f1']:.3f} time={secs:.0f}s train_loss={tr_l[-1]:.3f} val_loss={va_l[-1]:.3f}")
    return out

# ---- Task 3 : baseline BEFORE fine-tuning (random head => ~chance)
m0 = BertClassifier("linear")
print("\nBaseline (pretrained BERT + untrained head):", metrics(testA.label.values, predict(m0, list(testA.text), 32)))

# ---- Task 1 experiment : minimal vs cleaned text
print("\n--- Preprocessing: Version A (raw) vs Version B (cleaned) ---")
rA = run("A: raw text", trainA, valA, testA)
rB = run("B: cleaned text", trainB, valB, testB)

# ---- experiments table (E1..E5)
print("\n--- Experiments ---")
E1 = rA                                                                           # baseline
E2 = run("E2: max_len=128", trainA, valA, testA, L=128)                           # sequence length
E3 = run("E3: lr=5e-5, epochs=4", trainA, valA, testA, lr=5e-5, epochs=4)         # hyper-parameters
E4 = run("E4: dense+dropout head", trainA, valA, testA, head="dense")             # classification head
# Task 6 : imbalance - make an imbalanced training set (keep only 15% of negatives)
imb = pd.concat([trainA[trainA.label == 1], trainA[trainA.label == 0].sample(frac=0.15, random_state=0)])
print("Imbalanced train distribution:", imb.label.value_counts().to_dict())
E5a = run("E5a: imbalanced, CE loss", imb, valA, testA)
E5b = run("E5b: imbalanced, weighted loss", imb, valA, testA, class_weight=True)
E1.update(name="E1: baseline")

# =====================================================================
# TASK 7 : error & robustness analysis with the best model
# =====================================================================
best = max([E1, E2, E3, E4], key=lambda r: r["f1"])["model"]; bestL = max([E1, E2, E3, E4], key=lambda r: r["f1"])["L"]
challenge = [  # (text, true label, category)
    ("The servce was awsome", 1, "spelling"), ("The delivry was terible", 0, "spelling"),
    ("The product is gr8", 1, "abbreviation"), ("The hotel was gud tbh", 1, "abbreviation"),
    ("The movie was not good", 0, "negation"), ("The staff was not bad at all", 1, "negation"),
    ("The service exceeded my expectations", 1, "paraphrase"), ("I would never recommend this phone", 0, "paraphrase"),
    ("Great.", 1, "short"), ("Awful", 0, "short"),
    ("I waited for a very long time at the hotel and the room was dirty and the staff was rude, but in the end "
     "I must say the food they served in the restaurant was absolutely wonderful and I loved it", 1, "long"),
    ("Well, it is a phone", 0, "ambiguous"), ("The food was fine I guess", 1, "ambiguous"),
    ("The battery latency throttled badly", 0, "domain term"), ("The UI is snappy and the API is robust", 1, "domain term"),
    ("It worked as expected", 1, "ambiguous"),
]
ct = pd.DataFrame(challenge, columns=["text", "true", "category"])
ct["pred"] = predict(best, list(ct.text), bestL)
print("\n--- Robustness test ---")
print(ct[["text", "true", "pred", "category"]].to_string(index=False))
wrong = ct[ct.true != ct.pred]
print(f"\nMisclassified: {len(wrong)}/{len(ct)}  by category:\n", wrong.category.value_counts().to_string())

# =====================================================================
# TASK 8 : comparison table + required plots
# =====================================================================
runs = [E1, E2, E3, E4, E5a, E5b]
table = pd.DataFrame([{k: round(r[k], 3) for k in ["acc", "prec", "rec", "f1", "time", "val_loss"]} | {"experiment": r["name"]} for r in runs])
print("\n", table[["experiment", "acc", "prec", "rec", "f1", "time", "val_loss"]].to_string(index=False))

fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
for r in runs[:4]:
    ax[0].plot(r["tr_hist"], marker="o", label=r["name"]); ax[1].plot(r["va_hist"], marker="o", label=r["name"])
ax[0].set_title("Training loss vs epoch"); ax[1].set_title("Validation loss vs epoch"); ax[0].legend(fontsize=6)
ax[2].bar(range(len(runs)), [r["f1"] for r in runs]); ax[2].set_xticks(range(len(runs)))
ax[2].set_xticklabels([r["name"].split(":")[0] for r in runs]); ax[2].set_title("Macro-F1 per experiment")
ax[3].imshow(E5b["cm"], cmap="Blues"); ax[3].set_title("Confusion matrix (E5b)")
for (i, j), v in np.ndenumerate(E5b["cm"]): ax[3].text(j, i, v, ha="center", va="center")
plt.tight_layout(); plt.show()
