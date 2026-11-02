
import os, glob, hashlib, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, f1_score)

DATA_DIR = os.environ.get("DATA_DIR", "UTKFace")
MAX_IMAGES, IMG_SIZE, EPOCHS, BATCH, LR = 2000, 224, 3, 32, 1e-3
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
pd.set_option("display.width", 200)

GENDER = {0: "Male", 1: "Female"}
RACE = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
AGE_NAMES = ["Child", "Teenager", "Young adult", "Middle-aged", "Older adult"]
def age_group(a):                                               # 0-12, 13-19, 20-39, 40-59, 60+
    return 0 if a <= 12 else 1 if a <= 19 else 2 if a <= 39 else 3 if a <= 59 else 4

# ---------------------------------------------------------------- (demo only) synthetic dataset
def make_synthetic(folder="UTKFace_synthetic", n=700):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    ranges = [(1, 12), (13, 19), (20, 39), (40, 59), (60, 90)]
    for i in range(n):
        g = rng.choice(5, p=[.1, .1, .45, .25, .1]); age = int(rng.integers(*ranges[g]))
        img = Image.new("RGB", (96, 96), tuple(int(x) for x in rng.integers(0, 80, 3)))
        shade = 60 + g * 45 + int(rng.integers(-15, 15))                    # older group => brighter "face"
        ImageDraw.Draw(img).ellipse([20, 15, 76, 85], fill=(shade, shade - 20, shade - 30))
        arr = np.clip(np.array(img) + rng.normal(0, 8, (96, 96, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(f"{folder}/{age}_{rng.integers(0,2)}_{rng.integers(0,5)}_2017{i:011d}.jpg")
    Image.new("RGB", (96, 96)).save(f"{folder}/notaname.jpg")                   # malformed filename
    Image.new("RGB", (96, 96)).save(f"{folder}/30_7_1_20170101000000000.jpg")   # invalid gender label
    open(f"{folder}/25_0_0_20170101000000001.jpg", "w").write("corrupt")      # unreadable image
    return folder

if not os.path.isdir(DATA_DIR) or not glob.glob(f"{DATA_DIR}/*.jpg"):
    print(f"[!] '{DATA_DIR}' not found -> creating SYNTHETIC data (demo only)")
    DATA_DIR = make_synthetic()

# =====================================================================
# TASK 1 : parse filenames, validate, metadata
# =====================================================================
files = sorted(glob.glob(f"{DATA_DIR}/*.jpg"))
records, bad = [], dict(malformed=0, invalid_label=0, unreadable=0, duplicate=0)
seen_hash = set()
for f in files:
    parts = os.path.basename(f).split("_")
    if len(parts) < 4 or not all(p.isdigit() for p in parts[:3]):
        bad["malformed"] += 1; continue
    age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
    if not (1 <= age <= 116 and gender in GENDER and race in RACE):
        bad["invalid_label"] += 1; continue
    try:
        Image.open(f).verify()                                                  # readable?
    except Exception:
        bad["unreadable"] += 1; continue
    h = hashlib.md5(open(f, "rb").read()).hexdigest()
    if h in seen_hash:
        bad["duplicate"] += 1; continue
    seen_hash.add(h)
    records.append(dict(path=f, age=age, gender=gender, race=race, age_group=age_group(age)))
meta = pd.DataFrame(records)
meta["gender_name"] = meta.gender.map(GENDER); meta["race_name"] = meta.race.map(RACE)
meta["age_group_name"] = meta.age_group.map(dict(enumerate(AGE_NAMES)))
meta.to_csv("utk_metadata.csv", index=False)
print(f"Total files={len(files)} | valid={len(meta)} | excluded={bad}")

fig, ax = plt.subplots(1, 6, figsize=(15, 3))
for a, (_, r) in zip(ax, meta.sample(6, random_state=1).iterrows()):
    a.imshow(Image.open(r.path)); a.set_title(f"{r.age}y {r.gender_name}\n{r.race_name}", fontsize=8); a.axis("off")
plt.suptitle("Sample images with extracted labels"); plt.show()

# =====================================================================
# TASK 2 : EDA
# =====================================================================
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].hist(meta.age, bins=30); ax[0, 0].set_title("Age distribution")
meta.age_group_name.value_counts().reindex(AGE_NAMES).plot.bar(ax=ax[0, 1], title="Images per age group")
meta.gender_name.value_counts().plot.bar(ax=ax[1, 0], title="Gender label")
meta.race_name.value_counts().plot.bar(ax=ax[1, 1], title="Race/ethnicity label")
plt.tight_layout(); plt.show()
print("\nAge group x gender\n", pd.crosstab(meta.age_group_name, meta.gender_name))
print("\nAge group x race\n", pd.crosstab(meta.age_group_name, meta.race_name))
print("\nMost represented age group:", meta.age_group_name.value_counts().idxmax(),
      "| least:", meta.age_group_name.value_counts().idxmin())

# =====================================================================
# TASK 3 : preprocessing + stratified split 70/15/15
# =====================================================================
if len(meta) > MAX_IMAGES:                                       # keep CPU time manageable
    meta, _ = train_test_split(meta, train_size=MAX_IMAGES, stratify=meta.age_group, random_state=SEED)
train_df, rest = train_test_split(meta, test_size=0.30, stratify=meta.age_group, random_state=SEED)
val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest.age_group, random_state=SEED)
assert not (set(train_df.path) & set(val_df.path)) and not (set(train_df.path) & set(test_df.path)) \
    and not (set(val_df.path) & set(test_df.path)), "overlap between splits!"
print(f"\nSplit sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)} (no file overlap)")
print(pd.DataFrame({"train": train_df.age_group.value_counts(), "val": val_df.age_group.value_counts(),
                    "test": test_df.age_group.value_counts()}).sort_index())
print("Limitation: images of the same person may appear in different splits (no person id in UTKFace).")

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]         # ImageNet statistics
tfm = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class FaceDS(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return tfm(Image.open(r.path).convert("RGB")), int(r.age_group)       # RGB -> resize -> tensor -> normalise
train_dl = DataLoader(FaceDS(train_df), batch_size=BATCH, shuffle=True)
val_dl = DataLoader(FaceDS(val_df), batch_size=BATCH)
test_dl = DataLoader(FaceDS(test_df), batch_size=BATCH)

# =====================================================================
# TASK 4 : transfer learning with ResNet-18 (frozen backbone, new 5-class head)
# =====================================================================
try:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
except Exception as e:
    print("[!] pretrained weights not available (", type(e).__name__, ") -> random init (accuracy will be lower)")
    model = models.resnet18(weights=None)
for p in model.parameters():
    p.requires_grad = False                                       # freeze backbone
model.fc = nn.Linear(model.fc.in_features, 5)                     # new trainable layer
opt = torch.optim.Adam(model.fc.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss()
print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

def run_epoch(dl, train):
    model.train(train); tot_loss = correct = n = 0
    # frozen backbone: keep BatchNorm in eval mode so its statistics are not changed
    if train:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
    with torch.set_grad_enabled(train):
        for x, y in dl:
            out = model(x); loss = lossf(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot_loss / n, correct / n

hist, best_val, best_state, best_ep = [], 1e9, None, 0
for ep in range(1, EPOCHS + 1):
    tl, ta = run_epoch(train_dl, True); vl, va = run_epoch(val_dl, False)
    hist.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
    print(f"epoch {ep}: train loss={tl:.3f} acc={ta:.3f} | val loss={vl:.3f} acc={va:.3f}")
    if vl < best_val:                                              # pick epoch with the VALIDATION set (never the test set)
        best_val, best_ep, best_state = vl, ep, {k: v.clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state); model.eval()
H = pd.DataFrame(hist); H.to_csv("training_history.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(H.epoch, H.train_loss, label="train"); ax[0].plot(H.epoch, H.val_loss, label="val"); ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(H.epoch, H.train_acc, label="train"); ax[1].plot(H.epoch, H.val_acc, label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
plt.tight_layout(); plt.show()

# =====================================================================
# TASK 5 : test-set evaluation
# =====================================================================
probs, ys = [], []
with torch.no_grad():
    for x, y in test_dl:
        probs.append(torch.softmax(model(x), 1)); ys.append(y)
probs, y_true = torch.cat(probs).numpy(), torch.cat(ys).numpy()
y_pred = probs.argmax(1)
acc, bacc = accuracy_score(y_true, y_pred), balanced_accuracy_score(y_true, y_pred)
mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
print(f"\nTEST: accuracy={acc:.3f} balanced_acc={bacc:.3f} macro P={mp:.3f} R={mr:.3f} F1={mf:.3f}")
print(classification_report(y_true, y_pred, labels=range(5), target_names=AGE_NAMES, zero_division=0))
cm = confusion_matrix(y_true, y_pred, labels=range(5))
plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt="d", xticklabels=AGE_NAMES, yticklabels=AGE_NAMES, cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (test)"); plt.show()
rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
print("Strongest group (recall):", AGE_NAMES[rec.argmax()], "| weakest:", AGE_NAMES[rec.argmin()])
pd.DataFrame(classification_report(y_true, y_pred, labels=range(5), target_names=AGE_NAMES, output_dict=True, zero_division=0)).T.to_csv("test_metrics.csv")

# =====================================================================
# TASK 6 : subgroup audit (same predictions)
# =====================================================================
res = test_df.reset_index(drop=True).assign(pred=y_pred, true=y_true)
def audit(col, names):
    rows = []
    for g, d in res.groupby(col):
        rows.append(dict(group=names[g], n=len(d), accuracy=accuracy_score(d.true, d.pred),
                         balanced_acc=balanced_accuracy_score(d.true, d.pred) if d.true.nunique() > 1 else np.nan,
                         macro_f1=f1_score(d.true, d.pred, average="macro", zero_division=0)))
    return pd.DataFrame(rows)
for col, names, fn in [("gender", GENDER, "subgroup_gender.csv"), ("race", RACE, "subgroup_race.csv")]:
    t = audit(col, names); t.to_csv(fn, index=False); print(f"\nSubgroup audit by {col}\n", t.round(3).to_string(index=False))
    plt.figure(figsize=(5, 3)); plt.bar(t.group, t.accuracy); plt.title(f"Accuracy by {col} label"); plt.ylim(0, 1); plt.show()
print("NOTE: small n => unstable metrics; differences alone do not prove bias (class mix, image quality, labels differ).")

# =====================================================================
# TASK 7 : error analysis + LIME + Grad-CAM
# =====================================================================
wrong = res[res.pred != res.true].head(3)
fig, ax = plt.subplots(1, max(len(wrong), 1), figsize=(10, 3.2)); ax = np.atleast_1d(ax)
for a, (_, r) in zip(ax, wrong.iterrows()):
    a.imshow(Image.open(r.path)); a.set_title(f"actual: {AGE_NAMES[r.true]}\npred: {AGE_NAMES[r.pred]}", fontsize=8); a.axis("off")
plt.suptitle("Misclassified test images"); plt.show()

def preprocess_np(batch):                                          # numpy HxWx3 (0..255) -> normalised tensor
    x = torch.tensor(np.asarray(batch), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)

def predict_fn(batch):                                             # LIME needs: images -> class probabilities
    with torch.no_grad():
        return torch.softmax(model(preprocess_np(batch)), 1).numpy()

sample = (wrong.iloc[0] if len(wrong) else res.iloc[0])
img = np.array(Image.open(sample.path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
p = predict_fn(img[None])[0]; pred_cls = int(p.argmax())
print(f"\nLIME image: actual={AGE_NAMES[sample.true]} predicted={AGE_NAMES[pred_cls]} confidence={p.max():.2f}")
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    expl = lime_image.LimeImageExplainer().explain_instance(img, predict_fn, top_labels=5, hide_color=0, num_samples=300)
    lime_img, mask = expl.get_image_and_mask(pred_cls, positive_only=False, num_features=6, hide_rest=False)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    ax[0].imshow(img); ax[0].set_title("Original")
    ax[1].imshow(mark_boundaries(lime_img / 255.0 if lime_img.max() > 1 else lime_img, mask)); ax[1].set_title("LIME (green=supports, red=against)")
except ImportError:
    print("lime not installed (pip install lime) -> skipping LIME"); fig, ax = plt.subplots(1, 3, figsize=(11, 3.8)); ax[0].imshow(img)

# Grad-CAM written by hand: weight layer4 feature maps by the mean gradient of the class score
feats, grads = {}, {}
h1 = model.layer4.register_forward_hook(lambda m, i, o: feats.update(a=o))
h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: grads.update(a=go[0]))
x = preprocess_np(img[None]).requires_grad_(True)                    # requires_grad so a graph exists with a frozen backbone
model(x)[0, pred_cls].backward()
w = grads["a"].mean(dim=(2, 3), keepdim=True)                       # importance of each feature map
cam = torch.relu((w * feats["a"]).sum(1))[0].detach().numpy()
cam = np.array(Image.fromarray(cam).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
h1.remove(); h2.remove()
ax[-1].imshow(img); ax[-1].imshow(cam, cmap="jet", alpha=0.45); ax[-1].set_title("Grad-CAM overlay")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()

# =====================================================================
# TASK 8 : results table
# =====================================================================
summary = pd.DataFrame([dict(valid_images=len(meta), train=len(train_df), val=len(val_df), test=len(test_df),
                             test_accuracy=round(acc, 3), balanced_accuracy=round(bacc, 3), macro_f1=round(mf, 3), best_val_epoch=best_ep)])
print("\nRESULTS TABLE\n", summary.to_string(index=False)); summary.to_csv("results_summary.csv", index=False)
print("Reminder: this is age-group classification only - NOT genetic-disorder detection or clinical diagnosis.")
