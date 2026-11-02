import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from collections import Counter
from sklearn.datasets import load_digits
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
summary = []   # rows for the final comparison table (Task 5)


# PART A : TABULAR
try:
    df = pd.read_csv("titanic.csv")
except FileNotFoundError:               # make a Titanic-like dataset
    n = 500
    df = pd.DataFrame({
        "Survived": rng.integers(0, 2, n),
        "Pclass": rng.integers(1, 4, n),
        "Sex": rng.choice(["male", "female"], n),
        "Age": rng.normal(30, 14, n).round(0).clip(1, 80),
        "Fare": rng.exponential(30, n).round(2),
        "Embarked": rng.choice(["S", "C", "Q"], n),
    })
    df.loc[rng.choice(n, 60, replace=False), "Age"] = np.nan         # missing values
    df.loc[rng.choice(n, 5, replace=False), "Embarked"] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "Fare"] = 600            # outliers
    df = pd.concat([df, df.sample(20, random_state=1)])              # duplicates

# ---- Task 1 : profiling
print("Shape:", df.shape)
print(df.dtypes)
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df.describe())
orig_rows, orig_missing, orig_dups = len(df), int(df.isnull().sum().sum()), int(df.duplicated().sum())

clean = df.copy()
# ---- Task 2 : cleaning
clean = clean.drop_duplicates()                                       # duplicates
for col in clean.columns:                                             # missing values
    if pd.api.types.is_numeric_dtype(clean[col]):
        clean[col] = clean[col].fillna(clean[col].median())           # numeric -> median
    else:
        clean[col] = clean[col].fillna(clean[col].mode()[0])          # categorical -> mode

# outliers using IQR rule on Fare (clip instead of delete)
q1, q3 = clean["Fare"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("Outliers in Fare:", ((clean["Fare"] < low) | (clean["Fare"] > high)).sum())
clean["Fare"] = clean["Fare"].clip(low, high)

clean["Sex"] = clean["Sex"].map({"male": 0, "female": 1})             # label encoding
clean = pd.get_dummies(clean, columns=["Embarked"], dtype=int)        # one-hot encoding
num_cols = ["Age", "Fare"]
clean[num_cols] = StandardScaler().fit_transform(clean[num_cols])     # feature scaling
clean = clean.drop_duplicates()       # imputation can create new duplicate rows

# visualisations: histogram, box plot, correlation heatmap
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df["Fare"].hist(ax=ax[0], bins=30); ax[0].set_title("Fare histogram (original)")
sns.boxplot(data=df[["Age", "Fare"]], ax=ax[1]); ax[1].set_title("Box plot (original)")
sns.heatmap(clean.corr(), annot=True, fmt=".1f", ax=ax[2]); ax[2].set_title("Correlation (cleaned)")
plt.tight_layout(); plt.show()

summary.append(["Tabular", orig_rows, len(clean), orig_missing, int(clean.isnull().sum().sum()),
                orig_dups, int(clean.duplicated().sum())])


print("\n========== B. TEXT ==========")
reviews = [
    "I LOVED this movie!!! <br>Best film ever http://imdb.com/x",
    "Terrible plot, waste of time... <b>never</b> again :(",
    "The acting was great, and the story was moving.",
    "The acting was great, and the story was moving.",          # duplicate
    "Worst movie. Boring, boring, boring!!!",
    "Loved the soundtrack & the visuals #amazing",
    "Not good. The characters were poorly written, running away from the plot.",
    "A wonderful, heartwarming film. Watching it again tomorrow!",
]
text_df = pd.DataFrame({"text": reviews})
print("Dataset shape:", text_df.shape, "| duplicates:", text_df.duplicated().sum())

try:
    from nltk.stem import PorterStemmer
    stem = PorterStemmer().stem            # stemmer needs no download
except Exception:
    stem = lambda w: re.sub(r"(ing|ed|ly|s)$", "", w)   # very crude fallback

def clean_text(t):
    t = t.lower()                                  # lowercase
    t = re.sub(r"<.*?>", " ", t)                   # HTML tags
    t = re.sub(r"http\S+|www\S+", " ", t)          # URLs
    t = re.sub(r"[^a-z\s]", " ", t)                # punctuation + special chars + digits
    tokens = t.split()                             # tokenization
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]   # stopword removal
    return [stem(w) for w in tokens]               # stemming

text_df = text_df.drop_duplicates().reset_index(drop=True)        # duplicate removal
text_df["tokens"] = text_df["text"].apply(clean_text)
text_df["clean"] = text_df["tokens"].apply(" ".join)
print(text_df[["text", "clean"]])

all_words = [w for toks in text_df["tokens"] for w in toks]
raw_words = [w for t in reviews for w in t.lower().split()]
print("Vocabulary: raw =", len(set(raw_words)), "| cleaned =", len(set(all_words)))
print("Total tokens: raw =", len(raw_words), "| cleaned =", len(all_words))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
top = Counter(all_words).most_common(20)
ax[0].barh([w for w, _ in top][::-1], [c for _, c in top][::-1]); ax[0].set_title("Top frequent words")
ax[1].hist([len(t.split()) for t in reviews], alpha=.6, label="original")
ax[1].hist(text_df["tokens"].apply(len), alpha=.6, label="cleaned")
ax[1].set_title("Sentence length distribution"); ax[1].legend()
plt.tight_layout(); plt.show()
try:                                              # optional word cloud
    from wordcloud import WordCloud
    plt.imshow(WordCloud(background_color="white").generate(" ".join(all_words)))
    plt.axis("off"); plt.title("Word cloud"); plt.show()
except ImportError:
    print("wordcloud not installed -> skipped (bar chart above shows the same info)")

summary.append(["Text", len(reviews), len(text_df), 0, 0,
                len(reviews) - len(set(reviews)), 0])

# =====================================================================
# PART C : IMAGE
# =====================================================================
print("\n========== C. IMAGE ==========")
digits = load_digits()
images = list(digits.images[:300])                  # 8x8 grayscale, values 0..16
labels = list(digits.target[:300])
images += [None, None]; labels += [0, 1]            # fake corrupted images
images += [images[0].copy(), images[1].copy()]; labels += [labels[0], labels[1]]  # duplicates

# Task 1 profiling
shapes = Counter(None if im is None else im.shape for im in images)
print("Image shapes:", shapes, "| total:", len(images))

# Task 4 cleaning
good_imgs, good_lbls, seen = [], [], set()
for im, lb in zip(images, labels):
    if im is None:                                        # remove corrupted
        continue
    h = hash(im.tobytes())
    if h in seen:                                         # remove duplicates
        continue
    seen.add(h)
    im = (im / 16.0 * 255).astype(np.uint8)
    im = cv2.resize(im, (32, 32))                         # resize
    good_imgs.append(im.astype(np.float32) / 255.0)       # normalize to 0-1
    good_lbls.append(lb)
good_imgs = np.array(good_imgs)
print("After cleaning:", good_imgs.shape, "| pixel range:", good_imgs.min(), "-", good_imgs.max())

fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
ax[0].imshow(np.hstack(list(good_imgs[:6])), cmap="gray"); ax[0].set_title("Sample images"); ax[0].axis("off")
ax[1].hist(good_imgs.ravel(), bins=30); ax[1].set_title("Pixel intensity histogram")
ax[2].bar(*np.unique(good_lbls, return_counts=True)); ax[2].set_title("Class distribution")
plt.tight_layout(); plt.show()

summary.append(["Image", len(images), len(good_imgs), 2, 0, 2, 0])

# =====================================================================
# TASK 5 : comparison table
# =====================================================================
table = pd.DataFrame(summary, columns=["Dataset", "Samples(orig)", "Samples(clean)",
                                       "Missing(orig)", "Missing(clean)", "Dups(orig)", "Dups(clean)"])
print("\n========== COMPARISON ==========")
print(table.to_string(index=False))
