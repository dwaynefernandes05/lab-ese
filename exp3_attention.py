"""
EXP 3 - Scaled dot-product attention from scratch (NumPy + Matplotlib)
Attention(Q,K,V) = softmax(Q K^T / sqrt(dk)) V
Run:  python exp3_attention.py
"""
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

# ---------- softmax (numerically stable: subtract the row max) ----------
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# ---------- attention ----------
def attention(Q, K, V, scale=True):
    dk = K.shape[-1]
    scores = Q @ K.T                          # 1. similarity   (n x n)
    if scale:
        scores = scores / np.sqrt(dk)         # 2. scaling
    weights = softmax(scores)                 # 3. normalisation (rows sum to 1)
    output = weights @ V                      # 4. weighted sum of values
    return scores, weights, output

def show_heatmap(w, labels, title, ax=None):
    ax = ax or plt.gca()
    im = ax.imshow(w, cmap="viridis")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", color="w")
    ax.set_xlabel("Key"); ax.set_ylabel("Query"); ax.set_title(title)
    plt.colorbar(im, ax=ax)

# =====================================================================
# 1. Worked example from the manual (Q = K)
# =====================================================================
Q = K = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
V = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
scores, weights, out = attention(Q, K, V)
print("Q K^T:\n", Q @ K.T)
print("Scaled scores:\n", scores)
print("Attention weights:\n", weights)
print("Row sums:", weights.sum(axis=1))
print("Output:\n", out)

# =====================================================================
# 2. With vs without scaling (use bigger dimension so the difference shows)
# =====================================================================
rng = np.random.default_rng(0)
d = 64
Qb, Kb, Vb = rng.normal(size=(4, d)), rng.normal(size=(4, d)), rng.normal(size=(4, d))
_, w_scaled, _ = attention(Qb, Kb, Vb, scale=True)
_, w_unscaled, _ = attention(Qb, Kb, Vb, scale=False)
print("\nMax weight per row (scaled)  :", w_scaled.max(axis=1))
print("Max weight per row (unscaled):", w_unscaled.max(axis=1), "<- more peaked (close to 1)")

# =====================================================================
# 3. Changing V changes the output but NOT the weights
# =====================================================================
_, w2, out2 = attention(Q, K, V * 10)
print("\nWeights unchanged when V changes:", np.allclose(weights, w2))
print("Output changed:", not np.allclose(out, out2))

# =====================================================================
# 4. Self-attention with real token labels: Q = XWq, K = XWk, V = XWv
# =====================================================================
tokens = ["The", "cat", "sat", "down"]
X = rng.normal(size=(4, 8))                      # 4 tokens, embedding dim 8
Wq, Wk, Wv = (rng.normal(size=(8, 4)) for _ in range(3))
_, w_self, out_self = attention(X @ Wq, X @ Wk, X @ Wv)
print("\nSelf-attention weights:\n", w_self)
print("Output shape:", out_self.shape)

# =====================================================================
# 5. Multi-head attention (bonus): split dim into heads, concat outputs
# =====================================================================
def multi_head(X, n_heads=2, d_k=4):
    outs = []
    for _ in range(n_heads):
        Wq, Wk, Wv = (rng.normal(size=(X.shape[1], d_k)) for _ in range(3))
        outs.append(attention(X @ Wq, X @ Wk, X @ Wv)[2])
    return np.concatenate(outs, axis=1)
print("Multi-head output shape:", multi_head(X).shape)

# =====================================================================
# Plots
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
show_heatmap(weights, ["t1", "t2", "t3"], "Worked example", ax[0])
show_heatmap(w_unscaled, list("abcd"), "d=64 WITHOUT scaling", ax[1])
show_heatmap(w_self, tokens, "Self-attention (tokens)", ax[2])
plt.tight_layout(); plt.show()
