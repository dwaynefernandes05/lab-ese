"""
EXP 2 - Generative image models: Autoencoder, VAE and GAN  (+ one modification each, compared)
Run:  python exp2_generative_models.py          (CPU, PyTorch, ~2-3 min)

Dataset: MNIST if torchvision already has it in ./data, otherwise sklearn 'digits' (8x8, offline).
Models use simple fully-connected layers so they are fast on CPU and easy to explain.

Baseline vs Modified (change ONLY one thing):
  AE  : latent_dim 2  -> 16
  VAE : latent_dim 2  -> 16
  GAN : learning rate 0.0002 -> 0.001
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

torch.manual_seed(0); np.random.seed(0)
EPOCHS, BATCH = 50, 64

# ---------------------------------------------------------------- data
def load_data():
    try:
        from torchvision import datasets
        ds = datasets.MNIST("./data", train=True, download=False)       # no internet needed if cached
        x = ds.data[:6000].float().div(255).view(-1, 784)
        side = 28
    except Exception:
        from sklearn.datasets import load_digits
        x = torch.tensor(load_digits().data, dtype=torch.float32) / 16.0   # 1797 x 64, range [0,1]
        side = 8
    return x, side

X, SIDE = load_data()
D = X.shape[1]
n_train = int(0.85 * len(X)); Xtr, Xte = X[:n_train], X[n_train:]
loader = lambda: torch.utils.data.DataLoader(Xtr, batch_size=BATCH, shuffle=True)
print(f"Data: {tuple(X.shape)}  image={SIDE}x{SIDE}  train={len(Xtr)} test={len(Xte)}")

# ---------------------------------------------------------------- models
class AE(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, D), nn.Sigmoid())
    def forward(self, x):
        return self.dec(self.enc(x))

class VAE(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(nn.Linear(D, 128), nn.ReLU())
        self.mu, self.logvar = nn.Linear(128, latent), nn.Linear(128, latent)    # encoder outputs mean & log-variance
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, D), nn.Sigmoid())
    def forward(self, x):
        h = self.enc(x); mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)                  # reparameterisation trick
        return self.dec(z), mu, logvar

def vae_loss(recon, x, mu, logvar):
    rec = F.binary_cross_entropy(recon, x, reduction="sum")   # (manual uses MSE; BCE works better on tiny 8x8 images)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())                 # KL( q(z|x) || N(0,I) )
    return (rec + kl) / x.size(0)

class Generator(nn.Module):
    def __init__(self, zdim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(zdim, 128), nn.LeakyReLU(0.2), nn.Linear(128, D), nn.Sigmoid())
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

n_params = lambda *ms: sum(p.numel() for m in ms for p in m.parameters() if p.requires_grad)

# ---------------------------------------------------------------- training functions
def train_ae(latent):
    torch.manual_seed(0)
    m = AE(latent); opt = torch.optim.Adam(m.parameters(), lr=1e-3); hist = []; t0 = time.time()
    for ep in range(EPOCHS):
        tot = 0
        for x in loader():
            loss = F.mse_loss(m(x), x)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(x)
        hist.append(tot / len(Xtr))
    return m, hist, time.time() - t0

def train_vae(latent):
    torch.manual_seed(0)
    m = VAE(latent); opt = torch.optim.Adam(m.parameters(), lr=1e-3); hist = []; t0 = time.time()
    for ep in range(EPOCHS):
        tot = 0
        for x in loader():
            recon, mu, lv = m(x); loss = vae_loss(recon, x, mu, lv)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(x)
        hist.append(tot / len(Xtr))
    return m, hist, time.time() - t0

def train_gan(lr, zdim=32):
    torch.manual_seed(0)
    G, Dn = Generator(zdim), Discriminator()
    oG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(Dn.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCELoss(); gh, dh = [], []; t0 = time.time()
    for ep in range(EPOCHS):
        g_tot = d_tot = 0
        for real in loader():
            b = len(real); ones, zeros = torch.ones(b, 1), torch.zeros(b, 1)
            fake = G(torch.randn(b, zdim))
            # --- train discriminator: real -> 1, fake -> 0
            d_loss = bce(Dn(real), ones) + bce(Dn(fake.detach()), zeros)
            oD.zero_grad(); d_loss.backward(); oD.step()
            # --- train generator: make D say "real"
            g_loss = bce(Dn(fake), ones)
            oG.zero_grad(); g_loss.backward(); oG.step()
            g_tot += g_loss.item() * b; d_tot += d_loss.item() * b
        gh.append(g_tot / len(Xtr)); dh.append(d_tot / len(Xtr))
    return (G, Dn), (gh, dh), time.time() - t0

# ---------------------------------------------------------------- metrics
def recon_metrics(recon, x):
    """MSE, PSNR, SSIM averaged over 100 test images"""
    r = recon[:100].detach().numpy().reshape(-1, SIDE, SIDE); o = x[:100].numpy().reshape(-1, SIDE, SIDE)
    mse = float(((r - o) ** 2).mean())
    return mse, np.mean([psnr(a, b, data_range=1) for a, b in zip(o, r)]), \
        np.mean([ssim(a, b, data_range=1) for a, b in zip(o, r)])

def show(imgs, title, ax):
    n = 8; imgs = imgs[:n * 2].detach().numpy().reshape(-1, SIDE, SIDE)
    grid = np.vstack([np.hstack(imgs[:n]), np.hstack(imgs[n:2 * n])])
    ax.imshow(grid, cmap="gray"); ax.set_title(title); ax.axis("off")

# =====================================================================
# 1. AUTOENCODER : baseline latent=2 vs modified latent=16
# =====================================================================
print("\n--- AUTOENCODER ---")
rows = []
fig, ax = plt.subplots(2, 3, figsize=(13, 6)); show(Xte, "Original", ax[0, 0]); ax[1, 0].axis("off")
for k, lat in enumerate([2, 16]):
    m, h, t = train_ae(lat)
    mse, ps, ss = recon_metrics(m(Xte), Xte)
    rows.append((f"AE latent={lat}", n_params(m), round(t, 1), round(h[-1], 4), round(mse, 4), round(ps, 1), round(ss, 3)))
    show(m(Xte), f"AE latent={lat} reconstructions", ax[0, k + 1]); ax[1, k + 1].plot(h); ax[1, k + 1].set_title("train loss (MSE)")
plt.tight_layout(); plt.show()

# =====================================================================
# 2. VAE : baseline latent=2 vs modified latent=16 ; also GENERATE new images from noise
# =====================================================================
print("--- VAE ---")
fig, ax = plt.subplots(2, 3, figsize=(13, 6))
for k, lat in enumerate([2, 16]):
    m, h, t = train_vae(lat)
    recon, _, _ = m(Xte); mse, ps, ss = recon_metrics(recon, Xte)
    rows.append((f"VAE latent={lat}", n_params(m), round(t, 1), round(h[-1], 2), round(mse, 4), round(ps, 1), round(ss, 3)))
    with torch.no_grad():
        gen = m.dec(torch.randn(16, lat))                                # sample z ~ N(0,I) -> decode
    show(gen, f"VAE latent={lat}: NEW images", ax[0, k]); ax[1, k].plot(h); ax[1, k].set_title("train loss (recon+KL)")
ax[0, 2].axis("off"); ax[1, 2].axis("off")
plt.tight_layout(); plt.show()

# =====================================================================
# 3. GAN : baseline lr=0.0002 vs modified lr=0.001
# =====================================================================
print("--- GAN ---")
fig, ax = plt.subplots(2, 2, figsize=(10, 6)); gan_rows = []
for k, lr in enumerate([0.0002, 0.001]):
    (G, Dn), (gh, dh), t = train_gan(lr)
    t1 = time.time()
    with torch.no_grad():
        fake = G(torch.randn(16, 32))
    inf_t = time.time() - t1
    # no pretrained Inception on CPU/offline -> cheap quality proxy: mean SSIM of each fake vs its nearest real image
    f = fake.numpy().reshape(-1, SIDE, SIDE); real = Xte[:300].numpy()
    near = [real[np.argmin(((real - x.ravel()) ** 2).sum(1))].reshape(SIDE, SIDE) for x in f]
    prox = np.mean([ssim(a, b, data_range=1) for a, b in zip(f, near)])
    gan_rows.append((f"GAN lr={lr}", n_params(G, Dn), round(t, 1), round(gh[-1], 3), round(dh[-1], 3), round(prox, 3), f"{inf_t*1000:.1f} ms"))
    show(fake, f"GAN lr={lr}", ax[0, k]); ax[1, k].plot(gh, label="G loss"); ax[1, k].plot(dh, label="D loss"); ax[1, k].legend()
plt.tight_layout(); plt.show()

# =====================================================================
# COMPARISON TABLES
# =====================================================================
import pandas as pd
print("\nAE / VAE comparison (lower MSE, higher PSNR/SSIM = better reconstruction)")
print(pd.DataFrame(rows, columns=["Model", "Params", "Time(s)", "FinalLoss", "MSE", "PSNR", "SSIM"]).to_string(index=False))
print("\nGAN comparison")
print(pd.DataFrame(gan_rows, columns=["Model", "Params", "Time(s)", "G loss", "D loss", "SSIM-to-nearest-real", "Inference(16 imgs)"]).to_string(index=False))
