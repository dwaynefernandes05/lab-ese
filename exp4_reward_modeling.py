"""
EXP 4 - Reward modelling (preference learning) + Q-learning on a 4x4 GridWorld
Run:  python exp4_reward_modeling.py        (CPU, PyTorch, ~1-2 min)

Layout     S . . .      actions: 0=Up 1=Down 2=Left 3=Right
           . X . .      env reward: +10 goal, -1 normal move, -5 hit obstacle
           . . . .
           . . . G

Pipeline:  preference pairs -> reward model (Bradley-Terry loss) -> Q-learning on the learned reward
The file has:  BASE PIPELINE  +  5 modifications (run any one of them in the exam)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
N, GOAL, OBST, START = 4, 15, 5, 0
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # Up Down Left Right
ARROWS = ["^", "v", "<", ">"]

# =====================================================================
# 1. ENVIRONMENT
# =====================================================================
def step(s, a):
    """returns next_state, true_reward, done"""
    r, c = divmod(s, N)
    nr, nc = r + MOVES[a][0], c + MOVES[a][1]
    if not (0 <= nr < N and 0 <= nc < N):            # wall -> stay
        return s, -1, False
    ns = nr * N + nc
    if ns == OBST:                                   # obstacle -> stay, -5
        return s, -5, False
    if ns == GOAL:
        return ns, 10, True
    return ns, -1, False

def dist(s):                                         # Manhattan distance to goal
    r, c = divmod(s, N)
    return abs(r - 3) + abs(c - 3)

# =====================================================================
# 2. PREFERENCE GENERATOR  (simulated human)
# =====================================================================
def prefer_action(s, a1, a2):
    """True if a1 is preferred: the action that gets closer to the goal wins"""
    d1, d2 = dist(step(s, a1)[0]), dist(step(s, a2)[0])
    if d1 == d2:
        return None                                   # tie -> skip pair
    return d1 < d2

def make_pairs(n=800, noise=0.0, states=range(16), pref_fn=prefer_action):
    """returns list of (state, winner_action, loser_action). noise = prob. of flipping label"""
    pairs = []
    states = [s for s in states if s not in (GOAL, OBST)]
    while len(pairs) < n:
        s = int(np.random.choice(states))
        a1, a2 = np.random.choice(4, 2, replace=False)
        p = pref_fn(s, a1, a2)
        if p is None:
            continue
        w, l = (a1, a2) if p else (a2, a1)
        if np.random.rand() < noise:                  # label noise (Modification 1)
            w, l = l, w
        pairs.append((s, int(w), int(l)))
    return pairs

# =====================================================================
# 3. REWARD MODEL  r(s,a)  -> scalar.   input = one-hot(state)+one-hot(action)
# =====================================================================
def encode(s, a):
    x = torch.zeros(20); x[s] = 1; x[16 + a] = 1
    return x

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_reward_model(pairs, epochs=150, lr=0.01):
    """Bradley-Terry loss:  -log sigmoid( r(winner) - r(loser) ).
    returns model, loss curve, train acc, test acc (80/20 split)"""
    np.random.shuffle(pairs)
    cut = int(0.8 * len(pairs))
    tr, te = pairs[:cut], pairs[cut:]
    def tens(ps):
        W = torch.stack([encode(s, w) for s, w, l in ps])
        L = torch.stack([encode(s, l) for s, w, l in ps])
        return W, L
    Wtr, Ltr = tens(tr); Wte, Lte = tens(te)
    model = RewardModel(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        loss = -F.logsigmoid(model(Wtr) - model(Ltr)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = lambda W, L: (model(W) > model(L)).float().mean().item()
        return model, losses, acc(Wtr, Ltr), acc(Wte, Lte)

def reward_table(model, center=True):
    """16x4 table of learned rewards r(s,a).
    The Bradley-Terry loss only sees DIFFERENCES between rewards, so the absolute level
    of r is arbitrary. center=True shifts the table so the best value is 0 and all others
    are negative (like a step cost) -> the agent is motivated to FINISH the episode.
    If rewards were left positive the agent could loop forever to farm them (see Mod 4)."""
    with torch.no_grad():
        t = np.array([[model(encode(s, a)).item() for a in range(4)] for s in range(16)])
    return t - t.max() if center else t

# =====================================================================
# 4. Q-LEARNING  (reward_fn decides which reward the agent sees)
# =====================================================================
def q_learning(reward_fn, episodes=600, alpha=0.2, gamma=0.9, eps=0.2, seed=0, max_steps=30):
    rng = np.random.default_rng(seed)
    Q = np.zeros((16, 4)); curve = []
    for _ in range(episodes):
        s, total = START, 0
        for _ in range(max_steps):
            a = rng.integers(4) if rng.random() < eps else int(Q[s].argmax())
            ns, true_r, done = step(s, a)
            r = reward_fn(s, a, true_r)                              # <- learned or env reward
            target = r + (0 if done else gamma * Q[ns].max())
            Q[s, a] += alpha * (target - Q[s, a])
            total += true_r; s = ns
            if done:
                break
        curve.append(total)
    return Q, curve

def evaluate(Q, rm_table=None):
    """follow greedy policy from START. returns dict of metrics"""
    s, path, true_total, model_total = START, [START], 0, 0
    for _ in range(20):
        a = int(Q[s].argmax()); ns, r, done = step(s, a)
        true_total += r
        if rm_table is not None:
            model_total += rm_table[s, a]
        s = ns; path.append(s)
        if done:
            break
    return dict(success=int(s == GOAL), steps=len(path) - 1, true_reward=true_total,
                model_reward=round(float(model_total), 2), path=path)

env_reward_fn = lambda s, a, true_r: true_r
def learned_reward_fn(table):
    return lambda s, a, true_r: table[s, a]

def show_policy(Q, title=""):
    print(title)
    for r in range(N):
        print("  " + " ".join("X" if r * N + c == OBST else "G" if r * N + c == GOAL
                              else ARROWS[int(Q[r * N + c].argmax())] for c in range(N)))

def smooth(x, k=30):
    return np.convolve(x, np.ones(k) / k, mode="valid")

# =====================================================================
# 5. BASE PIPELINE (always run first, as the manual asks)
# =====================================================================
print("=" * 60, "\nBASE PIPELINE (clean preferences)")
pairs = make_pairs(800)
rm, losses, tr_acc, te_acc = train_reward_model(pairs)
tab = reward_table(rm)
print(f"Reward model: train acc={tr_acc:.2f}  test acc={te_acc:.2f}  final loss={losses[-1]:.3f}")
Q_base, curve_base = q_learning(learned_reward_fn(tab))
res = evaluate(Q_base, tab); print("Result:", res); show_policy(Q_base, "Policy:")

# =====================================================================
# MODIFICATION 1 : noisy preferences
# =====================================================================
def modification1(noise_levels=(0.0, 0.1, 0.3, 0.5)):
    print("\n" + "=" * 60, "\nMOD 1 - Noisy human preferences")
    plt.figure(figsize=(10, 3.5)); rows = []
    for nz in noise_levels:
        m, ls, tr, te = train_reward_model(make_pairs(800, noise=nz))     # separate RM per noise level
        Q, cv = q_learning(learned_reward_fn(reward_table(m)))
        r = evaluate(Q); rows.append((nz, round(te, 2), r["success"], r["steps"], r["true_reward"]))
        plt.subplot(1, 2, 1); plt.plot(ls, label=f"noise={nz}")
        plt.subplot(1, 2, 2); plt.plot(smooth(cv), label=f"noise={nz}")
    plt.subplot(1, 2, 1); plt.title("Reward-model loss"); plt.legend()
    plt.subplot(1, 2, 2); plt.title("RL learning curve (true reward)"); plt.legend()
    plt.tight_layout(); plt.show()
    print("noise | RM test acc | success | steps | true reward")
    for row in rows: print(row)

# =====================================================================
# MODIFICATION 2 : multi-objective preferences
# =====================================================================
def make_multi_pref(weights):
    """3 normalised factors in [0,1]:  progress, obstacle-avoidance, efficiency (no bump into wall)"""
    def factors(s, a):
        ns, _, _ = step(s, a)
        progress = (dist(s) - dist(ns) + 1) / 2                        # -1,0,1 -> 0,.5,1
        r, c = divmod(s, N); nr, nc = r + MOVES[a][0], c + MOVES[a][1]
        hit_obst = (0 <= nr < N and 0 <= nc < N and nr * N + nc == OBST)
        hit_wall = not (0 <= nr < N and 0 <= nc < N)
        avoid = 0.0 if hit_obst else 1.0                               # obstacle avoidance
        efficient = 0.0 if (hit_wall or hit_obst) else 1.0             # wasted move = bad
        return np.array([progress, avoid, efficient])
    def pref(s, a1, a2):
        s1, s2 = factors(s, a1) @ weights, factors(s, a2) @ weights    # weighted score
        return None if abs(s1 - s2) < 1e-9 else s1 > s2
    return pref

def modification2():
    print("\n" + "=" * 60, "\nMOD 2 - Multi-objective preferences  (progress, obstacle, efficiency)")
    for name, w in {"goal-focused": [0.8, 0.1, 0.1], "safety-focused": [0.1, 0.8, 0.1]}.items():
        m, ls, tr, te = train_reward_model(make_pairs(800, pref_fn=make_multi_pref(np.array(w))))
        tab2 = reward_table(m); Q, _ = q_learning(learned_reward_fn(tab2))
        print(f"\n{name} weights={w}  RM test acc={te:.2f}  ->", {k: v for k, v in evaluate(Q).items()})
        show_policy(Q, "Policy:")

# =====================================================================
# MODIFICATION 3 : trajectory-level preferences
# =====================================================================
def rollout(s, length=4):
    traj = []
    for _ in range(length):
        a = int(np.random.randint(4)); traj.append((s, a)); s = step(s, a)[0]
    return traj, s

def traj_score(traj, end_state):                 # human: ends closer to goal & fewer bumps
    bumps = sum(step(s, a)[0] == s for s, a in traj)
    return -dist(end_state) - 0.5 * bumps

def modification3(n=600):
    print("\n" + "=" * 60, "\nMOD 3 - Trajectory-level preferences")
    data = []
    while len(data) < n:
        s0 = int(np.random.choice([s for s in range(16) if s not in (GOAL, OBST)]))
        t1, e1 = rollout(s0); t2, e2 = rollout(s0)
        sc1, sc2 = traj_score(t1, e1), traj_score(t2, e2)
        if sc1 != sc2:
            data.append((t1, t2) if sc1 > sc2 else (t2, t1))               # (preferred, rejected)
    print("Example preferred trajectory:", data[0][0], "\nExample rejected  trajectory:", data[0][1])
    model = RewardModel(); opt = torch.optim.Adam(model.parameters(), lr=0.01)
    tot = lambda tr: model(torch.stack([encode(s, a) for s, a in tr])).sum()   # trajectory score = sum of step rewards
    cut = int(0.8 * n); losses = []
    for _ in range(100):
        loss = sum(-F.logsigmoid(tot(w) - tot(l)) for w, l in data[:cut]) / cut
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    with torch.no_grad():
        acc = np.mean([float(tot(w) > tot(l)) for w, l in data[cut:]])
    print(f"Trajectory preference test accuracy = {acc:.2f}")
    Q, _ = q_learning(learned_reward_fn(reward_table(model)))
    print("RL result:", evaluate(Q)); show_policy(Q, "Policy:")

# =====================================================================
# MODIFICATION 4 : reward hacking + mitigation
# =====================================================================
def modification4():
    print("\n" + "=" * 60, "\nMOD 4 - Reward hacking")
    # weakness: preference data only from the top row (states 0-3) -> RM knows nothing about the rest
    pairs_small = make_pairs(300, states=[0, 1, 2, 3])
    m, _, _, te = train_reward_model(pairs_small)
    tab4 = reward_table(m, center=False)
    tab4 = tab4 - tab4.min() + 0.5            # RM is only defined up to a shift (BT loss uses differences) -> all positive
    Q, _ = q_learning(learned_reward_fn(tab4))
    r = evaluate(Q, tab4); print("BEFORE mitigation:", r)
    print("  -> agent reaches goal?", bool(r["success"]), "| model reward collected:", r["model_reward"])
    show_policy(Q, "Policy (hacked):")
    # mitigation: normalise + clip the learned reward and MIX it with the env reward (goal bonus)
    z = np.clip((tab4 - tab4.mean()) / (tab4.std() + 1e-8), -1, 1)
    mixed = lambda s, a, true_r: 0.5 * z[s, a] + true_r
    Q2, _ = q_learning(mixed)
    r2 = evaluate(Q2, tab4); print("AFTER mitigation :", r2); show_policy(Q2, "Policy (mitigated):")

# =====================================================================
# MODIFICATION 5 : three-agent comparison (several seeds)
# =====================================================================
def modification5(seeds=(0, 1, 2, 3, 4)):
    print("\n" + "=" * 60, "\nMOD 5 - Three agents")
    results = {"A env reward": [], "B clean RM": [], "C noisy+small-data RM": []}
    curves = {k: [] for k in results}
    for sd in seeds:
        np.random.seed(sd); torch.manual_seed(sd)
        tabB = reward_table(train_reward_model(make_pairs(800))[0])
        tabC = reward_table(train_reward_model(make_pairs(60, noise=0.3))[0])   # imperfect feedback
        for name, fn in zip(results, [env_reward_fn, learned_reward_fn(tabB), learned_reward_fn(tabC)]):
            Q, cv = q_learning(fn, seed=sd); results[name].append(evaluate(Q)); curves[name].append(cv)
    print("agent | success rate | avg steps | avg true reward")
    for k, v in results.items():
        print(k, "|", np.mean([x["success"] for x in v]), "|", np.mean([x["steps"] for x in v]),
              "|", np.mean([x["true_reward"] for x in v]))
    for k, cv in curves.items():
        plt.plot(smooth(np.mean(cv, axis=0)), label=k)
    plt.legend(); plt.title("Learning curves (mean of seeds)"); plt.xlabel("episode"); plt.ylabel("true reward"); plt.show()

# ---------------- run everything (comment out what you don't need) ----------------
if __name__ == "__main__":
    plt.figure(figsize=(9, 3)); plt.subplot(1, 2, 1); plt.plot(losses); plt.title("Base RM loss")
    plt.subplot(1, 2, 2); plt.plot(smooth(curve_base)); plt.title("Base RL curve"); plt.tight_layout(); plt.show()
    modification1()
    modification2()
    modification3()
    modification4()
    modification5()
