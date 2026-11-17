"""DQN agent for discriminator subset selection (Sec. IV-C).

The agent scores each discriminator in its local pool and the top-k are made
active for the round.  Three corrections relative to a naive implementation
matter for the agent to learn anything at all, and each is exercised by an
ablation:

  * the reward (Eq. 16) carries a real training-quality term, computed from the
    *previous* round's loss, not from the current metrics dict compared to
    itself (which is identically zero);
  * credit is assigned to every discriminator in the selected subset, not only
    to the first index, since the action is a set;
  * a target network stabilises the bootstrap, and transitions record the
    genuine successor state.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

STATE_DIM = 7  # Eq. (14)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, s):
        return self.net(s)


def normalise(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def compute_reward(metrics, prev_loss, weights):
    """Eq. (16): relative GAN-loss improvement minus normalised resource costs.

    `prev_loss` is last round's Wasserstein loss for this generator; passing the
    current round's metrics as its own predecessor makes the first term vanish.
    """
    w1, w2, w3, w4, w5 = weights
    cur = metrics.get("wasserstein", 0.0)
    if prev_loss is None or abs(prev_loss) < 1e-9:
        improvement = 0.0
    else:
        improvement = (prev_loss - cur) / (abs(prev_loss) + 1e-9)
    improvement = float(np.clip(improvement, -1.0, 1.0))

    lat = normalise(metrics.get("latency", 0.0), 0, metrics.get("latency_max", 60.0))
    traf = normalise(metrics.get("network_traffic", 0.0), 0, metrics.get("traffic_max", 1e6))
    cpu = normalise(metrics.get("cpu_usage", 0.0), 0, metrics.get("cpu_max", 100.0))
    mem = normalise(metrics.get("memory_usage", 0.0), 0, metrics.get("memory_max", 1000.0))

    return w1 * improvement - w2 * lat - w3 * traf - w4 * cpu - w5 * mem


class DQNAgent:
    """Per-generator agent selecting a top-k discriminator subset."""

    def __init__(self, pool_size, k, seed=0, device="cpu",
                 lr=1e-3, gamma=0.99, eps0=1.0, eps_min=0.01, eps_decay=0.995,
                 buffer=10000, batch_size=64, target_sync=10):
        self.pool_size = pool_size
        self.k = min(k, pool_size)
        self.gamma = gamma
        self.eps = eps0
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.device = device
        self._steps = 0

        self.rng = random.Random(seed)
        self.q = QNetwork(STATE_DIM, pool_size).to(device)
        self.target = QNetwork(STATE_DIM, pool_size).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer)

    # -- acting --------------------------------------------------------------
    def act(self, state, greedy=False):
        """Return (subset, source) with subset a list of 0-based pool indices."""
        if not greedy and self.rng.random() < self.eps:
            return self.rng.sample(range(self.pool_size), self.k), "explore"
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qv = self.q(s)[0]
        return torch.topk(qv, self.k).indices.tolist(), "greedy"

    def q_values(self, state):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q(s)[0].cpu().numpy()

    # -- learning ------------------------------------------------------------
    def remember(self, state, subset, reward, next_state, done):
        self.memory.append((np.asarray(state, np.float32), list(subset), float(reward),
                            np.asarray(next_state, np.float32), float(done)))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = self.rng.sample(list(self.memory), self.batch_size)
        s = torch.as_tensor(np.stack([b[0] for b in batch]), device=self.device)
        r = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(np.stack([b[3] for b in batch]), device=self.device)
        d = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Double-DQN target: select with the online net, evaluate with the
            # target net, which curbs the usual Q-value overestimation.
            best = self.q(s2).argmax(1, keepdim=True)
            boot = self.target(s2).gather(1, best).squeeze(1)
            y = r + (1.0 - d) * self.gamma * boot

        qs = self.q(s)
        # The action is a *set*: each selected discriminator shares the credit,
        # so the shared return is regressed onto every member of the subset.
        mask = torch.zeros_like(qs)
        for i, b in enumerate(batch):
            mask[i, b[1]] = 1.0
        n_sel = mask.sum(1).clamp(min=1)
        pred = (qs * mask).sum(1) / n_sel

        loss = nn.functional.mse_loss(pred, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self._steps += 1
        if self._steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        if self.eps > self.eps_min:
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return float(loss.item())


# -- non-learning selectors used as ablation / baseline arms -----------------

class RandomSelector:
    """Random-k subset: the paper's 'Random Parameter Selection' control."""
    name = "random"

    def __init__(self, pool_size, k, seed=0, **kw):
        self.pool_size, self.k, self.rng = pool_size, min(k, pool_size), random.Random(seed)
        self.eps = 0.0

    def act(self, state, greedy=False):
        return self.rng.sample(range(self.pool_size), self.k), "random"

    def remember(self, *a, **kw):
        pass

    def replay(self):
        return None

    def q_values(self, state):
        return np.zeros(self.pool_size, np.float32)


class GreedyLossSelector:
    """Myopic heuristic: pick the k discriminators with the lowest recent loss.

    This is the selection rule the original implementation fell back to, and it
    is the baseline the RL agent has to beat to justify the DQN at all.
    """
    name = "greedy"

    def __init__(self, pool_size, k, seed=0, **kw):
        self.pool_size, self.k = pool_size, min(k, pool_size)
        self.losses = np.full(pool_size, np.inf, np.float64)
        self.eps = 0.0

    def observe_losses(self, losses):
        for i, v in losses.items():
            if v is not None and np.isfinite(v):
                self.losses[i] = v

    def act(self, state, greedy=False):
        if not np.isfinite(self.losses).any():
            return list(range(self.k)), "greedy-init"
        return np.argsort(self.losses)[: self.k].tolist(), "greedy-loss"

    def remember(self, *a, **kw):
        pass

    def replay(self):
        return None

    def q_values(self, state):
        return np.zeros(self.pool_size, np.float32)


class RoundRobinSelector:
    """Deterministic rotation: every discriminator gets equal participation."""
    name = "roundrobin"

    def __init__(self, pool_size, k, seed=0, **kw):
        self.pool_size, self.k, self.cursor = pool_size, min(k, pool_size), 0
        self.eps = 0.0

    def act(self, state, greedy=False):
        sel = [(self.cursor + i) % self.pool_size for i in range(self.k)]
        self.cursor = (self.cursor + self.k) % self.pool_size
        return sel, "roundrobin"

    def remember(self, *a, **kw):
        pass

    def replay(self):
        return None

    def q_values(self, state):
        return np.zeros(self.pool_size, np.float32)


class AllSelector:
    """Use every discriminator each round (no selection at all)."""
    name = "all"

    def __init__(self, pool_size, k=None, seed=0, **kw):
        self.pool_size, self.k = pool_size, pool_size
        self.eps = 0.0

    def act(self, state, greedy=False):
        return list(range(self.pool_size)), "all"

    def remember(self, *a, **kw):
        pass

    def replay(self):
        return None

    def q_values(self, state):
        return np.zeros(self.pool_size, np.float32)


SELECTORS = {
    "dqn": DQNAgent,
    "random": RandomSelector,
    "greedy": GreedyLossSelector,
    "roundrobin": RoundRobinSelector,
    "all": AllSelector,
}


class ResidenceAwareSelector:
    """Mobility-aware discriminator selection.

    Loss-based selection answers "which discriminator is most informative?" but
    in a vehicular network the operative question is "which discriminator will
    still be here when its feedback is needed?".  A vehicle about to exit the
    serving RSU's coverage contributes a stale or lost update regardless of how
    good its local loss is, and the round must then proceed with fewer critics.

    We score each candidate by expected useful contribution over the round,

        score_j = informativeness_j  x  P(complete_j)  x  throughput_j

    where P(complete_j) is the probability that vehicle j remains in coverage
    long enough to return its update.  Writing tau_j for its residence time and
    T_round for the round duration, and modelling the residual dwell time as
    exponential with mean tau_j (the memoryless assumption standard in cell
    residence analysis), this is

        P(complete_j) = exp(-T_round / tau_j).

    The three factors are the three things that actually determine whether a
    selected node helps: it must carry information, survive to deliver it, and
    deliver it over a link fast enough to matter.  Loss-greedy selection uses
    only the first; random uses none.
    """
    name = "residence"

    def __init__(self, pool_size, k, seed=0, alpha=1.0, beta=1.0, **kw):
        self.pool_size, self.k = pool_size, min(k, pool_size)
        self.alpha = alpha          # weight on survival probability
        self.beta = beta            # weight on link throughput
        self.losses = np.full(pool_size, np.inf, np.float64)
        self.tau = np.full(pool_size, np.inf, np.float64)
        self.rate = np.ones(pool_size, np.float64)
        self.eps = 0.0
        self.t_round = 1.0

    # -- observations fed by the trainer each round -------------------------
    def observe_losses(self, losses):
        for i, v in losses.items():
            if v is not None and np.isfinite(v):
                self.losses[i] = v

    def observe_mobility(self, tau, rate, t_round):
        self.tau = np.asarray(tau, float)
        self.rate = np.asarray(rate, float)
        self.t_round = max(float(t_round), 1e-6)

    # -- scoring -------------------------------------------------------------
    def act(self, state, greedy=False):
        finite = np.isfinite(self.losses)
        if not finite.any():
            return list(range(self.k)), "residence-init"

        # Informativeness: a larger critic loss means the generator is still
        # fooling this critic, so its gradient carries more signal.  Normalised
        # to [0,1] across the pool so the three factors are commensurable.
        L = np.where(finite, self.losses, np.nanmedian(self.losses[finite]))
        lo, hi = L.min(), L.max()
        info = (L - lo) / (hi - lo) if hi - lo > 1e-12 else np.ones_like(L)

        # Survival: probability of remaining in coverage for the whole round.
        with np.errstate(divide="ignore", over="ignore"):
            p_complete = np.exp(-self.t_round / np.maximum(self.tau, 1e-6))
        p_complete = np.clip(np.nan_to_num(p_complete, nan=0.0), 0.0, 1.0)

        # Throughput, normalised; a slow link delays the whole synchronisation
        # because the round closes on its slowest member.
        r = self.rate / (self.rate.max() + 1e-12)

        score = (info + 1e-3) * (p_complete ** self.alpha) * (r ** self.beta)
        return np.argsort(-score)[: self.k].tolist(), "residence"

    def remember(self, *a, **kw):
        pass

    def replay(self):
        return None

    def q_values(self, state):
        return np.zeros(self.pool_size, np.float32)


SELECTORS["residence"] = ResidenceAwareSelector
