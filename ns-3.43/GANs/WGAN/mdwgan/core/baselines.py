"""Reference implementations of the comparison methods.

Each baseline is trained on the *same* NGSIM partitions, with the *same*
conditional generator/critic backbone and the *same* evaluation protocol as the
proposed framework, so differences reflect the coordination strategy rather
than architecture or data handling.  Where a paper's description leaves a
detail unspecified, the choice is noted inline and kept as close to the
proposed method as possible so the comparison does not advantage either side.
"""
import time

import numpy as np
import torch

from .models import Generator, Critic, gradient_penalty
from .data import build_node_dataset_cached
from .network import NetworkModel, NetworkParams


# ---------------------------------------------------------------------------
# HVS-GAN  (Wu, Lin & Wang, "Federated training generative adversarial networks
# for heterogeneous vehicle scheduling in IoV", IEEE IoT-J 2024; ref [19])
#
# Method as described: a *federated* GAN in which each selected intelligent
# connected vehicle (ICV) trains a full local GAN (generator and discriminator)
# on its own data; a central server aggregates the local models by FedAvg each
# round.  Vehicle heterogeneity -- differing model sizes, compute and training
# cost -- is handled by a DDPG agent at the server that *schedules* which
# vehicles participate, trading training contribution against cost.
#
# Key architectural contrasts with the proposed framework, which are the point
# of the comparison:
#   * one discriminator per vehicle, each paired with its own local generator;
#     there is no multi-discriminator ensemble feeding a shared generator
#   * coordination is centralised parameter aggregation, not per-RSU control
#   * scheduling is a continuous-action DDPG policy, not discrete subset choice
# ---------------------------------------------------------------------------

class _Actor(torch.nn.Module):
    def __init__(self, s, a, h=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(s, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, a), torch.nn.Sigmoid())   # per-vehicle score in [0,1]

    def forward(self, x):
        return self.net(x)


class _CriticQ(torch.nn.Module):
    def __init__(self, s, a, h=128):
        super().__init__()
        self.fs = torch.nn.Linear(s, h)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(h + a, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))

    def forward(self, s, a):
        return self.net(torch.cat([torch.relu(self.fs(s)), a], dim=1))


class _OUNoise:
    """Ornstein-Uhlenbeck exploration noise, as used by DDPG."""

    def __init__(self, size, rng, theta=0.15, sigma=0.2):
        self.size, self.rng, self.theta, self.sigma = size, rng, theta, sigma
        self.state = np.zeros(size)

    def sample(self):
        self.state = self.state + self.theta * (-self.state) \
            + self.sigma * self.rng.standard_normal(self.size)
        return self.state


class HVSGAN:
    """Federated GAN with DDPG-based heterogeneous vehicle scheduling."""

    name = "hvs-gan"

    def __init__(self, cfg, node_ids, device):
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(cfg.seed)
        self.data = [build_node_dataset_cached(
            f"{cfg.data_root}/discriminator_{n}/data.csv", window=cfg.window,
            test_frac=cfg.test_frac, seed=cfg.seed, max_tracks=cfg.max_tracks)
            for n in node_ids]
        nf = self.data[0]["n_features"]
        self.n_vehicles = len(node_ids)

        # Per-vehicle local GAN (full generator + discriminator on each node).
        self.gens = [Generator(cfg.window, nf, noise_dim=cfg.noise_dim).to(device)
                     for _ in node_ids]
        self.discs = [Critic(cfg.window, nf).to(device) for _ in node_ids]
        self.gopts = [torch.optim.Adam(g.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
                      for g in self.gens]
        self.dopts = [torch.optim.Adam(d.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))
                      for d in self.discs]

        # The global model the server maintains and broadcasts.
        self.global_gen = Generator(cfg.window, nf, noise_dim=cfg.noise_dim).to(device)

        # Heterogeneity: vehicles differ in compute capability and hence in the
        # cost of participating.  Drawn once per run and held fixed.
        self.capability = self.rng.uniform(0.4, 1.0, self.n_vehicles)
        self.cost = self.rng.uniform(0.5, 1.5, self.n_vehicles)

        # DDPG scheduler: state = per-vehicle (loss, capability, cost, staleness)
        self.s_dim = 4 * self.n_vehicles
        self.actor = _Actor(self.s_dim, self.n_vehicles).to(device)
        self.critic = _CriticQ(self.s_dim, self.n_vehicles).to(device)
        self.actor_t = _Actor(self.s_dim, self.n_vehicles).to(device)
        self.critic_t = _CriticQ(self.s_dim, self.n_vehicles).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.noise = _OUNoise(self.n_vehicles, self.rng)
        self.buffer = []

        self.losses = np.zeros(self.n_vehicles)
        self.staleness = np.zeros(self.n_vehicles)
        # Uniform interface: `gen` is the model the harness measures and
        # evaluates, whichever coordination scheme produced it.
        self.gen = self.global_gen
        self.prev = None
        self.history = []

    # -- helpers ------------------------------------------------------------
    def _batch(self, i):
        d = self.data[i]
        ctx, tgt = d["ctx_train"], d["tgt_train"]
        if len(ctx) == 0:
            return None, None
        idx = torch.randint(0, len(ctx), (min(self.cfg.batch_size, len(ctx)),))
        return (torch.as_tensor(ctx[idx], device=self.device),
                torch.as_tensor(tgt[idx], device=self.device))

    def _state(self):
        return np.concatenate([self.losses, self.capability,
                               self.cost, self.staleness]).astype(np.float32)

    def _schedule(self, state):
        """DDPG actor scores every vehicle; the top-scoring subset is selected."""
        with torch.no_grad():
            s = torch.as_tensor(state, device=self.device).unsqueeze(0)
            a = self.actor(s)[0].cpu().numpy()
        a = np.clip(a + self.noise.sample(), 0.0, 1.0)
        k = max(1, int(self.cfg.top_k))
        return np.argsort(-a)[:k].tolist(), a

    # -- one federated round -------------------------------------------------
    def run_round(self, rnd, net_model, metrics_norm):
        cfg = self.cfg
        t0 = time.time()
        state = self._state()
        selected, action = self._schedule(state)

        # Server broadcasts the global generator to the scheduled vehicles.
        for i in selected:
            self.gens[i].load_state_dict(self.global_gen.state_dict())

        d_steps = g_steps = 0
        for i in selected:
            # Local epochs scale with the vehicle's capability: a weaker ICV
            # completes fewer, which is the heterogeneity HVS-GAN schedules around.
            E = max(1, int(round(cfg.fixed_E_D * self.capability[i])))
            for _ in range(E):
                ctx, real_y = self._batch(i)
                if ctx is None:
                    break
                with torch.no_grad():
                    fake_y = self.gens[i](ctx)
                gp = gradient_penalty(self.discs[i], ctx, real_y, fake_y)
                loss = -(self.discs[i](ctx, real_y).mean()
                         - self.discs[i](ctx, fake_y).mean()) + cfg.lambda_gp * gp
                self.dopts[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discs[i].parameters(), 1.0)
                self.dopts[i].step()
                d_steps += 1
            self.losses[i] = float(loss.item()) if d_steps else 0.0

            EG = max(1, int(round(cfg.fixed_E_G * self.capability[i])))
            for _ in range(EG):
                ctx, real_y = self._batch(i)
                if ctx is None:
                    break
                fake_y = self.gens[i](ctx)
                adv = -self.discs[i](ctx, fake_y).mean()
                total = adv
                if cfg.lambda_sup > 0:
                    total = total + cfg.lambda_sup * torch.nn.functional.smooth_l1_loss(
                        fake_y, real_y)
                self.gopts[i].zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.gens[i].parameters(), 1.0)
                self.gopts[i].step()
                g_steps += 1

        # FedAvg aggregation at the server, weighted by local sample count.
        if selected:
            w = np.array([len(self.data[i]["ctx_train"]) for i in selected], float)
            w = w / w.sum()
            gsd = {k: sum(float(wi) * self.gens[i].state_dict()[k].float()
                          for wi, i in zip(w, selected))
                   for k in self.global_gen.state_dict()}
            self.global_gen.load_state_dict(gsd)

        self.staleness += 1
        for i in selected:
            self.staleness[i] = 0

        # Reward: training progress against the scheduling cost of the chosen
        # vehicles -- the trade-off HVS-GAN's DDPG agent is defined to optimise.
        prog = -float(np.mean([self.losses[i] for i in selected])) if selected else 0.0
        cost = float(np.mean([self.cost[i] for i in selected])) if selected else 0.0
        reward = prog - 0.5 * cost
        nxt = self._state()
        self.buffer.append((state, action.astype(np.float32), reward, nxt))
        if len(self.buffer) > 2000:
            self.buffer.pop(0)
        self._ddpg_update()

        work = (d_steps + g_steps) * cfg.batch_size / 1000.0
        net = net_model.round_cost(
            gid=0, subset=selected, rnd=rnd,
            payload_bytes=net_model.payload_bytes(self.global_gen, cfg),
            E_D=cfg.fixed_E_D, E_G=cfg.fixed_E_G,
            compute_seconds=time.time() - t0,
            work_units=work, sync_steps=max(g_steps, 1))
        rec = {"round": rnd, "generator": 0, "subset": selected,
               "E_D": cfg.fixed_E_D, "E_G": cfg.fixed_E_G,
               "wasserstein": float(np.mean([self.losses[i] for i in selected])) if selected else 0.0,
               "reward": reward, "d_steps": d_steps, "g_steps": g_steps,
               "n_active": len(selected), **net}
        self.history.append(rec)
        return rec

    def _ddpg_update(self, batch=32, gamma=0.99, tau=0.01):
        if len(self.buffer) < batch:
            return
        idx = self.rng.choice(len(self.buffer), batch, replace=False)
        s = torch.as_tensor(np.stack([self.buffer[i][0] for i in idx]), device=self.device)
        a = torch.as_tensor(np.stack([self.buffer[i][1] for i in idx]), device=self.device)
        r = torch.as_tensor(np.array([self.buffer[i][2] for i in idx], np.float32),
                            device=self.device).unsqueeze(1)
        s2 = torch.as_tensor(np.stack([self.buffer[i][3] for i in idx]), device=self.device)
        with torch.no_grad():
            y = r + gamma * self.critic_t(s2, self.actor_t(s2))
        closs = torch.nn.functional.mse_loss(self.critic(s, a), y)
        self.copt.zero_grad(); closs.backward(); self.copt.step()
        aloss = -self.critic(s, self.actor(s)).mean()
        self.aopt.zero_grad(); aloss.backward(); self.aopt.step()
        for t, src in ((self.actor_t, self.actor), (self.critic_t, self.critic)):
            for pt, ps in zip(t.parameters(), src.parameters()):
                pt.data.copy_(tau * ps.data + (1 - tau) * pt.data)

    # -- evaluation (identical protocol to the proposed framework) -----------
    @torch.no_grad()
    def evaluate(self):
        from .trainer import mode_coverage, js_proxy, crps_ensemble
        self.gen.eval()
        K = self.cfg.eval_k_samples
        errs, abss, reals, fakes, mins, crps, picp, n = [], [], [], [], [], [], [], 0
        for d in self.data:
            ctx, tgt = d["ctx_test"], d["tgt_test"]
            if len(ctx) == 0:
                continue
            c = torch.as_tensor(ctx, device=self.device)
            draws = torch.stack([self.gen(c) for _ in range(K)]).cpu().numpy()
            pred = draws.mean(0)
            errs.append((pred - tgt) ** 2); abss.append(np.abs(pred - tgt))
            reals.append(tgt); fakes.append(draws[0])
            mins.append(np.linalg.norm(draws - tgt[None], axis=2).min(0))
            crps.append(crps_ensemble(draws, tgt) * len(tgt))
            lo, hi = np.percentile(draws, 5, axis=0), np.percentile(draws, 95, axis=0)
            picp.append(float(((tgt >= lo) & (tgt <= hi)).mean()) * len(tgt))
            n += len(tgt)
        self.gen.train()
        if not errs:
            return {"rmse": float("nan")}
        e = np.concatenate(errs); real = np.concatenate(reals); fake = np.concatenate(fakes)
        return {"rmse": float(np.sqrt(e.mean())),
                "mae": float(np.concatenate(abss).mean()),
                "min_ade": float(np.concatenate(mins).mean()),
                "crps": float(sum(crps) / max(n, 1)),
                "picp": float(sum(picp) / max(n, 1)),
                "mode_coverage": mode_coverage(real, fake),
                "js_proxy": js_proxy(real, fake)}


# ---------------------------------------------------------------------------
# FedAvg-GAN: the standard federated-learning control.  Identical to HVS-GAN
# except that participants are chosen uniformly at random each round instead of
# by a learned scheduler.  Isolates what the DDPG scheduling actually buys.
# ---------------------------------------------------------------------------

class FedAvgGAN(HVSGAN):
    name = "fedavg-gan"

    def _schedule(self, state):
        k = max(1, int(self.cfg.top_k))
        sel = self.rng.choice(self.n_vehicles, size=k, replace=False).tolist()
        a = np.zeros(self.n_vehicles, np.float32)
        a[sel] = 1.0
        return sel, a

    def _ddpg_update(self, *a, **k):
        return      # no scheduler to train


# ---------------------------------------------------------------------------
# MD-GAN (Hardy, Le Merrer & Sericola, IPDPS 2019): one central generator, one
# discriminator per worker, every worker active each round.  Workers swap
# discriminators periodically so no discriminator overfits its local shard.
# This is the canonical multi-discriminator reference the proposed framework
# extends, so it isolates the contribution of *selective* participation.
# ---------------------------------------------------------------------------

class MDGAN:
    name = "md-gan"

    def __init__(self, cfg, node_ids, device, swap_every=5):
        self.cfg, self.device = cfg, device
        self.rng = np.random.default_rng(cfg.seed)
        self.swap_every = swap_every
        self.data = [build_node_dataset_cached(
            f"{cfg.data_root}/discriminator_{n}/data.csv", window=cfg.window,
            test_frac=cfg.test_frac, seed=cfg.seed, max_tracks=cfg.max_tracks)
            for n in node_ids]
        nf = self.data[0]["n_features"]
        self.K = len(node_ids)
        self.gen = Generator(cfg.window, nf, noise_dim=cfg.noise_dim).to(device)
        self.gopt = torch.optim.Adam(self.gen.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
        self.discs = [Critic(cfg.window, nf).to(device) for _ in node_ids]
        self.dopts = [torch.optim.Adam(d.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))
                      for d in self.discs]
        # Which data shard each discriminator currently sits on.
        self.assign = list(range(self.K))
        self.history = []

    def _batch(self, shard):
        d = self.data[shard]
        ctx, tgt = d["ctx_train"], d["tgt_train"]
        if len(ctx) == 0:
            return None, None
        idx = torch.randint(0, len(ctx), (min(self.cfg.batch_size, len(ctx)),))
        return (torch.as_tensor(ctx[idx], device=self.device),
                torch.as_tensor(tgt[idx], device=self.device))

    def run_round(self, rnd, net_model, metrics_norm):
        cfg = self.cfg
        t0 = time.time()
        d_steps = g_steps = 0
        losses = []
        # All K discriminators participate every round -- the defining contrast
        # with the proposed framework's selected subset.
        for j in range(self.K):
            shard = self.assign[j]
            for _ in range(cfg.fixed_E_D):
                ctx, real_y = self._batch(shard)
                if ctx is None:
                    break
                with torch.no_grad():
                    fake_y = self.gen(ctx)
                gp = gradient_penalty(self.discs[j], ctx, real_y, fake_y)
                loss = -(self.discs[j](ctx, real_y).mean()
                         - self.discs[j](ctx, fake_y).mean()) + cfg.lambda_gp * gp
                self.dopts[j].zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discs[j].parameters(), 1.0)
                self.dopts[j].step(); d_steps += 1
            losses.append(float(loss.item()) if d_steps else 0.0)

        for _ in range(cfg.fixed_E_G):
            self.gopt.zero_grad()
            total = 0.0
            for j in range(self.K):
                ctx, real_y = self._batch(self.assign[j])
                if ctx is None:
                    continue
                fake_y = self.gen(ctx)
                t = -self.discs[j](ctx, fake_y).mean() / self.K
                if cfg.lambda_sup > 0:
                    t = t + cfg.lambda_sup * torch.nn.functional.smooth_l1_loss(
                        fake_y, real_y) / self.K
                total = total + t
            if not torch.is_tensor(total):
                break
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
            self.gopt.step(); g_steps += 1

        # Periodic discriminator swap between workers (Hardy et al.).
        if rnd % self.swap_every == 0 and rnd > 0:
            self.rng.shuffle(self.assign)

        work = (d_steps + g_steps * self.K) * cfg.batch_size / 1000.0
        net = net_model.round_cost(
            gid=0, subset=list(range(self.K)), rnd=rnd,
            payload_bytes=net_model.payload_bytes(self.gen, cfg),
            E_D=cfg.fixed_E_D, E_G=cfg.fixed_E_G,
            compute_seconds=time.time() - t0,
            work_units=work, sync_steps=max(g_steps, 1))
        rec = {"round": rnd, "generator": 0, "subset": list(range(self.K)),
               "E_D": cfg.fixed_E_D, "E_G": cfg.fixed_E_G,
               "wasserstein": float(np.mean(losses)) if losses else 0.0,
               "reward": 0.0, "d_steps": d_steps, "g_steps": g_steps,
               "n_active": self.K, **net}
        self.history.append(rec)
        return rec

    evaluate = HVSGAN.evaluate


# ---------------------------------------------------------------------------
# Early-update MD-GAN (Jafari Kaleibar & Litoiu, MASCOTS 2025): the authors'
# earlier multi-discriminator design, ported from scratch/dsrc-simulations4.cc
# into the common protocol.  Every discriminator trains every round at fixed
# epochs, but a node *transmits* only when its loss improves on its own best so
# far: a critic sends feedback only if its loss improved, and the generator
# trains on whatever feedback arrived (asynchronous, minimum one report) and
# broadcasts new samples only if its own loss improved.  No discriminator swap
# and no selection, learning or coordination -- fixed training parameters.
# ---------------------------------------------------------------------------

class EarlyUpdateMDGAN(MDGAN):
    name = "early-update-mdgan"

    def __init__(self, cfg, node_ids, device, min_feedback=1):
        super().__init__(cfg, node_ids, device)
        self.min_feedback = min_feedback
        self.best_d = [float("inf")] * self.K
        self.best_g = float("inf")

    def run_round(self, rnd, net_model, metrics_norm):
        cfg = self.cfg
        t0 = time.time()
        d_steps = g_steps = 0
        losses, improved = [], []
        for j in range(self.K):
            loss = None
            for _ in range(cfg.fixed_E_D):
                ctx, real_y = self._batch(j)
                if ctx is None:
                    break
                with torch.no_grad():
                    fake_y = self.gen(ctx)
                gp = gradient_penalty(self.discs[j], ctx, real_y, fake_y)
                loss = -(self.discs[j](ctx, real_y).mean()
                         - self.discs[j](ctx, fake_y).mean()) + cfg.lambda_gp * gp
                self.dopts[j].zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discs[j].parameters(), 1.0)
                self.dopts[j].step(); d_steps += 1
            if loss is None:
                continue
            lv = float(loss.item())
            losses.append(lv)
            # Early update: transmit only if this critic beat its own best.
            if lv < self.best_d[j]:
                self.best_d[j] = lv
                improved.append(j)

        g_loss = None
        if len(improved) >= self.min_feedback:
            for _ in range(cfg.fixed_E_G):
                self.gopt.zero_grad()
                total = 0.0
                for j in improved:
                    ctx, real_y = self._batch(j)
                    if ctx is None:
                        continue
                    fake_y = self.gen(ctx)
                    t = -self.discs[j](ctx, fake_y).mean() / len(improved)
                    if cfg.lambda_sup > 0:
                        t = t + cfg.lambda_sup * torch.nn.functional.smooth_l1_loss(
                            fake_y, real_y) / len(improved)
                    total = total + t
                if not torch.is_tensor(total):
                    break
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
                self.gopt.step(); g_steps += 1
                g_loss = float(total.item())
        g_improved = g_loss is not None and g_loss < self.best_g
        if g_improved:
            self.best_g = g_loss

        # Traffic: uplink feedback from improved critics only; downlink sample
        # broadcast to every critic only when the generator improved.
        payload = net_model.payload_bytes(self.gen, cfg)
        uplinks = improved
        work = (d_steps + g_steps * max(len(improved), 1)) * cfg.batch_size / 1000.0
        net = net_model.round_cost(
            gid=0, subset=uplinks, rnd=rnd, payload_bytes=payload,
            E_D=cfg.fixed_E_D, E_G=cfg.fixed_E_G,
            compute_seconds=time.time() - t0,
            work_units=work, sync_steps=max(g_steps, 1))
        sync = max(g_steps, 1)
        # round_cost counted one down + one up per uplink critic; correct it to
        # (uplinks) up + (K if generator improved else 0) down.
        down = (self.K if g_improved else 0)
        net["network_traffic"] = float(payload * net_model.p.protocol_overhead
                                       * sync * (len(uplinks) + down))
        rec = {"round": rnd, "generator": 0, "subset": list(range(self.K)),
               "E_D": cfg.fixed_E_D, "E_G": cfg.fixed_E_G,
               "wasserstein": float(np.mean(losses)) if losses else 0.0,
               "reward": 0.0, "d_steps": d_steps, "g_steps": g_steps,
               "n_active": len(improved), "n_uplink": len(uplinks),
               "g_broadcast": int(g_improved), **net}
        self.history.append(rec)
        return rec


BASELINES = {"hvs-gan": HVSGAN, "fedavg-gan": FedAvgGAN, "md-gan": MDGAN,
             "early-update-mdgan": EarlyUpdateMDGAN}
