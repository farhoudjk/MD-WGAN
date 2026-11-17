"""Federated MD-WGAN training loop (Algorithm 1).

One RSU-hosted generator per cell, each paired with a pool of vehicle-hosted
discriminators holding disjoint NGSIM partitions.  Each round runs the four
steps of Sec. IV-A: DQN subset selection, game-theoretic epoch allocation,
distributed WGAN-GP training with ensemble generator feedback, and monitoring.

Every mechanism the paper claims as a contribution can be switched off here
independently, which is what the ablation suite drives.
"""
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from .data import (build_node_dataset, build_node_dataset_cached,
                   build_spatial_partitions)
from .models import Generator, Critic, gradient_penalty
from .game import GameCoordinator, GameParams, DeadlineCoordinator
from .agent import (SELECTORS, DQNAgent, GreedyLossSelector,
                    ResidenceAwareSelector, compute_reward)
from .setsel import SET_SELECTORS, LazyGreedySelector, GeneticSelector
SELECTORS.update(SET_SELECTORS)
from .network import NetworkModel, NetworkParams


@dataclass
class Config:
    """A full experiment arm.  Every ablation is a field flip here."""
    name: str = "full"
    seed: int = 0
    rounds: int = 60
    n_generators: int = 4
    disc_per_gen: int = 5
    top_k: int = 3
    window: int = 8
    batch_size: int = 64
    noise_dim: int = 16
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    lambda_gp: float = 10.0
    max_tracks: int = None          # cap tracks/node to bound runtime
    test_frac: float = 0.2

    # -- ablation switches ---------------------------------------------------
    selector: str = "dqn"           # dqn | random | greedy | roundrobin | all
    use_game: bool = True           # False -> fixed (E_D, E_G)
    adaptive_game: bool = True      # False -> static (paper Table IV) utilities
    use_ensemble: bool = True       # False -> single-discriminator feedback
    ensemble_weighting: str = "uniform"   # uniform | performance
    fixed_E_D: int = 50
    fixed_E_G: int = 5
    conditional: bool = True        # False -> unconditional G(z) (old code)
    lambda_sup: float = 10.0        # supervised (reconstruction) weight; 0 -> pure adversarial
    sup_mode: str = "mean"          # mean | best_of_k -- see generator step
    sup_k: int = 8                  # draws for best_of_k
    lambda_div: float = 0.0         # diversity (mode-seeking) regulariser weight
    use_adversarial: bool = True    # False -> supervised-only (no critic feedback)
    noniid: str = "natural"         # natural | shuffled | segment | lane | iid-split
    source_csv: str = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/data.csv"
    mobility_dropout: bool = True   # vehicles leaving coverage lose their update
    deadline_game: bool = False     # allocate epochs under a residence-derived deadline
    deadline_quantile: float = 0.5  # which selected vehicle's dwell sets the budget
    adaptive_cfg: str = "none"      # none | static | online -- Eq.(8)-aware control
    adaptive_knobs: tuple = ()      # restrict control to these knobs (ablation)
    method: str = "mdwgan"          # mdwgan | hvs-gan | fedavg-gan | md-gan
    gamma_nash: float = 0.7         # interference coefficient (GameParams)

    # Reward weights (w1..w5), Eq. (16).
    reward_weights: tuple = (0.2, 0.2, 0.2, 0.2, 0.2)
    # Objective weights (alpha_G, alpha_1..4), Eq. (8).
    objective_weights: tuple = (0.2, 0.2, 0.2, 0.2, 0.2)

    data_root: str = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets"
    out_root: str = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/mdwgan/runs"
    device: str = "cpu"
    eval_every: int = 1
    eval_samples: int = 16
    eval_k_samples: int = 20   # draws per context for minADE/CRPS/PICP


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _shuffle_partitions(datasets, seed=0):
    """Re-deal every node's windows uniformly, yielding IID local partitions."""
    rng = np.random.default_rng(seed)
    out = [dict(d) for d in datasets]
    for split in ("train", "test"):
        ctx = np.concatenate([d[f"ctx_{split}"] for d in datasets])
        tgt = np.concatenate([d[f"tgt_{split}"] for d in datasets])
        perm = rng.permutation(len(ctx))
        ctx, tgt = ctx[perm], tgt[perm]
        for i, chunk in enumerate(np.array_split(np.arange(len(ctx)), len(datasets))):
            out[i][f"ctx_{split}"] = ctx[chunk]
            out[i][f"tgt_{split}"] = tgt[chunk]
    return out


class GeneratorCell:
    """One RSU: a generator, its discriminator pool, and its local controller."""

    def __init__(self, gid, cfg, node_ids, device):
        self.gid = gid
        self.cfg = cfg
        self.device = device
        self.node_ids = node_ids

        self.data = []
        if cfg.noniid in ("segment", "lane", "iid-split"):
            # Partition the NGSIM source directly, with one global scaler, so
            # cross-node heterogeneity survives normalisation.  The shipped
            # per-node CSVs are random splits scaled per node, which makes every
            # shard N(0,1) and every subset equivalent.
            scheme = "random" if cfg.noniid == "iid-split" else cfg.noniid
            parts = build_spatial_partitions(
                cfg.source_csv, n_nodes=cfg.n_generators * cfg.disc_per_gen,
                window=cfg.window, test_frac=cfg.test_frac, seed=cfg.seed,
                max_tracks=cfg.max_tracks, scheme=scheme)
            self.data = [parts[(nid - 1) % len(parts)] for nid in node_ids]
        else:
            for nid in node_ids:
                path = os.path.join(cfg.data_root, f"discriminator_{nid}", "data.csv")
                self.data.append(build_node_dataset_cached(
                    path, window=cfg.window, test_frac=cfg.test_frac,
                    seed=cfg.seed, max_tracks=cfg.max_tracks))
        if cfg.noniid == "shuffled":
            # Destroy the spatial partitioning by pooling all of this cell's
            # windows and redealing them uniformly, giving every discriminator
            # an IID view.  Contrasting with "natural" measures how much of the
            # framework's benefit comes from coping with heterogeneity.
            self.data = _shuffle_partitions(self.data, seed=cfg.seed + gid)

        nf = self.data[0]["n_features"]
        self.n_features = nf
        self.gen = Generator(cfg.window, nf, noise_dim=cfg.noise_dim).to(device)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))

        self.critics, self.critic_opts = [], []
        for _ in node_ids:
            c = Critic(cfg.window, nf).to(device)
            self.critics.append(c)
            self.critic_opts.append(torch.optim.Adam(c.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9)))

        # Pairwise shard similarity drives the submodular redundancy term.
        # Cosine similarity of per-node target means and stds is a cheap proxy
        # for "these two discriminators see the same distribution".
        feats = []
        for d in self.data:
            t = d["tgt_train"]
            feats.append(np.concatenate([t.mean(0), t.std(0)]) if len(t)
                         else np.zeros(2 * nf))
        F = np.asarray(feats, float)
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        self.shard_sim = np.clip(Fn @ Fn.T, 0.0, 1.0)

        pool = len(node_ids)
        sel_cls = SELECTORS[cfg.selector]
        self.selector = sel_cls(pool_size=pool, k=cfg.top_k, seed=cfg.seed + gid)
        gp = GameParams(adaptive=cfg.adaptive_game, gamma_nash=cfg.gamma_nash)
        self.game = (DeadlineCoordinator(gp) if cfg.deadline_game
                     else GameCoordinator(gp))

        import random as _random
        self._rng = _random.Random(cfg.seed * 977 + gid)

        # Objective-adaptive controller (Eq. 8 aware). `None` reproduces the
        # fixed-configuration behaviour of every baseline.
        self.ctrl = None
        self.sync_every = 1
        if cfg.adaptive_cfg != "none":
            from .adaptive import StaticPolicy, OnlineController
            a = cfg.objective_weights
            base = dict(top_k=cfg.top_k, E_G=cfg.fixed_E_G,
                        sync_every=1, lambda_sup=cfg.lambda_sup)
            if cfg.adaptive_cfg == "static":
                if cfg.adaptive_knobs:
                    from .adaptive import RestrictedPolicy
                    base["sup_mode"] = cfg.sup_mode
                    self.ctrl = RestrictedPolicy(a, base, only=cfg.adaptive_knobs)
                else:
                    self.ctrl = StaticPolicy(a, base)
                self.ctrl.cfg = self.ctrl.configure()
                self.ctrl.cfg.setdefault("sup_mode", cfg.sup_mode)
            else:
                self.ctrl = OnlineController(a, base)

        self.prev_wasserstein = None
        self.prev_state = None
        self.prev_subset = None
        self.disc_loss_ema = {i: None for i in range(pool)}
        self.history = []

    # -- helpers -------------------------------------------------------------
    def _batch(self, i, split="train"):
        d = self.data[i]
        ctx, tgt = d[f"ctx_{split}"], d[f"tgt_{split}"]
        if len(ctx) == 0:
            return None, None
        idx = torch.randint(0, len(ctx), (min(self.cfg.batch_size, len(ctx)),))
        return (torch.as_tensor(ctx[idx], device=self.device),
                torch.as_tensor(tgt[idx], device=self.device))

    def _gen_forward(self, ctx):
        """Honour the `conditional` switch: unconditional mode zeroes the context."""
        if self.cfg.conditional:
            return self.gen(ctx)
        return self.gen(torch.zeros_like(ctx))

    def state_vector(self, net):
        """Eq. (14): the 7-dimensional RL state."""
        losses = [v for v in self.disc_loss_ema.values() if v is not None]
        return np.array([
            float(np.mean(losses)) if losses else 0.0,
            float(self.last_mean_score) if hasattr(self, "last_mean_score") else 0.0,
            net["latency"], net["cpu_usage"], net["memory_usage"],
            net["network_traffic"] / 1e6,
            float(self.prev_mean_score) if hasattr(self, "prev_mean_score") else 0.0,
        ], dtype=np.float32)

    # -- one adaptation cycle ------------------------------------------------
    def run_round(self, rnd, net_model, metrics_norm):
        cfg = self.cfg
        t0 = time.time()

        # --- Step 0: objective-adaptive configuration -----------------------
        # The controller reads the deployment's own alpha vector (Eq. 8) and
        # moves the knobs that drive the terms that weighting cares about.
        # Without it the framework sits at one operating point and can only win
        # the scenario that point happens to suit.
        k_eff, eg_eff, sup_eff = cfg.top_k, None, cfg.sup_mode
        if self.ctrl is not None:
            c = self.ctrl.cfg
            k_eff = min(int(c["top_k"]), len(self.node_ids))
            eg_eff = int(c["E_G"])
            sup_eff = c["sup_mode"]
            self.sync_every = int(c["sync_every"])
            if hasattr(self.selector, "k"):
                self.selector.k = k_eff

        # --- Step 1: adaptive decision engine -------------------------------
        if isinstance(self.selector, (GreedyLossSelector, ResidenceAwareSelector,
                                      LazyGreedySelector, GeneticSelector)):
            self.selector.observe_losses(self.disc_loss_ema)
        if isinstance(self.selector, (LazyGreedySelector, GeneticSelector)):
            pool = len(self.node_ids)
            lat = np.array([net_model.round_trip(self.gid, i,
                            net_model.payload_bytes(self.gen, cfg))
                            for i in range(pool)])
            tau = np.array([net_model.residence_time(self.gid, i) for i in range(pool)])
            T = net_model.p.round_interval
            p_ok = np.clip(np.exp(-T / np.maximum(tau, 1e-6)), 0.0, 1.0)
            self.selector.observe_context(sim=self.shard_sim, lat=lat,
                                          cost=np.ones(pool), p_ok=p_ok)
        if isinstance(self.selector, ResidenceAwareSelector):
            # Mobility state comes from the same channel model that produces
            # the latency and traffic figures, so selection and cost accounting
            # are consistent with one another.
            pool = len(self.node_ids)
            self.selector.observe_mobility(
                [net_model.residence_time(self.gid, i) for i in range(pool)],
                [net_model.link_quality(self.gid, i) for i in range(pool)],
                net_model.p.round_interval)
        state = self.prev_state if self.prev_state is not None else np.zeros(7, np.float32)
        subset, src = self.selector.act(state)
        subset = sorted(set(subset))

        # --- Step 2: game-theoretic epoch allocation ------------------------
        if cfg.use_game:
            if cfg.deadline_game:
                # The round must complete while the selected vehicles are still
                # in coverage, so their residence times set the time budget.
                taus = [net_model.residence_time(self.gid, i) for i in subset]
                taus = [t for t in taus if math.isfinite(t)]
                self.game.set_budget(
                    float(np.quantile(taus, cfg.deadline_quantile)) if taus else None)
            E_D, E_G = self.game.allocate()
        else:
            E_D, E_G = cfg.fixed_E_D, cfg.fixed_E_G
        if eg_eff is not None:
            # The controller's generator budget overrides the schedule: E_G is
            # the knob that trades convergence speed against compute, so it is
            # the one the objective weighting most needs to move.
            E_G = eg_eff

        # --- Step 3: distributed training -----------------------------------
        d_t0 = time.time()
        d_losses, w_dists, mean_scores = {}, [], []
        d_steps = 0          # critic gradient steps actually taken
        for i in subset:
            critic, opt = self.critics[i], self.critic_opts[i]
            last_loss, last_w = 0.0, 0.0
            for _ in range(E_D):
                ctx, real_y = self._batch(i)
                if ctx is None:
                    break
                with torch.no_grad():
                    fake_y = self._gen_forward(ctx)
                real_s, fake_s = critic(ctx, real_y), critic(ctx, fake_y)
                gp = gradient_penalty(critic, ctx, real_y, fake_y)
                # Critic minimises the negative Wasserstein estimate plus GP.
                loss = -(real_s.mean() - fake_s.mean()) + cfg.lambda_gp * gp
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt.step()
                last_loss = float(loss.item())
                last_w = float((real_s.mean() - fake_s.mean()).item())
                d_steps += 1
            d_losses[i] = last_loss
            w_dists.append(last_w)
            with torch.no_grad():
                ctx, _ = self._batch(i)
                if ctx is not None:
                    mean_scores.append(float(critic(ctx, self._gen_forward(ctx)).mean().item()))
        d_secs = (time.time() - d_t0) / max(len(subset) * max(E_D, 1), 1)

        # EMA of per-discriminator loss, used by state vector + greedy baseline.
        for i, v in d_losses.items():
            prev = self.disc_loss_ema[i]
            self.disc_loss_ema[i] = v if prev is None else 0.7 * prev + 0.3 * v

        # Vehicles that leave the serving RSU's coverage before the round
        # closes cannot return their critic feedback, so their update is lost
        # and the generator must proceed with whatever arrived.  This is the
        # cost that mobility-aware selection exists to avoid; with it disabled
        # the environment is effectively static and every selector is equal.
        dropped = []
        if cfg.mobility_dropout:
            T = net_model.p.round_interval
            for i in subset:
                tau = net_model.residence_time(self.gid, i)
                p_ok = math.exp(-T / max(tau, 1e-6)) if math.isfinite(tau) else 1.0
                if self._rng.random() > p_ok:
                    dropped.append(i)
            subset_eff = [i for i in subset if i not in dropped]
            if not subset_eff:          # all departed: round contributes nothing
                subset_eff = []
        else:
            subset_eff = list(subset)

        # Generator update against the ensemble (Eqs. 23-24).
        g_t0 = time.time()
        if cfg.use_ensemble:
            active = subset_eff
        else:
            # Single-discriminator feedback: the original implementation's
            # behaviour, kept as an ablation arm.
            active = subset_eff[:1]

        if not active:
            # Every selected vehicle left coverage before the round closed, so
            # no critic feedback arrived and the generator cannot be updated.
            # The round still costs what it cost; this is the failure mode that
            # mobility-aware selection exists to avoid.
            weights = []
        elif cfg.ensemble_weighting == "performance" and len(active) > 1:
            # Weight each critic by how well it is currently discriminating:
            # a critic with a larger loss is less informative, so it is damped.
            raw = np.array([1.0 / (1.0 + abs(self.disc_loss_ema.get(i) or 1.0)) for i in active])
            weights = (raw / raw.sum()).tolist()
        else:
            weights = [1.0 / len(active)] * len(active)

        g_loss_val = 0.0
        adv_val = sup_val = 0.0
        g_steps = 0          # generator gradient steps actually taken
        for _ in range(E_G):
            self.gen_opt.zero_grad()
            total = 0.0
            for w, i in zip(weights, active):
                ctx, real_y = self._batch(i)
                if ctx is None:
                    continue
                fake_y = self._gen_forward(ctx)
                if cfg.use_adversarial:
                    adv = -self.critics[i](ctx, fake_y).mean()
                    term = w * adv
                    adv_val = float(adv.item())
                else:
                    # Supervised-only arm: isolates how much the adversarial
                    # game contributes beyond plain conditional regression.
                    term = torch.zeros((), device=fake_y.device)

                if cfg.lambda_div > 0 and cfg.conditional:
                    # Mode-seeking regulariser (Mao et al., MSGAN): maximise the
                    # ratio of output distance to latent distance, which
                    # directly penalises a generator for ignoring z.  Without
                    # it the supervised term drives every draw to the
                    # conditional mean and the model stops being generative.
                    z1 = torch.randn(ctx.size(0), self.gen.noise_dim, device=ctx.device)
                    z2 = torch.randn(ctx.size(0), self.gen.noise_dim, device=ctx.device)
                    y1, y2 = self.gen(ctx, z1), self.gen(ctx, z2)
                    d_y = (y1 - y2).abs().mean(1)
                    d_z = (z1 - z2).abs().mean(1)
                    term = term - w * cfg.lambda_div * (d_y / (d_z + 1e-8)).mean()

                if cfg.lambda_sup > 0:
                    # The critic rewards distributional realism, which does not
                    # by itself constrain per-sample error; without this term the
                    # generator is free to drift anywhere inside the data
                    # manifold and RMSE/MAE diverge.  Standard conditional-GAN
                    # practice, and ablated via lambda_sup=0.
                    if sup_eff == "best_of_k":
                        # Best-of-k ("winner-take-all") supervision: only the
                        # closest of k draws is penalised, so the generator may
                        # spread its other draws over alternative futures
                        # instead of collapsing every draw onto the mean.
                        draws = torch.stack([self._gen_forward(ctx)
                                             for _ in range(cfg.sup_k)])
                        err = (draws - real_y.unsqueeze(0)).abs().mean(2)
                        best = err.argmin(0)
                        pick = draws[best, torch.arange(ctx.size(0))]
                        sup = torch.nn.functional.smooth_l1_loss(pick, real_y)
                    else:
                        sup = torch.nn.functional.smooth_l1_loss(fake_y, real_y)
                    term = term + w * cfg.lambda_sup * sup
                    sup_val = float(sup.item())
                total = total + term
            if not torch.is_tensor(total):
                break
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
            self.gen_opt.step()
            g_loss_val = float(total.item())
            g_steps += 1
        g_secs = (time.time() - g_t0) / max(E_G, 1)

        wasserstein = float(np.mean(w_dists)) if w_dists else 0.0

        # --- Step 4: monitoring ---------------------------------------------
        # Algorithmic compute for the round, in units of batch-gradient steps.
        # Critic steps are weighted by the batch they process; generator steps
        # by the number of critics they must query (the ensemble cost).
        work = (d_steps + g_steps * max(len(active), 1)) * cfg.batch_size / 1000.0
        # One synchronisation per generator update: more generator epochs means
        # more parameter exchanges with the selected discriminators.
        net = net_model.round_cost(
            gid=self.gid, subset=subset, rnd=rnd,
            payload_bytes=net_model.payload_bytes(self.gen, self.cfg),
            E_D=E_D, E_G=E_G, compute_seconds=time.time() - t0,
            work_units=work, sync_steps=max(g_steps, 1) / max(self.sync_every, 1))

        # Close the loop: the online controller sees the normalised Eq. (8)
        # terms this round actually realised and re-tunes the knobs that drive
        # whichever term is consuming more than its weighted share.
        if self.ctrl is not None and hasattr(self.ctrl, "update"):
            nz = lambda v, m: float(np.clip(v / max(m, 1e-9), 0.0, 1.0))
            self.ctrl.update([
                nz(abs(wasserstein), 1.0),
                nz(net["cpu_usage"], metrics_norm.get("cpu_max", 100.0)),
                nz(net["memory_usage"], metrics_norm.get("memory_max", 2000.0)),
                nz(net["latency"], metrics_norm.get("latency_max", 60.0)),
                nz(net["network_traffic"], metrics_norm.get("traffic_max", 5e6)),
            ])

        self.prev_mean_score = getattr(self, "last_mean_score", 0.0)
        self.last_mean_score = float(np.mean(mean_scores)) if mean_scores else 0.0

        metrics = dict(net)
        metrics["wasserstein"] = wasserstein
        metrics.update(metrics_norm)
        reward = compute_reward(metrics, self.prev_wasserstein, cfg.reward_weights)

        next_state = self.state_vector(net)
        if self.prev_state is not None and self.prev_subset is not None:
            self.selector.remember(self.prev_state, self.prev_subset, reward, next_state, 0.0)
        dqn_loss = self.selector.replay()

        # Feed realised effectiveness/cost back into the coordinator.
        if cfg.use_game and cfg.adaptive_game:
            d_imp = g_imp = None
            if self.prev_wasserstein is not None and abs(self.prev_wasserstein) > 1e-9:
                rel = (self.prev_wasserstein - wasserstein) / (abs(self.prev_wasserstein) + 1e-9)
                # The critic gains when the Wasserstein estimate grows (better
                # separation); the generator gains when it shrinks.
                d_imp, g_imp = max(-rel, 0.0), max(rel, 0.0)
            self.game.observe({
                "d_loss_improvement": d_imp, "g_loss_improvement": g_imp,
                "E_D": E_D, "E_G": E_G,
                "d_epoch_seconds": d_secs, "g_epoch_seconds": g_secs,
            })

        self.prev_state, self.prev_subset = next_state, subset
        self.prev_wasserstein = wasserstein

        rec = {
            "round": rnd, "generator": self.gid, "subset": subset, "select_src": src,
            "E_D": E_D, "E_G": E_G, "wasserstein": wasserstein,
            "g_loss": g_loss_val, "g_adv": adv_val, "g_sup": sup_val,
            "d_steps": d_steps, "g_steps": g_steps, "reward": reward, "epsilon": float(self.selector.eps),
            "dqn_loss": dqn_loss, "mean_score": self.last_mean_score,
            "n_active": len(active), "n_dropped": len(dropped), **net,
        }
        self.history.append(rec)
        return rec

    # -- evaluation ----------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        """Held-out next-step accuracy plus distributional and probabilistic scores.

        RMSE/MAE alone are misleading for a conditional generative model: both
        are minimised by predicting the conditional mean, which is precisely
        mode collapse.  A generator that ignores z entirely scores well on them.
        We therefore also report:

          * minADE_k  -- best-of-k error, the standard multi-modal trajectory
            metric (Social-GAN and successors): a model is credited if *any* of
            k sampled futures is close, which rewards covering the plausible
            set rather than averaging it away.
          * CRPS      -- continuous ranked probability score, a strictly proper
            scoring rule for the whole predictive distribution.  It is minimised
            only by the true conditional law, so unlike RMSE it cannot be gamed
            by collapsing to the mean.
          * PICP      -- prediction-interval coverage probability: the fraction
            of targets inside the model's own central 90% interval.  Measures
            calibration; 0.90 is ideal, below means overconfident.
          * mode coverage / JS -- marginal distributional fidelity.
        """
        self.gen.eval()
        K = self.cfg.eval_k_samples
        errs, abss, reals, fakes = [], [], [], []
        min_errs, crps_acc, picp_acc, n_acc = [], [], [], 0
        for d in self.data:
            ctx, tgt = d["ctx_test"], d["tgt_test"]
            if len(ctx) == 0:
                continue
            c = torch.as_tensor(ctx, device=self.device)
            if not self.cfg.conditional:
                c = torch.zeros_like(c)
            # Draw K independent futures per context; everything below is a
            # different reduction of the same sample set.
            draws = torch.stack([self.gen(c) for _ in range(K)]).cpu().numpy()
            pred = draws.mean(0)                     # conditional-mean estimate

            errs.append((pred - tgt) ** 2)
            abss.append(np.abs(pred - tgt))
            reals.append(tgt)
            fakes.append(draws[0])                   # a single draw, for fidelity

            # minADE_k: error of the closest of the K sampled futures.
            per_draw = np.linalg.norm(draws - tgt[None], axis=2)   # (K, N)
            min_errs.append(per_draw.min(0))

            crps_acc.append(crps_ensemble(draws, tgt) * len(tgt))
            lo = np.percentile(draws, 5, axis=0)
            hi = np.percentile(draws, 95, axis=0)
            picp_acc.append(float(((tgt >= lo) & (tgt <= hi)).mean()) * len(tgt))
            n_acc += len(tgt)
        self.gen.train()
        if not errs:
            return {"rmse": float("nan"), "mae": float("nan")}
        e = np.concatenate(errs)
        a = np.concatenate(abss)
        real = np.concatenate(reals)
        fake = np.concatenate(fakes)
        return {
            "rmse": float(np.sqrt(e.mean())),
            "mae": float(a.mean()),
            "min_ade": float(np.concatenate(min_errs).mean()),
            "crps": float(sum(crps_acc) / max(n_acc, 1)),
            "picp": float(sum(picp_acc) / max(n_acc, 1)),
            "per_feature_rmse": np.sqrt(e.mean(0)).tolist(),
            "mode_coverage": mode_coverage(real, fake),
            "js_proxy": js_proxy(real, fake),
        }


def crps_ensemble(draws, target):
    """CRPS of an ensemble forecast, averaged over samples and features.

    CRPS = E|X - y| - 0.5 E|X - X'|, the standard energy-form estimator for a
    finite ensemble.  It is a *strictly proper* scoring rule: it is minimised
    only when the predictive distribution equals the true conditional law, so a
    generator cannot improve it by collapsing onto the conditional mean the way
    it can with RMSE.
    """
    K = draws.shape[0]
    term1 = np.abs(draws - target[None]).mean(0)
    # Mean absolute pairwise spread within the ensemble.
    diffs = np.abs(draws[:, None] - draws[None, :]).mean((0, 1))
    return float((term1 - 0.5 * diffs).mean())


def mode_coverage(real, fake, bins=20):
    """Fraction of occupied real-data histogram cells the generator reaches.

    A standard, cheap mode-collapse diagnostic: a collapsed generator covers
    few of the bins the real data occupies, even when its RMSE looks fine.
    """
    cov = []
    for j in range(real.shape[1]):
        lo, hi = real[:, j].min(), real[:, j].max()
        if hi - lo < 1e-9:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        r = np.histogram(real[:, j], edges)[0] > 0
        f = np.histogram(fake[:, j], edges)[0] > 0
        cov.append((r & f).sum() / max(r.sum(), 1))
    return float(np.mean(cov)) if cov else float("nan")


def js_proxy(real, fake, bins=20):
    """Per-feature Jensen-Shannon divergence between marginal histograms."""
    out = []
    for j in range(real.shape[1]):
        lo = min(real[:, j].min(), fake[:, j].min())
        hi = max(real[:, j].max(), fake[:, j].max())
        if hi - lo < 1e-9:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p = np.histogram(real[:, j], edges)[0].astype(float) + 1e-9
        q = np.histogram(fake[:, j], edges)[0].astype(float) + 1e-9
        p, q = p / p.sum(), q / q.sum()
        m = 0.5 * (p + q)
        kl = lambda x, y: float((x * np.log(x / y)).sum())
        out.append(0.5 * kl(p, m) + 0.5 * kl(q, m))
    return float(np.mean(out)) if out else float("nan")


def run_experiment(cfg, progress=False):
    """Execute one arm end to end and return its full record."""
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, int(os.environ.get("MDWGAN_THREADS", "2"))))
    device = torch.device(cfg.device)

    net_model = NetworkModel(NetworkParams(), seed=cfg.seed)
    cells = []
    if cfg.method != "mdwgan":
        # Comparison methods share the data, backbone and evaluation protocol;
        # only the coordination strategy differs.
        from .baselines import BASELINES
        cls = BASELINES[cfg.method]
        for g in range(cfg.n_generators):
            nodes = [g * cfg.disc_per_gen + i + 1 for i in range(cfg.disc_per_gen)]
            cells.append(cls(cfg, nodes, device))
    else:
        for g in range(cfg.n_generators):
            nodes = [g * cfg.disc_per_gen + i + 1 for i in range(cfg.disc_per_gen)]
            cells.append(GeneratorCell(g, cfg, nodes, device))

    # Normalisation constants for the reward's resource terms (Eq. 16).
    # Normalisation constants for Eqs. (8) and (16).  These bound each term so
    # the weighted sum is a convex combination over commensurable quantities.
    # cpu_max is the work (in batch-gradient-step units) of the most expensive
    # schedule the configuration can select: every discriminator active, at the
    # largest epoch allocation in each player's strategy set.
    max_E_D = max(GameParams().E_D_range) if cfg.use_game else cfg.fixed_E_D
    max_E_G = max(GameParams().E_G_range) if cfg.use_game else cfg.fixed_E_G
    pool = cfg.disc_per_gen
    cpu_max = (pool * max_E_D + max_E_G * pool) * cfg.batch_size / 1000.0
    # Likewise, the traffic of synchronising with the whole pool every epoch.
    n_params = sum(q.numel() for q in cells[0].gen.parameters())
    traffic_max = (2.0 * pool * n_params * 4 * 1.08) * max_E_G
    metrics_norm = {"latency_max": 60.0, "traffic_max": traffic_max,
                    "cpu_max": cpu_max, "memory_max": 2000.0}

    rounds, evals = [], []
    t_start = time.time()
    # Per-round heartbeat, so a long sweep's progress is observable from disk
    # without attaching to the process.
    hb_path = os.environ.get("MDWGAN_HEARTBEAT")
    for rnd in range(1, cfg.rounds + 1):
        for cell in cells:
            rounds.append(cell.run_round(rnd, net_model, metrics_norm))
        if hb_path:
            try:
                with open(hb_path, "w") as f:
                    f.write(f"{cfg.name} seed{cfg.seed} round {rnd}/{cfg.rounds} "
                            f"{time.time() - t_start:.0f}s\n")
            except OSError:
                pass
        if rnd % cfg.eval_every == 0 or rnd == cfg.rounds:
            per_gen = [c.evaluate() for c in cells]
            ev = {"round": rnd}
            for key in ("rmse", "mae", "min_ade", "crps", "picp",
                        "mode_coverage", "js_proxy"):
                ev[key] = float(np.nanmean([p.get(key, np.nan) for p in per_gen]))
            evals.append(ev)
            if progress:
                print(f"[{cfg.name} s{cfg.seed}] round {rnd:3d} "
                      f"rmse={ev['rmse']:.4f} mae={ev['mae']:.4f}", flush=True)

    final = evals[-1] if evals else {}
    cost = objective_cost(rounds, cfg, metrics_norm, final)
    result = {
        "config": asdict(cfg),
        "rounds": rounds,
        "evals": evals,
        "final": final,
        "objective_cost": cost,
        "equilibrium_rate": float(np.mean([c.game.equilibrium_gap() for c in cells]))
        if all(hasattr(c, "game") for c in cells) else float("nan"),
        "wall_seconds": time.time() - t_start,
    }
    return result


def objective_cost(rounds, cfg, norm, final):
    """Eq. (8) evaluated over the last 20% of rounds (post-convergence)."""
    if not rounds:
        return float("nan")
    aG, a1, a2, a3, a4 = cfg.objective_weights
    tail = rounds[max(0, int(len(rounds) * 0.8)):]
    mean = lambda k: float(np.mean([r[k] for r in tail]))
    # Training term uses a sigmoid-squashed Wasserstein estimate so the cost is
    # commensurable with the four normalised resource terms.
    w = mean("wasserstein")
    train_term = 1.0 / (1.0 + math.exp(-abs(w)))
    return (aG * train_term
            + a1 * mean("cpu_usage") / norm["cpu_max"]
            + a2 * mean("memory_usage") / norm["memory_max"]
            + a3 * mean("latency") / norm["latency_max"]
            + a4 * mean("network_traffic") / norm["traffic_max"])
