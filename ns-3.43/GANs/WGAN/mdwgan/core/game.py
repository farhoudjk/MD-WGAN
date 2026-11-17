"""Game-theoretic epoch allocation between generator and discriminators.

Sec. IV-D of the paper.  The two players bid for training epochs; utilities
trade effectiveness (diminishing returns in epochs) against resource cost and
against an interference term driven by the opponent's effectiveness.

The key departure from a fixed schedule is that the effectiveness rates and
per-epoch costs are *measured online* rather than fixed constants: kappa is
re-estimated from the loss improvement each player actually realised per epoch
last round, and c_epoch from the wall-clock seconds each epoch actually took.
A coordinator whose inputs never change would return the same allocation every
round and would be indistinguishable from a constant, so this state-dependence
is what the `fixed-epochs` and `static-game` ablations are designed to isolate.
"""
import math
from dataclasses import dataclass, field


@dataclass
class GameParams:
    kappa_D: float = 0.05          # discriminator effectiveness rate
    kappa_G: float = 0.10          # generator effectiveness rate
    c_epoch_D: float = 0.3         # per-epoch cost (seconds)
    c_epoch_G: float = 0.8
    lambda_D: float = 0.05         # cost weight relative to effectiveness
    lambda_G: float = 0.05
    gamma_nash: float = 0.7        # interference coefficient
    E_D_range: tuple = tuple(range(45, 76, 5))
    E_G_range: tuple = tuple(range(2, 16))
    # Exponential-moving-average horizon for the online estimates.
    ema: float = 0.6
    adaptive: bool = True          # False reproduces the static formulation


@dataclass
class GameState:
    """Online estimates maintained across rounds."""
    kappa_D: float = None
    kappa_G: float = None
    c_epoch_D: float = None
    c_epoch_G: float = None
    history: list = field(default_factory=list)


class GameCoordinator:
    def __init__(self, params=None):
        self.p = params or GameParams()
        self.s = GameState(self.p.kappa_D, self.p.kappa_G,
                           self.p.c_epoch_D, self.p.c_epoch_G)

    # -- utility components (Eqs. 20-22) -------------------------------------
    def _eff_D(self, E_D):
        return 1.0 - math.exp(-self.s.kappa_D * E_D)

    def _eff_G(self, E_G):
        return 1.0 - math.exp(-self.s.kappa_G * E_G)

    def utility_D(self, E_D, E_G):
        return (self._eff_D(E_D)
                - self.p.lambda_D * self.s.c_epoch_D * E_D
                - self.p.gamma_nash * self._eff_G(E_G))

    def utility_G(self, E_D, E_G):
        return (self._eff_G(E_G)
                - self.p.lambda_G * self.s.c_epoch_G * E_G
                - self.p.gamma_nash * self._eff_D(E_D))

    # -- online re-estimation ------------------------------------------------
    def observe(self, obs):
        """Update effectiveness/cost estimates from the round that just ran.

        obs keys (all optional): d_loss_improvement, g_loss_improvement,
        E_D, E_G, d_epoch_seconds, g_epoch_seconds.
        """
        if not self.p.adaptive or not obs:
            return
        a = self.p.ema

        def upd(cur, new, lo, hi):
            if new is None or not math.isfinite(new):
                return cur
            return a * cur + (1 - a) * min(max(new, lo), hi)

        # Invert the effectiveness model: if a player realised relative loss
        # improvement r over E epochs, then 1-exp(-kappa*E) ~ r, so
        # kappa ~ -ln(1-r)/E.  Clamped to keep the model well posed.
        for who, imp_key, e_key in (("D", "d_loss_improvement", "E_D"),
                                    ("G", "g_loss_improvement", "E_G")):
            imp, E = obs.get(imp_key), obs.get(e_key)
            if imp is not None and E:
                r = min(max(float(imp), 0.0), 0.95)
                k_hat = -math.log(1.0 - r) / max(int(E), 1)
                cur = self.s.kappa_D if who == "D" else self.s.kappa_G
                new = upd(cur, k_hat, 1e-3, 1.0)
                if who == "D":
                    self.s.kappa_D = new
                else:
                    self.s.kappa_G = new

        for who, sec_key in (("D", "d_epoch_seconds"), ("G", "g_epoch_seconds")):
            sec = obs.get(sec_key)
            if sec is not None:
                cur = self.s.c_epoch_D if who == "D" else self.s.c_epoch_G
                new = upd(cur, float(sec), 1e-3, 100.0)
                if who == "D":
                    self.s.c_epoch_D = new
                else:
                    self.s.c_epoch_G = new

    # -- best-response coordination (Sec. IV-D3) -----------------------------
    def allocate(self):
        """Two-pass bounded best-response search over the discrete strategy sets."""
        p = self.p

        # Pass 1: for each E_D, the generator's best response.
        cand1, best1 = None, -float("inf")
        for E_D in p.E_D_range:
            E_G = max(p.E_G_range, key=lambda e: self.utility_G(E_D, e))
            u = self.utility_D(E_D, E_G)
            if u > best1:
                best1, cand1 = u, (E_D, E_G)

        # Pass 2: symmetrically, for each E_G the discriminator's best response.
        cand2, best2 = None, -float("inf")
        for E_G in p.E_G_range:
            E_D = max(p.E_D_range, key=lambda e: self.utility_D(e, E_G))
            u = self.utility_G(E_D, E_G)
            if u > best2:
                best2, cand2 = u, (E_D, E_G)

        # Select the pair with the higher joint payoff.
        def joint(pair):
            return self.utility_D(*pair) + self.utility_G(*pair)

        choice = cand1 if joint(cand1) >= joint(cand2) else cand2
        self.s.history.append({
            "E_D": choice[0], "E_G": choice[1],
            "kappa_D": self.s.kappa_D, "kappa_G": self.s.kappa_G,
            "c_epoch_D": self.s.c_epoch_D, "c_epoch_G": self.s.c_epoch_G,
            "joint_payoff": joint(choice),
            "is_equilibrium": cand1 == cand2,
        })
        return choice

    def equilibrium_gap(self):
        """Fraction of rounds where both passes agreed (a true fixed point).

        Reported in the paper's coordination analysis: the two-pass search is an
        approximation, and this quantifies how often it coincides with a mutual
        best response.
        """
        h = self.s.history
        if not h:
            return float("nan")
        return sum(1 for r in h if r["is_equilibrium"]) / len(h)


# ---------------------------------------------------------------------------
# Deadline-constrained epoch allocation.
#
# The unconstrained formulation above saturates: with no term that penalises a
# long round, the best response is always the largest epoch count in the
# strategy set, and after a few rounds the allocation is a constant.  That makes
# the mechanism indistinguishable from a fixed schedule -- which our ablation
# confirmed.
#
# In a vehicular deployment the binding constraint is not an abstract cost but a
# *deadline*: the round must finish while the selected vehicles are still in
# coverage.  Writing T_budget for the time the scheduler is willing to spend and
# t_D, t_G for measured per-epoch wall-clock costs, the feasible set is
#
#     E_D * t_D + E_G * t_G  <=  T_budget,
#
# which has an interior optimum and moves with the network state: when vehicles
# are fast-moving (short residence) the budget shrinks and the allocation must
# shift toward whichever player converts epochs into progress most efficiently.
# ---------------------------------------------------------------------------

class DeadlineCoordinator(GameCoordinator):
    """Best-response coordination under a per-round time budget."""

    def __init__(self, params=None, budget=None):
        super().__init__(params)
        self.budget = budget            # seconds; None -> unconstrained
        self.last_budget = None

    def set_budget(self, seconds):
        """Set the deadline for the coming round (e.g. from residence times)."""
        self.budget = None if seconds is None else max(float(seconds), 1e-3)
        self.last_budget = self.budget

    def _feasible(self, E_D, E_G):
        if self.budget is None:
            return True
        return E_D * self.s.c_epoch_D + E_G * self.s.c_epoch_G <= self.budget

    def allocate(self):
        p = self.p
        feas = [(d, g) for d in p.E_D_range for g in p.E_G_range
                if self._feasible(d, g)]
        if not feas:
            # Budget cannot accommodate even the smallest schedule: run the
            # cheapest feasible pair rather than skipping the round entirely.
            choice = (min(p.E_D_range), min(p.E_G_range))
            self.s.history.append({"E_D": choice[0], "E_G": choice[1],
                                   "kappa_D": self.s.kappa_D, "kappa_G": self.s.kappa_G,
                                   "c_epoch_D": self.s.c_epoch_D,
                                   "c_epoch_G": self.s.c_epoch_G,
                                   "joint_payoff": float("nan"),
                                   "is_equilibrium": False,
                                   "budget": self.budget, "infeasible": True})
            return choice

        # Two-pass best response restricted to the feasible set.
        cand1, best1 = None, -float("inf")
        for E_D in sorted({d for d, _ in feas}):
            gs = [g for d, g in feas if d == E_D]
            E_G = max(gs, key=lambda g: self.utility_G(E_D, g))
            u = self.utility_D(E_D, E_G)
            if u > best1:
                best1, cand1 = u, (E_D, E_G)
        cand2, best2 = None, -float("inf")
        for E_G in sorted({g for _, g in feas}):
            ds = [d for d, g in feas if g == E_G]
            E_D = max(ds, key=lambda d: self.utility_D(d, E_G))
            u = self.utility_G(E_D, E_G)
            if u > best2:
                best2, cand2 = u, (E_D, E_G)

        joint = lambda pr: self.utility_D(*pr) + self.utility_G(*pr)
        choice = cand1 if joint(cand1) >= joint(cand2) else cand2
        self.s.history.append({
            "E_D": choice[0], "E_G": choice[1],
            "kappa_D": self.s.kappa_D, "kappa_G": self.s.kappa_G,
            "c_epoch_D": self.s.c_epoch_D, "c_epoch_G": self.s.c_epoch_G,
            "joint_payoff": joint(choice), "is_equilibrium": cand1 == cand2,
            "budget": self.budget, "infeasible": False,
        })
        return choice
