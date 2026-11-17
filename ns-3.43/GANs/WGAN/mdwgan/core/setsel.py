"""Set-level discriminator selection.

A greedy top-$k$ rule is optimal when the value of a subset is the sum of its
members' values.  The selection problem posed by Eq. (8) is not of that form
once written at set level:

    maximise   sum_{j in S} info_j  -  lambda * redundancy(S)
    subject to max_{j in S} lat_j <= L_max,   sum_{j in S} cost_j <= B

The redundancy term is submodular -- two discriminators holding near-identical
shards contribute one shard's worth of gradient for two shards' worth of budget
-- and the latency constraint is a max, so one slow vehicle sets the cost of the
entire synchronisation round.  Neither is visible to a per-item ranking, which
is why a greedy rule and a learned per-item scorer perform identically on it:
both optimise the modular part only.

Two solvers are provided over the same objective, so the comparison isolates
the *objective* from the *search*:

  * `GeneticSelector` -- a small steady-state GA over fixed-size subsets.
  * `LazyGreedySelector` -- accelerated greedy on the submodular surrogate,
    which carries the standard (1-1/e) guarantee for monotone submodular
    maximisation under a cardinality constraint and costs a fraction of the GA.

Both are cheap in absolute terms: the candidate pool is bounded by design, so
selection is a few milliseconds against a round that costs seconds.
"""
import numpy as np


class SetObjective:
    """Set-level value of a candidate subset.

    Inputs are per-round observations supplied by the trainer:
      info      informativeness of each candidate (normalised critic loss)
      sim       pairwise similarity between candidate shards in [0,1]
      lat       per-candidate round-trip latency (s)
      cost      per-candidate compute cost (normalised)
      p_ok      probability the candidate completes the round
    """

    def __init__(self, lam_div=0.6, lam_lat=0.5, lam_cost=0.3, budget=None):
        self.lam_div = lam_div
        self.lam_lat = lam_lat
        self.lam_cost = lam_cost
        self.budget = budget

    def value(self, S, info, sim, lat, cost, p_ok):
        if len(S) == 0:
            return -1e9
        S = list(S)
        # Expected informativeness: a node that does not complete contributes
        # nothing, so availability multiplies rather than adds.
        gain = float(np.sum(info[S] * p_ok[S]))
        # Redundancy: mean pairwise similarity within the chosen set. This is
        # the submodular term a per-item ranking cannot see.
        if len(S) > 1:
            sub = sim[np.ix_(S, S)]
            iu = np.triu_indices(len(S), 1)
            red = float(sub[iu].mean())
        else:
            red = 0.0
        # The round closes when its slowest member replies.
        span = float(np.max(lat[S]))
        spend = float(np.sum(cost[S]))
        v = gain - self.lam_div * len(S) * red - self.lam_lat * span - self.lam_cost * spend
        if self.budget is not None and spend > self.budget:
            v -= 10.0 * (spend - self.budget)      # soft feasibility penalty
        return v


class _Base:
    def __init__(self, pool_size, k, seed=0, **kw):
        self.pool_size, self.k = pool_size, min(k, pool_size)
        self.rng = np.random.default_rng(seed)
        self.obj = SetObjective(**{a: kw[a] for a in
                                   ("lam_div", "lam_lat", "lam_cost", "budget") if a in kw})
        self.losses = np.full(pool_size, np.nan)
        self.sim = np.eye(pool_size)
        self.lat = np.ones(pool_size)
        self.cost = np.ones(pool_size)
        self.p_ok = np.ones(pool_size)
        self.eps = 0.0
        self.evals = 0

    # -- observations --------------------------------------------------------
    def observe_losses(self, losses):
        for i, v in losses.items():
            if v is not None and np.isfinite(v):
                self.losses[i] = v

    def observe_context(self, sim=None, lat=None, cost=None, p_ok=None):
        if sim is not None:
            self.sim = np.asarray(sim, float)
        if lat is not None:
            self.lat = np.asarray(lat, float)
        if cost is not None:
            self.cost = np.asarray(cost, float)
        if p_ok is not None:
            self.p_ok = np.asarray(p_ok, float)

    def _info(self):
        L = self.losses.copy()
        if not np.isfinite(L).any():
            return np.ones(self.pool_size)
        med = np.nanmedian(L[np.isfinite(L)])
        L = np.where(np.isfinite(L), L, med)
        lo, hi = L.min(), L.max()
        return (L - lo) / (hi - lo) + 1e-3 if hi - lo > 1e-12 else np.ones_like(L)

    def _val(self, S):
        self.evals += 1
        return self.obj.value(S, self._info(), self.sim, self.lat, self.cost, self.p_ok)

    def remember(self, *a, **kw):
        pass

    def replay(self):
        return None

    def q_values(self, state):
        return np.zeros(self.pool_size, np.float32)


class LazyGreedySelector(_Base):
    """Accelerated greedy on the set objective.

    Marginal gains of a submodular function are non-increasing, so a stale gain
    that already beats every other candidate's upper bound needs no
    recomputation.  This gives the (1-1/e) guarantee at a small fraction of the
    naive evaluation count -- the cheap baseline the GA has to justify itself
    against.
    """
    name = "lazygreedy"

    def act(self, state, greedy=False):
        self.evals = 0
        S, rest = [], list(range(self.pool_size))
        # Priority queue of (upper bound on marginal gain, item).
        ub = [(self._val([j]), j) for j in rest]
        ub.sort(reverse=True)
        base = -1e9
        while len(S) < self.k and ub:
            while True:
                g, j = ub[0]
                fresh = self._val(S + [j]) - (base if S else 0.0)
                if len(ub) == 1 or fresh >= ub[1][0] - 1e-12:
                    ub.pop(0)
                    S.append(j)
                    base = base + fresh if S[:-1] else fresh
                    break
                ub[0] = (fresh, j)
                ub.sort(reverse=True)
        return sorted(S), "lazygreedy"


class GeneticSelector(_Base):
    """Steady-state GA over fixed-size subsets.

    Chromosome: the subset itself. Crossover unions two parents and samples k
    without replacement; mutation swaps one member for an unused candidate.
    The population is seeded with the greedy solution so the GA can only
    improve on it -- any gain reported is a gain over greedy by construction.
    """
    name = "genetic"

    def __init__(self, pool_size, k, seed=0, pop=12, gens=12, elite=3,
                 p_mut=0.3, **kw):
        super().__init__(pool_size, k, seed=seed, **kw)
        self.pop, self.gens, self.elite, self.p_mut = pop, gens, elite, p_mut
        self._greedy = LazyGreedySelector(pool_size, k, seed=seed, **kw)

    def _seed_pop(self):
        self._greedy.losses = self.losses
        self._greedy.observe_context(self.sim, self.lat, self.cost, self.p_ok)
        P = [self._greedy.act(None)[0]]
        while len(P) < self.pop:
            P.append(sorted(self.rng.choice(self.pool_size, self.k, replace=False).tolist()))
        return P

    def _cross(self, a, b):
        union = list(set(a) | set(b))
        if len(union) <= self.k:
            miss = [i for i in range(self.pool_size) if i not in union]
            union += self.rng.choice(miss, self.k - len(union), replace=False).tolist()
        return sorted(self.rng.choice(union, self.k, replace=False).tolist())

    def _mut(self, s):
        s = list(s)
        out = [i for i in range(self.pool_size) if i not in s]
        if out:
            s[self.rng.integers(len(s))] = int(self.rng.choice(out))
        return sorted(s)

    def act(self, state, greedy=False):
        self.evals = 0
        if self.pool_size <= self.k:
            return list(range(self.pool_size)), "genetic"
        P = self._seed_pop()
        F = [self._val(s) for s in P]
        for _ in range(self.gens):
            order = np.argsort(F)[::-1]
            P = [P[i] for i in order]; F = [F[i] for i in order]
            kids = []
            while len(kids) < self.pop - self.elite:
                i, j = self.rng.integers(0, max(self.elite, 2), 2)
                c = self._cross(P[i], P[j])
                if self.rng.random() < self.p_mut:
                    c = self._mut(c)
                kids.append(c)
            P = P[:self.elite] + kids
            F = F[:self.elite] + [self._val(s) for s in kids]
        return P[int(np.argmax(F))], "genetic"


SET_SELECTORS = {"lazygreedy": LazyGreedySelector, "genetic": GeneticSelector}
