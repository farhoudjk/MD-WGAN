"""Conditional WGAN-GP models for next-step vehicular trajectory prediction.

Both networks are conditioned on the observed context c (a window of W past
feature vectors), matching Eq. (1) and Sec. III-A2 of the paper: the generator
realises yhat = G(c, z) and the critic scores (c, y) pairs rather than marginal
samples.  Conditioning the critic is what makes the Wasserstein distance a
distance between *conditional* distributions, which is the quantity the paper's
RMSE/MAE evaluation actually probes.
"""
import torch
import torch.nn as nn


def _mlp(sizes, out_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LeakyReLU(0.2))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


class ContextEncoder(nn.Module):
    """Encodes the W x F context window into a fixed-width summary."""

    def __init__(self, window, n_features, hidden=64):
        super().__init__()
        self.net = _mlp([window * n_features, 128, hidden])
        self.act = nn.LeakyReLU(0.2)
        self.out_dim = hidden

    def forward(self, ctx):
        return self.act(self.net(ctx.flatten(1)))


class Generator(nn.Module):
    """yhat = G(c, z): predicts the next feature vector from context + noise.

    The head is a *bounded* residual around the last observed frame.  Two
    reasons, one statistical and one physical: an unbounded MLP extrapolates
    without limit on contexts outside its training support (producing deltas
    orders of magnitude larger than the data range), and a vehicle's state
    cannot change arbitrarily between consecutive samples anyway.  `max_delta`
    is expressed in standard deviations of the per-node scaler.
    """

    def __init__(self, window, n_features, noise_dim=16, hidden=128, max_delta=1.0):
        super().__init__()
        self.noise_dim = noise_dim
        self.max_delta = max_delta
        self.encoder = ContextEncoder(window, n_features)
        self.net = _mlp([self.encoder.out_dim + noise_dim, hidden, hidden, n_features],
                        out_activation=nn.Tanh())

    def forward(self, ctx, z=None):
        if z is None:
            z = torch.randn(ctx.size(0), self.noise_dim, device=ctx.device)
        h = self.encoder(ctx)
        delta = self.max_delta * self.net(torch.cat([h, z], dim=1))
        return track_last(ctx) + delta

    @torch.no_grad()
    def predict(self, ctx, n_samples=16):
        """Point prediction: the mean over n_samples latent draws.

        The Bayes-optimal estimator under squared error is the conditional mean,
        so RMSE/MAE are reported against this average rather than a single draw.
        """
        acc = 0.0
        for _ in range(n_samples):
            acc = acc + self(ctx)
        return acc / n_samples


def track_last(ctx):
    """The most recent observed frame in each context window."""
    return ctx[:, -1, :]


class Critic(nn.Module):
    """Conditional WGAN critic D(c, y): unbounded real-valued score."""

    def __init__(self, window, n_features, hidden=128):
        super().__init__()
        self.encoder = ContextEncoder(window, n_features)
        self.net = _mlp([self.encoder.out_dim + n_features, hidden, hidden, 1])

    def forward(self, ctx, y):
        h = self.encoder(ctx)
        return self.net(torch.cat([h, y], dim=1))


def gradient_penalty(critic, ctx, real_y, fake_y):
    """WGAN-GP penalty enforcing 1-Lipschitz continuity in y (Eq. 25)."""
    bs = real_y.size(0)
    eps = torch.rand(bs, 1, device=real_y.device)
    interp = (eps * real_y + (1 - eps) * fake_y).requires_grad_(True)
    scores = critic(ctx, interp)
    grads = torch.autograd.grad(
        outputs=scores, inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()
