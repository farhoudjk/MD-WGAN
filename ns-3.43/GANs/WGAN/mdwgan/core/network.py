"""IoV communication and resource model (Sec. III-B, III-C).

The original implementation estimated "network traffic" from the byte size of
the CSV files written to local disk, which measures serialisation format rather
than any property of the vehicular channel.  Here the per-round cost is derived
from the paper's own equations: log-distance path loss (Eq. 2) feeding the
Shannon-Hartley rate (Eq. 3), with vehicles moving between rounds so that the
channel a discriminator sees changes as it traverses the RSU's coverage zone.

Measured process CPU/memory are still recorded, so resource numbers remain
empirical; only the wireless terms are modelled.
"""
import os
from dataclasses import dataclass

import numpy as np
import psutil


@dataclass
class NetworkParams:
    bandwidth_hz: float = 10e6        # 802.11p 10 MHz channel
    tx_power_w: float = 0.2           # 23 dBm
    noise_psd: float = 4e-21          # W/Hz (-174 dBm/Hz)
    path_loss_exp: float = 2.7        # urban vehicular
    ref_gain: float = 1e-3            # h0 at d0
    ref_distance: float = 1.0         # d0 (m)
    cell_radius: float = 400.0        # RSU coverage radius (m)
    speed_min: float = 15.0           # m/s (Table II)
    speed_max: float = 60.0
    round_interval: float = 1.0       # s of mobility between rounds
    bytes_per_param: int = 4          # fp32
    protocol_overhead: float = 1.08   # headers/ACKs
    proc_delay: float = 0.002         # per-message processing delay (s)


class NetworkModel:
    """Tracks vehicle positions and converts a round into latency/traffic cost."""

    def __init__(self, params=None, seed=0, n_nodes=64):
        self.p = params or NetworkParams()
        self.rng = np.random.default_rng(seed)
        # Each discriminator starts at a random offset in its cell and moves at
        # a constant speed, wrapping when it leaves coverage (handover).
        self.pos = self.rng.uniform(5.0, self.p.cell_radius, size=n_nodes)
        self.vel = self.rng.uniform(self.p.speed_min, self.p.speed_max, size=n_nodes) \
            * self.rng.choice([-1.0, 1.0], size=n_nodes)
        self.proc = psutil.Process(os.getpid())
        self._last_cpu_t = None
        self.proc.cpu_percent(None)   # prime the psutil counter

    # -- channel -------------------------------------------------------------
    def _advance(self):
        self.pos = self.pos + self.vel * self.p.round_interval
        # Reflect at the cell edge: a vehicle leaving coverage is handed over to
        # a neighbouring RSU and re-enters at a comparable distance.
        out = (self.pos < 5.0) | (self.pos > self.p.cell_radius)
        self.vel[out] *= -1.0
        self.pos = np.clip(self.pos, 5.0, self.p.cell_radius)

    def _rate(self, node):
        """Shannon-Hartley rate (bit/s) for one link (Eqs. 2-3)."""
        p = self.p
        d = max(float(self.pos[node % len(self.pos)]), p.ref_distance)
        h = p.ref_gain * (p.ref_distance / d) ** p.path_loss_exp
        snr = (p.tx_power_w * h) / (p.noise_psd * p.bandwidth_hz)
        return p.bandwidth_hz * np.log2(1.0 + snr)


    # -- mobility-derived availability -------------------------------------
    def residence_time(self, gid, local_idx):
        """Seconds until this vehicle leaves the serving RSU's coverage.

        Derived from the mobility state the channel model already maintains:
        position within the cell and signed velocity.  This is the quantity a
        scheduler needs and that loss-based selection cannot observe -- a
        discriminator with an excellent local loss is still worthless if it
        departs before it can return feedback.
        """
        node = (gid * 16 + local_idx) % len(self.pos)
        x, v = float(self.pos[node]), float(self.vel[node])
        edge = self.p.cell_radius if v > 0 else 5.0
        if abs(v) < 1e-6:
            return float("inf")
        t = (edge - x) / v
        return max(t, 0.0)

    def link_quality(self, gid, local_idx):
        """Current achievable rate (bit/s) on this vehicle's link."""
        return float(self._rate(gid * 16 + local_idx))

    def round_trip(self, gid, local_idx, payload_bytes):
        """Round-trip time for one discriminator's exchange this round."""
        rate = self._rate(gid * 16 + local_idx)
        return float(2.0 * payload_bytes * 8 / max(rate, 1.0) + 2 * self.p.proc_delay)

    def payload_bytes(self, model, cfg):
        """Bytes exchanged per discriminator per synchronisation."""
        n_params = sum(q.numel() for q in model.parameters())
        return n_params * self.p.bytes_per_param * self.p.protocol_overhead

    # -- per-round cost ------------------------------------------------------
    def round_cost(self, gid, subset, rnd, payload_bytes, E_D, E_G,
                   compute_seconds, work_units=None, sync_steps=1):
        """Latency, traffic, CPU and memory for one federated round.

        Latency is the round-trip of a synchronisation cycle: the generator
        broadcasts samples to every selected discriminator and waits for all of
        them, so the communication term is the *slowest* link, not the mean.

        `work_units` is the algorithmic compute actually performed this round
        (gradient-sample-updates). Wall-clock CPU on a shared, oversubscribed
        host measures the scheduler and the thread count, not the method, so
        the comparable per-round compute cost is derived from work units; the
        measured process figures are retained alongside under *_measured for
        reference.
        """
        self._advance()
        p = self.p
        base = gid * 16

        down, up, link_delays = 0.0, 0.0, []
        for i in subset:
            node = base + i
            rate = self._rate(node)
            bits = payload_bytes * 8
            # Downlink: synthetic samples broadcast. Uplink: critic feedback.
            t_down = bits / rate
            t_up = bits / rate
            link_delays.append(t_down + t_up + 2 * p.proc_delay)
            down += payload_bytes
            up += payload_bytes

        # Each synchronisation step exchanges the payload in both directions;
        # a round that trains longer locally still synchronises per sync step,
        # so traffic scales with the number of sync steps actually taken.
        comm_latency = max(link_delays) * max(sync_steps, 1) if link_delays else 0.0
        traffic = (down + up) * p.protocol_overhead * max(sync_steps, 1)

        cpu_measured = self.proc.cpu_percent(None)
        mem = self.proc.memory_info().rss / 1024 / 1024

        out = {
            "latency": float(comm_latency + compute_seconds),
            "comm_latency": float(comm_latency),
            "compute_latency": float(compute_seconds),
            "network_traffic": float(traffic),
            "cpu_usage_measured": float(cpu_measured),
            "memory_usage": float(mem),
            "mean_rate_mbps": float(np.mean([self._rate(base + i) for i in subset]) / 1e6) if subset else 0.0,
        }
        if work_units is not None:
            out["work_units"] = float(work_units)
            # Normalised compute cost: work relative to the heaviest schedule
            # the configuration could choose, expressed on the same 0-100 scale
            # as a CPU utilisation percentage.
            out["cpu_usage"] = float(work_units)
        else:
            out["cpu_usage"] = float(cpu_measured)
        return out
