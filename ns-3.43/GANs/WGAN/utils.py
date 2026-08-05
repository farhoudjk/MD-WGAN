# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import time
import csv
import threading
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# ? Gradient Penalty (WGAN-GP)
# -------------------------------------------------------------------
def calculate_gradient_penalty(discriminator, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(0), 1).to(device)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# -------------------------------------------------------------------
# ? Generator vs Real Sample Comparison Visualization
# -------------------------------------------------------------------
def visualize_samples(fake, real, iteration, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dim = fake.shape[1]
    plt.figure(figsize=(12, 8))
    for i in range(min(10, dim)):
        plt.subplot(2, 5, i+1)
        plt.hist(real[:, i], bins=30, alpha=0.6, label='Real')
        plt.hist(fake[:, i], bins=30, alpha=0.6, label='Fake')
        plt.title(f'Feature {i+1}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_comparison_iter_{iteration}.png"))
    plt.close()


# -------------------------------------------------------------------
# ? CPU/Memory Monitoring Utility
# -------------------------------------------------------------------
def start_resource_monitor(log_path, interval=1.0, stop_event=None):
    process = psutil.Process(os.getpid())

    def monitor():
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "cpu_percent", "memory_mb"])
            while not stop_event.is_set():
                cpu = process.cpu_percent(interval=None)
                mem = process.memory_info().rss / 1024 / 1024
                writer.writerow([time.time(), cpu, mem])
                f.flush()
                time.sleep(interval)

    stop_event = stop_event or threading.Event()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return stop_event
