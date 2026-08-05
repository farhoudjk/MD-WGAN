#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt

base_dir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results"

def load_metrics_history(node_type, node_id):
    """Load or initialize metrics history for a node."""
    output_dir = f"{base_dir}/{node_type}_{node_id}"
    metrics_path = f"{output_dir}/metrics.json"
    history_path = f"{output_dir}/metrics_history.json"

    # Load current metrics
    if not os.path.exists(metrics_path):
        print(f"No metrics file found for {node_type} {node_id}")
        return []

    with open(metrics_path, "r") as f:
        current_metrics = json.load(f)

    # Load or initialize history
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = []

    # Append current metrics to history if new
    iteration = current_metrics.get("iteration", 1)
    if not any(m.get("iteration", 0) == iteration for m in history):
        history.append(current_metrics)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    return history

def interpolate_metrics(history, max_iterations=35):
    """Interpolate or pad metrics to cover iterations 1 to 35."""
    if not history:
        return {metric: [0] * max_iterations for metric in ['latency', 'cpu_usage', 'memory_usage', 'network_traffic']}

    # Sort by iteration
    history.sort(key=lambda x: x.get("iteration", 1))
    result = {metric: [0] * max_iterations for metric in ['latency', 'cpu_usage', 'memory_usage', 'network_traffic']}
    
    for metric in result:
        last_value = 0
        for i in range(max_iterations):
            for entry in history:
                if entry.get("iteration", 1) - 1 == i:
                    result[metric][i] = entry.get(metric, 0)
                    last_value = result[metric][i]
                    break
            else:
                result[metric][i] = last_value  # Use last known value or 0 if none

    return result

def plot_combined_metrics():
    """Generate a single chart per metric with all nodes and iterations 1 to 35."""
    NUM_GENERATORS = 4
    NUM_DISCRIMINATORS = 20
    max_iterations = 35
    node_labels = []

    # Collect metrics for all nodes
    all_metrics = {}
    for gen_id in range(1, NUM_GENERATORS + 1):
        history = load_metrics_history("generator", gen_id)
        all_metrics[f"Generator {gen_id}"] = interpolate_metrics(history, max_iterations)
        node_labels.append(f"Generator {gen_id}")

    for disc_id in range(1, NUM_DISCRIMINATORS + 1):
        history = load_metrics_history("discriminator", disc_id)
        all_metrics[f"Discriminator {disc_id}"] = interpolate_metrics(history, max_iterations)
        node_labels.append(f"Discriminator {disc_id}")

    if not all_metrics:
        print("No metrics available to plot.")
        return

    # Prepare data structures for each metric
    metrics_types = ['latency', 'cpu_usage', 'memory_usage', 'network_traffic']
    titles = ['Latency (s)', 'CPU Usage (s)', 'Memory Usage (MB)', 'Network Traffic (bytes)']
    colors = plt.cm.tab20(np.linspace(0, 1, len(node_labels)))  # Use tab20 for distinct colors

    for metric_idx, (metric_type, title) in enumerate(zip(metrics_types, titles)):
        plt.figure(figsize=(12, 6))

        # Plot a line for each node
        for node_idx, node_label in enumerate(node_labels):
            values = all_metrics[node_label][metric_type]
            plt.plot(range(1, max_iterations + 1), values, marker='o', label=node_label, color=colors[node_idx], linewidth=1)

        # Customize the plot
        plt.title(f"Combined {title} Across All Nodes (Iterations 1-35)")
        plt.xlabel("Iteration")
        plt.ylabel(title)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.xticks(range(1, max_iterations + 1, 5))  # Show every 5th iteration for clarity
        plt.tight_layout()

        # Save the chart
        output_path = f"{base_dir}/combined_{metric_type}_chart.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved {metric_type} chart to {output_path}")

if __name__ == "__main__":
    plot_combined_metrics()