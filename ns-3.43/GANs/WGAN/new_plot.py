# Re-import libraries due to kernel reset
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Define the directory where metrics.json files are stored
base_results_dir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results"

# Function to load CPU and memory usage data from all generator and discriminator folders
def load_resource_metrics(results_dir):
    data = []

    # Match all metrics.json files in generator_* and discriminator_* folders
    metric_files = glob.glob(os.path.join(results_dir, "generator_*/metrics.json")) + \
                   glob.glob(os.path.join(results_dir, "discriminator_*/metrics.json"))

    for file in metric_files:
        try:
            df = pd.read_json(file)
            if isinstance(df, dict):  # convert dict to dataframe
                df = pd.DataFrame([df])
            df['node'] = os.path.basename(os.path.dirname(file))
            data.append(df)
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    if not data:
        return pd.DataFrame()
    
    return pd.concat(data, ignore_index=True)

# Load the data
metrics_df = load_resource_metrics(base_results_dir)

# Process and plot the metrics
output_paths = []

if not metrics_df.empty and 'timestamp' in metrics_df.columns:
    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'], unit='s', errors='coerce')
    metrics_df.sort_values(by='timestamp', inplace=True)

    # Plot 1: CPU usage for all nodes
    plt.figure(figsize=(12, 6))
    for node, group in metrics_df.groupby('node'):
        plt.plot(group['timestamp'], group['cpu_usage'], label=node)
    plt.title('CPU Usage per Node Over Time')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage (sec)')
    plt.legend()
    plt.tight_layout()
    path1 = "/mnt/data/cpu_usage_all_nodes.png"
    plt.savefig(path1)
    output_paths.append(path1)

    # Plot 2: Total CPU usage
    total_cpu = metrics_df.groupby('timestamp')['cpu_usage'].sum()
    plt.figure(figsize=(12, 6))
    plt.plot(total_cpu.index, total_cpu.values)
    plt.title('Total CPU Usage Over Time')
    plt.xlabel('Time')
    plt.ylabel('Total CPU Usage (sec)')
    plt.tight_layout()
    path2 = "/mnt/data/total_cpu_usage.png"
    plt.savefig(path2)
    output_paths.append(path2)

    # Plot 3: Memory usage for all nodes
    plt.figure(figsize=(12, 6))
    for node, group in metrics_df.groupby('node'):
        plt.plot(group['timestamp'], group['memory_usage'], label=node)
    plt.title('Memory Usage per Node Over Time')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage (MB)')
    plt.legend()
    plt.tight_layout()
    path3 = "/mnt/data/memory_usage_all_nodes.png"
    plt.savefig(path3)
    output_paths.append(path3)

    # Plot 4: Total Memory usage
    total_mem = metrics_df.groupby('timestamp')['memory_usage'].sum()
    plt.figure(figsize=(12, 6))
    plt.plot(total_mem.index, total_mem.values)
    plt.title('Total Memory Usage Over Time')
    plt.xlabel('Time')
    plt.ylabel('Total Memory Usage (MB)')
    plt.tight_layout()
    path4 = "/mnt/data/total_memory_usage.png"
    plt.savefig(path4)
    output_paths.append(path4)

output_paths
