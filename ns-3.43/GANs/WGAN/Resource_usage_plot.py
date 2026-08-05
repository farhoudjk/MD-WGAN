import json
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(filename='/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/resource_usage.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_metrics(base_dir, num_generators):
    """Load resource usage metrics from metrics.json files."""
    resource_data = {
        'latency': [],
        'cpu_usage': [],
        'memory_usage': [],
        'network_traffic': []
    }
    generator_ids = []
    
    for gen_id in range(1, num_generators + 1):
        metrics_path = f"{base_dir}/generator_{gen_id}/metrics.json"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    resource_data['latency'].append(metrics.get('latency', 0))
                    resource_data['cpu_usage'].append(metrics.get('cpu_usage', 0))
                    resource_data['memory_usage'].append(metrics.get('memory_usage', 0))
                    resource_data['network_traffic'].append(metrics.get('network_traffic', 0) / 1024)  # Convert to KB
                    generator_ids.append(f'Gen {gen_id}')
                    logging.info(f"Loaded metrics for generator {gen_id}: {metrics}")
            except Exception as e:
                logging.error(f"Error reading metrics for generator {gen_id}: {e}")
                resource_data['latency'].append(0)
                resource_data['cpu_usage'].append(0)
                resource_data['memory_usage'].append(0)
                resource_data['network_traffic'].append(0)
                generator_ids.append(f'Gen {gen_id}')
        else:
            logging.warning(f"Metrics file not found for generator {gen_id}")
            resource_data['latency'].append(0)
            resource_data['cpu_usage'].append(0)
            resource_data['memory_usage'].append(0)
            resource_data['network_traffic'].append(0)
            generator_ids.append(f'Gen {gen_id}')
    
    return generator_ids, resource_data

def plot_resource_usage(generator_ids, resource_data, output_path):
    """Plot a bar chart of resource usage for each generator."""
    metrics = ['latency', 'cpu_usage', 'memory_usage', 'network_traffic']
    labels = ['Latency (s)', 'CPU Usage (s)', 'Memory Usage (MB)', 'Network Traffic (KB)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    x = np.arange(len(generator_ids))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = resource_data[metric]
        ax.bar(x + i * width, values, width, label=label, color=color)
    
    ax.set_xlabel('Generator')
    ax.set_ylabel('Resource Usage')
    ax.set_title('Resource Usage by Generator')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(generator_ids)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved resource usage chart to {output_path}")

def main():
    base_dir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results"
    num_generators = 4
    output_path = f"{base_dir}/resource_usage_chart.png"
    
    try:
        generator_ids, resource_data = load_metrics(base_dir, num_generators)
        plot_resource_usage(generator_ids, resource_data, output_path)
        print(f"Resource usage chart saved to {output_path}")
    except Exception as e:
        logging.error(f"Error generating resource usage chart: {e}")
        print(f"Error generating chart: {e}")

if __name__ == "__main__":
    main()