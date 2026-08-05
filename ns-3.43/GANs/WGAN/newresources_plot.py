import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_node_data(base_path, node_number):
    """Load resource data for a specific node."""
    node_path = f"{base_path}/hybrid_node_{node_number}/resource_log.csv"
    
    try:
        # Read the CSV file
        df = pd.read_csv(node_path)
        
        # Add node identifier
        df['node'] = node_number
        
        # Convert timestamp to datetime for better plotting
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Calculate relative time (seconds from start)
        df['relative_time'] = df['timestamp'] - df['timestamp'].min()
        
        return df
    except FileNotFoundError:
        print(f"Warning: File not found for node {node_number}")
        return None
    except Exception as e:
        print(f"Error loading data for node {node_number}: {e}")
        return None

def load_all_nodes_data(base_path, num_nodes=20):
    """Load data for all nodes."""
    all_data = []
    
    for node_num in range(1, num_nodes + 1):
        node_data = load_node_data(base_path, node_num)
        if node_data is not None:
            all_data.append(node_data)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def plot_cpu_usage_per_node(df):
    """Plot CPU usage over time for each node."""
    plt.figure(figsize=(15, 8))
    
    # Get unique nodes
    nodes = sorted(df['node'].unique())
    
    for node in nodes:
        node_data = df[df['node'] == node]
        plt.plot(node_data['relative_time'], node_data['cpu_percent'], 
                label=f'Node {node}', alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Over Time for Each IoV Node')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('cpu_usage_per_node.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_memory_usage_per_node(df):
    """Plot Memory usage over time for each node."""
    plt.figure(figsize=(15, 8))
    
    # Get unique nodes
    nodes = sorted(df['node'].unique())
    
    for node in nodes:
        node_data = df[df['node'] == node]
        plt.plot(node_data['relative_time'], node_data['memory_mb'], 
                label=f'Node {node}', alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Time for Each IoV Node')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('memory_usage_per_node.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_total_cpu_usage(df):
    """Plot total CPU usage (sum of all nodes) over time."""
    plt.figure(figsize=(12, 6))
    
    # Group by timestamp and sum CPU usage
    total_cpu = df.groupby('relative_time')['cpu_percent'].sum().reset_index()
    
    plt.plot(total_cpu['relative_time'], total_cpu['cpu_percent'], 
             linewidth=2, color='red', marker='o', markersize=4)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Total CPU Usage (%)')
    plt.title('Total CPU Usage (All Nodes Combined) Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('total_cpu_usage.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_summary_statistics(df):
    """Plot summary statistics for additional insights."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Average CPU usage per node
    avg_cpu = df.groupby('node')['cpu_percent'].mean().sort_values(ascending=False)
    axes[0, 0].bar(range(len(avg_cpu)), avg_cpu.values)
    axes[0, 0].set_title('Average CPU Usage per Node')
    axes[0, 0].set_xlabel('Node')
    axes[0, 0].set_ylabel('Average CPU (%)')
    axes[0, 0].set_xticks(range(len(avg_cpu)))
    axes[0, 0].set_xticklabels([f'Node {i}' for i in avg_cpu.index], rotation=45)
    
    # Average Memory usage per node
    avg_memory = df.groupby('node')['memory_mb'].mean().sort_values(ascending=False)
    axes[0, 1].bar(range(len(avg_memory)), avg_memory.values)
    axes[0, 1].set_title('Average Memory Usage per Node')
    axes[0, 1].set_xlabel('Node')
    axes[0, 1].set_ylabel('Average Memory (MB)')
    axes[0, 1].set_xticks(range(len(avg_memory)))
    axes[0, 1].set_xticklabels([f'Node {i}' for i in avg_memory.index], rotation=45)
    
    # CPU usage distribution
    axes[1, 0].hist(df['cpu_percent'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('CPU Usage Distribution')
    axes[1, 0].set_xlabel('CPU Usage (%)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Memory usage distribution
    axes[1, 1].hist(df['memory_mb'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Memory Usage Distribution')
    axes[1, 1].set_xlabel('Memory Usage (MB)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate all charts."""
    # Base path for the data
    base_path = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results"
    
    print("Loading data for all nodes...")
    df = load_all_nodes_data(base_path, num_nodes=20)
    
    if df.empty:
        print("No data loaded. Please check the file paths.")
        return
    
    print(f"Loaded data for {len(df['node'].unique())} nodes")
    print(f"Total data points: {len(df)}")
    print(f"Time range: {df['relative_time'].min():.1f} to {df['relative_time'].max():.1f} seconds")
    
    # Generate all charts
    print("\nGenerating charts...")
    
    print("1. CPU usage per node chart...")
    plot_cpu_usage_per_node(df)
    
    print("2. Memory usage per node chart...")
    plot_memory_usage_per_node(df)
    
    print("3. Total CPU usage chart...")
    plot_total_cpu_usage(df)
    
    print("4. Summary statistics...")
    plot_summary_statistics(df)
    
    print("\nAll charts generated successfully!")
    print("Files saved:")
    print("- cpu_usage_per_node.png")
    print("- memory_usage_per_node.png")
    print("- total_cpu_usage.png")
    print("- summary_statistics.png")

# Alternative function to load data from current directory (for testing)
def load_sample_data():
    """Load sample data from current directory for testing."""
    try:
        df = pd.read_csv('resource_log.csv')
        df['node'] = 1  # Assign node 1 for sample data
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['relative_time'] = df['timestamp'] - df['timestamp'].min()
        return df
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return pd.DataFrame()

# Run the main function
if __name__ == "__main__":
    main()
    
    # Uncomment below to test with sample data in current directory
    # print("\nTesting with sample data...")
    # sample_df = load_sample_data()
    # if not sample_df.empty:
    #     print("Sample data loaded successfully!")
    #     print(sample_df.head())