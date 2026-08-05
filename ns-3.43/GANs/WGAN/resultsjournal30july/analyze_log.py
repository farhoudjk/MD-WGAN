import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_resource_usage_multiple_methods(base_path, log_filename='resource_log.csv'):
    """
    Aggregates and plots total CPU and memory usage from both discriminator
    and generator nodes, with custom timestamps.

    Args:
        base_path (str): The base directory containing the node folders.
        log_filename (str): The name of the log file in each node's directory.
    """
    all_data_list = []

    # Process discriminator nodes (1 to 20)
    for i in range(1, 21):
        node_folder = f"discriminator_{i}"
        file_path = os.path.join(base_path, node_folder, log_filename)
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if all(col in df.columns for col in ['timestamp', 'cpu_percent', 'memory_mb']):
                    # Correct the timestamp column to start from 1 and increase by 5 seconds per row
                    df['timestamp'] = 1 + np.arange(len(df)) * 5
                    all_data_list.append(df)
                else:
                    print(f"Skipping {file_path}: Missing 'timestamp', 'cpu_percent', or 'memory_mb' columns.")
            except pd.errors.EmptyDataError:
                print(f"Skipping {file_path}: The file is empty.")
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
            
    # Process generator nodes (1 to 4)
    for i in range(1, 5):
        node_folder = f"generator_{i}"
        file_path = os.path.join(base_path, node_folder, log_filename)
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if all(col in df.columns for col in ['timestamp', 'cpu_percent', 'memory_mb']):
                    # Correct the timestamp column to start from 1 and increase by 5 seconds per row
                    df['timestamp'] = 1 + np.arange(len(df)) * 5
                    all_data_list.append(df)
                else:
                    print(f"Skipping {file_path}: Missing 'timestamp', 'cpu_percent', or 'memory_mb' columns.")
            except pd.errors.EmptyDataError:
                print(f"Skipping {file_path}: The file is empty.")
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    if not all_data_list:
        print("No data was loaded. Please check the file paths and contents.")
        return

    # Concatenate all dataframes into a single dataframe
    all_data = pd.concat(all_data_list, ignore_index=True)

    # Group by the corrected timestamp and sum the usage from all nodes
    total_usage = all_data.groupby('timestamp').agg({
        'cpu_percent': 'sum',
        'memory_mb': 'sum'
    }).reset_index()

    # Plotting total CPU usage
    plt.figure(figsize=(12, 6))
    plt.plot(total_usage['timestamp'], total_usage['cpu_percent'], label='Total CPU Usage', color='blue')
    plt.title('Total CPU Usage Over Simulation Time')
    plt.xlabel('Simulation Time (seconds)')
    plt.ylabel('Total CPU Usage (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('total_cpu_usage.png')
    plt.close()
    print("Saved 'total_cpu_usage.png'")

    # Plotting total memory usage
    plt.figure(figsize=(12, 6))
    plt.plot(total_usage['timestamp'], total_usage['memory_mb'], label='Total Memory Usage', color='red')
    plt.title('Total Memory Usage Over Simulation Time')
    plt.xlabel('Simulation Time (seconds)')
    plt.ylabel('Total Memory Usage (MB)')
    plt.grid(True)
    plt.legend()
    plt.savefig('total_memory_usage.png')
    plt.close()
    print("Saved 'total_memory_usage.png'")

if __name__ == "__main__":
    # Define the base directory where your node folders are located
    base_directory = '/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results'

    # Call the function to perform the analysis
    analyze_resource_usage_multiple_methods(base_directory)