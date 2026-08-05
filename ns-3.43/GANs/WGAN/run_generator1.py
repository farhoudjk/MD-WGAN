#!/usr/bin/env python3
import sys
import os
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import time
import json
import glob
import psutil
import threading

process = psutil.Process()

# Make sure we can import from other files in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your GAN modules
from generator import Generator
from utils import visualize_samples

def monitor_resources(node_id, output_dir, stop_event):
    usage_log = []
    start_time = time.time()
    while not stop_event.is_set():
        cpu = process.cpu_percent(interval=0.1)
        mem = process.memory_info().rss / (1024 * 1024)  # in MB
        usage_log.append({
            "timestamp": time.time() - start_time,
            "cpu_percent": cpu,
            "memory_mb": mem
        })
        time.sleep(0.5)  # Log every 0.5 seconds
    # Save to file
    with open(f"{output_dir}/resource_log_node_{node_id}.json", "w") as f:
        json.dump(usage_log, f, indent=2)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_generator.py <node_id> [sender_ip]")
        return 1
    
    node_id = int(sys.argv[1])
    sender_ip = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    
    print(f"Starting generator on server node {node_id}, received packet from {sender_ip}")
    
    # Create output directory for the generator
    output_dir = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/generator_{node_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Start resource monitoring in a separate thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(node_id, output_dir, stop_event))
    monitor_thread.start()
    
    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Number of discriminators in the system
        num_discriminators = 20
        
        # Load a sample dataset to get dimensions (assume all datasets have same structure)
        sample_data_path = '/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets/discriminator_1/data.csv'
        sample_df = pd.read_csv(sample_data_path)
        data_dim = sample_df.shape[1]
        
        # Hyperparameters
        noise_dim = 64
        batch_size = 64
        
        # Initialize generator model
        gen = Generator(noise_dim, data_dim).to(device)
        
        # Load previous state if available
        generator_path = f"{output_dir}/generator.pth"
        iteration = 0
        if os.path.exists(generator_path):
            gen.load_state_dict(torch.load(generator_path, map_location=device))
            print("Loaded previously trained generator")
            
            # Get iteration number from the filename if it exists
            iteration_file = f"{output_dir}/iteration.txt"
            if os.path.exists(iteration_file):
                with open(iteration_file, "r") as f:
                    iteration = int(f.read().strip())
        
        # Setup optimizer
        gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Generate fake samples for each discriminator
        with torch.no_grad():
            for disc_id in range(1, num_discriminators + 1):
                # Generate samples
                noise = torch.randn(500, noise_dim).to(device)
                fake_samples = gen(noise)
                
                # Check for NaN values and replace with zeros if necessary
                if torch.isnan(fake_samples).any():
                    print(f"Warning: NaN values detected in samples for discriminator {disc_id}. Replacing with zeros.")
                    fake_samples = torch.nan_to_num(fake_samples, nan=0.0)
                
                fake_samples_np = fake_samples.cpu().numpy()
                
                # Save samples for this discriminator
                samples_path = f"{output_dir}/samples_for_disc_{disc_id}.csv"
                pd.DataFrame(fake_samples_np).to_csv(samples_path, index=False)
                print(f"Generated samples for discriminator {disc_id} (shape: {fake_samples_np.shape})")
        
        # Wait for discriminator feedback (with timeout)
        print("Waiting for discriminator feedback...")
        timeout = 10  # seconds
        start_time = time.time()
        
        # Keep checking for feedback files until timeout
        discriminator_feedback = []
        while len(discriminator_feedback) < 4 and time.time() - start_time < timeout:
            for disc_id in range(1, num_discriminators + 1):
                feedback_path = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/discriminator_{disc_id}/feedback_for_generator.json"
                if os.path.exists(feedback_path):
                    # Check if we've already processed this feedback
                    if not any(d.get('id') == disc_id for d in discriminator_feedback):
                        with open(feedback_path, "r") as f:
                            feedback = json.load(f)
                            feedback['id'] = disc_id
                            discriminator_feedback.append(feedback)
                            print(f"Received feedback from discriminator {disc_id}")
            
            if len(discriminator_feedback) < 4:
                time.sleep(2)
        
        print(f"Collected feedback from {len(discriminator_feedback)} discriminators")
        
        # Train generator using discriminator feedback
        if discriminator_feedback:
            iteration += 1
            print(f"Training generator (iteration {iteration})")
            
            # Create a composite loss based on all discriminator feedback
            for _ in range(3):  # Generator update iterations
                gen_optimizer.zero_grad()
                noise = torch.randn(batch_size, noise_dim).to(device)
                fake_samples = gen(noise)
                
                # Check for NaN values in fake samples
                if torch.isnan(fake_samples).any():
                    print("Warning: NaN values detected in fake samples during training. Replacing with zeros.")
                    fake_samples = torch.nan_to_num(fake_samples, nan=0.0)
                
                # Fix proxy loss calculation: use mean score as target
                bce_loss = torch.nn.BCELoss()
                target_score = torch.ones(1, device=device)  # Ideal mean score (1.0)
                labels = torch.ones((batch_size, 1), device=device)
                predictions = torch.sigmoid(fake_samples.mean(dim=1, keepdim=True))
                proxy_loss = bce_loss(predictions, labels)
                
                # Apply scaling based on discriminator feedback
                avg_disc_score = np.mean([f.get('mean_score', 0) if not np.isnan(f.get('mean_score', 0)) else 0 for f in discriminator_feedback])
                scaled_loss = proxy_loss * (1.0 - min(max(avg_disc_score, -1.0), 1.0))
                
                if torch.isnan(scaled_loss):
                    print("Warning: NaN detected in scaled_loss. Skipping this iteration.")
                    continue
                
                scaled_loss.backward()
                gen_optimizer.step()
                
                print(f"Generator proxy loss: {scaled_loss.item():.4f}")
            
            # Save updated generator
            torch.save(gen.state_dict(), generator_path)
            
            # Save iteration number
            with open(f"{output_dir}/iteration.txt", "w") as f:
                f.write(str(iteration))
            
            # Generate and visualize samples
            with torch.no_grad():
                noise = torch.randn(500, noise_dim).to(device)
                generated_samples = gen(noise)
                
                # Check for NaN values
                if torch.isnan(generated_samples).any():
                    print("Warning: NaN values detected in final generated samples. Replacing with zeros.")
                    generated_samples = torch.nan_to_num(generated_samples, nan=0.0)
                
                generated_samples_np = generated_samples.cpu().numpy()
                
                # Save generated samples
                samples_path = f"{output_dir}/generated_samples_iter_{iteration}.csv"
                pd.DataFrame(generated_samples_np).to_csv(samples_path, index=False)
                
                # Try to load some real data for comparison
                try:
                    real_data = pd.read_csv(sample_data_path).values
                    real_data = (real_data - np.mean(real_data, axis=0)) / (np.std(real_data, axis=0) + 1e-7)
                    visualize_samples(generated_samples_np, real_data, iteration, output_dir)
                except Exception as e:
                    print(f"Could not visualize with real data comparison: {e}")
                    # Visualize the generated data only if it has valid values
                    plt.figure(figsize=(10, 8))
                    for i in range(data_dim):
                        plt.subplot(5, 2, i+1)
                        data_col = generated_samples_np[:, i]
                        if not np.all(np.isnan(data_col)):  # Skip if all NaN
                            plt.hist(data_col, bins=30)
                            plt.title(f"Feature {i+1}")
                        else:
                            plt.text(0.5, 0.5, "All NaN", ha='center', va='center')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/generated_hist_iter_{iteration}.png")
                    plt.close()
        
        print(f"Generator processing completed, iteration {iteration}")
        return 0
        
    except Exception as e:
        print(f"Error during generator processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Stop the monitoring thread
        stop_event.set()
        monitor_thread.join()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)