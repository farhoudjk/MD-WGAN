#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import time
import json
import socket
import psutil
import threading

process = psutil.Process()

# Make sure we can import from other files in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your GAN modules
from discriminator import Discriminator
from utils import calculate_gradient_penalty

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
        print("Usage: python3 run_discriminator.py <node_id> [sender_ip]")
        return 1
    
    node_id = int(sys.argv[1])
    sender_ip = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    
    print(f"Starting discriminator {node_id}, received packet from {sender_ip}")
    
    # Create output directory for this discriminator
    output_dir = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/discriminator_{node_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file path
    log_file = f"{output_dir}/training_log.txt"
    
    # Start resource monitoring in a separate thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(node_id, output_dir, stop_event))
    monitor_thread.start()
    
    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load dataset specific to this discriminator
        data_path = f'/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets/discriminator_{node_id}/data.csv'
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Hyperparameters
        noise_dim = 64
        data_dim = df.shape[1]
        batch_size = 64
        
        # Initialize discriminator model
        disc = Discriminator(data_dim).to(device)
        
        # Setup optimizer
        disc_optimizer = torch.optim.Adam(disc.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Data preprocessing
        real_data = torch.tensor(df.values, dtype=torch.float32)
        real_data = (real_data - real_data.mean(dim=0)) / (real_data.std(dim=0) + 1e-7)
        real_data = real_data.to(device)
        
        # Request fake samples from generator (server)
        # In a real distributed system, this would involve network communication
        # For simulation, we'll assume generated samples are available at a specific path
        generator_samples_path = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/generator_{node_id//5 + 1}/samples_for_disc_{node_id}.csv"
        
        # Wait for generator samples (with timeout)
        timeout = 60  # seconds
        start_time = time.time()
        while not os.path.exists(generator_samples_path) and time.time() - start_time < timeout:
            time.sleep(0.5)
            
        if not os.path.exists(generator_samples_path):
            # Create dummy data if generator samples aren't available
            print(f"Warning: No generator samples found at {generator_samples_path}")
            fake_data = torch.randn(500, data_dim).to(device)
        else:
            # Load generated samples
            fake_df = pd.read_csv(generator_samples_path)
            fake_data = torch.tensor(fake_df.values, dtype=torch.float32).to(device)
            
        print(f"Loaded {len(fake_data)} fake samples for training")
        
        # Training discriminator
        for iteration in range(30):  # Do 30 training iterations
            # Train on real and fake data
            disc_optimizer.zero_grad()
            
            # Sample batches
            real_indices = torch.randint(0, len(real_data), (batch_size,))
            fake_indices = torch.randint(0, len(fake_data), (batch_size,))
            
            real_batch = real_data[real_indices]
            fake_batch = fake_data[fake_indices]
            
            # Calculate WGAN-GP loss
            gradient_penalty = calculate_gradient_penalty(disc, real_batch, fake_batch, device)
            disc_loss = -(torch.mean(disc(real_batch)) - torch.mean(disc(fake_batch))) + 10 * gradient_penalty
            
            disc_loss.backward()
            disc_optimizer.step()
            
            print(f"Iteration {iteration+1}/30 | Disc Loss: {disc_loss.item():.4f}")
        
        # Save the trained discriminator
        torch.save(disc.state_dict(), f"{output_dir}/discriminator_{node_id}.pth")
        
        # Evaluate against generator samples to get gradient information
        with torch.no_grad():
            # Use all fake samples for final evaluation
            disc_output = disc(fake_data)
            mean_score = disc_output.mean().item()
            std_score = disc_output.std().item()
            
            # Create feedback for generator
            feedback = {
                "discriminator_id": node_id,
                "mean_score": mean_score,
                "std_score": std_score,
                "loss": disc_loss.item(),
                "timestamp": time.time()
            }
            
            # Save feedback for generator
            with open(f"{output_dir}/feedback_for_generator.json", "w") as f:
                json.dump(feedback, f, indent=2)
                
            print(f"Discriminator feedback saved: mean_score={mean_score:.4f}, loss={disc_loss.item():.4f}")
        
        print(f"Discriminator {node_id} training completed")
        return 0
        
    except Exception as e:
        print(f"Error during discriminator training: {str(e)}")
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