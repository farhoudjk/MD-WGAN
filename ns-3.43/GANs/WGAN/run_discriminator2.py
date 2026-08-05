# run_discriminator2.py (modified)
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import json
import psutil
import logging
import threading

from discriminator import Discriminator
from utils import calculate_gradient_penalty, start_resource_monitor

# Setup logging
logging.basicConfig(filename='/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/discriminator.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def measure_resources():
    process = psutil.Process()
    cpu_time = process.cpu_times().user + process.cpu_times().system
    memory = process.memory_info().rss / 1024 / 1024
    return cpu_time, memory

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_discriminator2.py <node_id> [sender_ip]")
        logging.error("Invalid arguments")
        return 1

    node_id = int(sys.argv[1])
    sender_ip = sys.argv[2] if len(sys.argv) > 2 else "unknown"

    output_dir = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/discriminator_{node_id}"
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()
    start_cpu, start_memory = measure_resources()

    monitor_log_path = f"{output_dir}/resource_log.csv"
    stop_monitor = threading.Event()
    monitor_event = start_resource_monitor(monitor_log_path, interval=1.0, stop_event=stop_monitor)
    
    try:
        # Start measuring resources and time
        start_time = time.time()
        start_cpu, start_memory = measure_resources()
        
        # Determine generator ID
        gen_id = ((node_id - 1) // 5) + 1
        
        # Load training configuration from corresponding generator
        config_path = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/generator_{gen_id}/training_config.json"
        E_D = 35  # Default value
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    E_D = config.get("E_D", 35)
                    print(f"Loaded training config: E_D={E_D}")
                    logging.info(f"Loaded training config: E_D={E_D}")
            except Exception as e:
                print(f"Error reading training config: {e}")
                logging.error(f"Error reading training config: {e}")
        
        # Load dataset
        data_path = f'/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets/discriminator_{node_id}/data.csv'
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with shape: {df.shape}")
        logging.info(f"Loaded dataset with shape: {df.shape}")
        
        # Hyperparameters
        noise_dim = 64
        data_dim = df.shape[1]
        batch_size = 64
        
        # Initialize discriminator
        disc = Discriminator(data_dim).to(device)
        
        # Setup optimizer
        disc_optimizer = torch.optim.Adam(disc.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Data preprocessing
        real_data = torch.tensor(df.values, dtype=torch.float32)
        real_data = (real_data - real_data.mean(dim=0)) / (real_data.std(dim=0) + 1e-7)
        real_data = real_data.to(device)
        
        # Load generator samples
        gen_id = ((node_id - 1) // 5) + 1
        generator_samples_path = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/generator_{gen_id}/samples_for_disc_{node_id}.csv"
        timeout = 60
        start_load_time = time.time()
        while not os.path.exists(generator_samples_path) and time.time() - start_load_time < timeout:
            time.sleep(0.5)
        
        if not os.path.exists(generator_samples_path):
            print(f"Warning: No generator samples at {generator_samples_path}")
            logging.warning(f"No generator samples at {generator_samples_path}")
            fake_data = torch.randn(500, data_dim).to(device)
        else:
            fake_df = pd.read_csv(generator_samples_path)
            fake_data = torch.tensor(fake_df.values, dtype=torch.float32).to(device)
        
        print(f"Loaded {len(fake_data)} fake samples")
        logging.info(f"Loaded {len(fake_data)} fake samples")
        
        # Training discriminator
        disc_loss = 0.0
        for iteration in range(E_D):
            disc_optimizer.zero_grad()
            real_indices = torch.randint(0, len(real_data), (batch_size,))
            fake_indices = torch.randint(0, len(fake_data), (batch_size,))
            
            real_batch = real_data[real_indices]
            fake_batch = fake_data[fake_indices]
            
            gradient_penalty = calculate_gradient_penalty(disc, real_batch, fake_batch, device)
            disc_loss = -(torch.mean(disc(real_batch)) - torch.mean(disc(fake_batch))) + 10 * gradient_penalty
            
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            disc_optimizer.step()
            
            print(f"Iteration {iteration+1}/{E_D} | Disc Loss: {disc_loss.item():.4f}")
            logging.info(f"Iteration {iteration+1}/{E_D} | Disc Loss: {disc_loss.item():.4f}")
        
        # Save discriminator
        torch.save(disc.state_dict(), f"{output_dir}/discriminator_{node_id}.pth")
        logging.info(f"Saved discriminator model to {output_dir}/discriminator_{node_id}.pth")
        
        # Evaluate and create feedback
        with torch.no_grad():
            disc_output = disc(fake_data)
            mean_score = disc_output.mean().item()
            std_score = disc_output.std().item()
            
            feedback = {
                "discriminator_id": node_id,
                "mean_score": mean_score,
                "std_score": std_score,
                "loss": disc_loss.item(),
                "timestamp": time.time()
            }
            
            feedback_path = f"{output_dir}/feedback_for_generator.json"
            with open(feedback_path, "w") as f:
                json.dump(feedback, f, indent=2)
            
            print(f"Feedback saved: mean_score={mean_score:.4f}, loss={disc_loss.item():.4f}")
            logging.info(f"Feedback saved to {feedback_path}: mean_score={mean_score:.4f}, loss={disc_loss.item():.4f}")
        
        # Measure end resources and latency
        end_time = time.time()
        end_cpu, end_memory = measure_resources()
        latency = end_time - start_time
        cpu_usage = end_cpu - start_cpu
        memory_usage = end_memory
        
        # Estimate network traffic
        network_traffic = os.path.getsize(generator_samples_path) if os.path.exists(generator_samples_path) else 0
        network_traffic += os.path.getsize(feedback_path)
        
        # Load or initialize metrics history
        history_path = f"{output_dir}/metrics_history.json"
        history = []
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        
        # Add new metrics with timestamp
        metrics = {
            "node_id": node_id,
            "timestamp": start_time,  # Use start time as the reference timestamp
            "latency": latency,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "network_traffic": network_traffic,
            "mean_score": mean_score,
            "loss": disc_loss.item()
        }
        history.append(metrics)
        
        # Save updated metrics history
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logging.info(f"Saved metrics history to {history_path}")
        
        # Save current metrics snapshot
        metrics_path = f"{output_dir}/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Saved metrics to {metrics_path}")
        
        print(f"Discriminator {node_id} training completed")
        logging.info(f"Discriminator {node_id} training completed")
        return 0
        
    except Exception as e:
        print(f"Error during discriminator training: {str(e)}")
        logging.error(f"Error during discriminator training: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        stop_monitor.set()
        end_time = time.time()
        end_cpu, end_memory = measure_resources()
        latency = end_time - start_time
        cpu_usage = end_cpu - start_cpu
        memory_usage = end_memory
        logging.info(f"Discriminator {node_id} completed in {latency:.2f}s with CPU {cpu_usage:.2f}s and memory {memory_usage:.2f}MB")
        return 0

if __name__ == "__main__":
    sys.exit(main())