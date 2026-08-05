#!/usr/bin/env python3
# run_hybrid_node.py - Combined Generator and Discriminator functionality
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import json
import glob
import psutil
import logging
import threading
from collections import deque

# Import existing modules
from discriminator import Discriminator
from generator import Generator
from utils import visualize_samples, start_resource_monitor, calculate_gradient_penalty

# Setup logging
logging.basicConfig(filename='/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/hybrid_node.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class HybridGAN(nn.Module):
    """Combined Generator and Discriminator in one model"""
    def __init__(self, noise_dim, data_dim):
        super(HybridGAN, self).__init__()
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        
        # Generator network
        self.generator = Generator(noise_dim, data_dim)
        
        # Discriminator network  
        self.discriminator = Discriminator(data_dim)
        
        # Shared feature extractor (optional enhancement)
        self.shared_features = nn.Sequential(
            nn.Linear(data_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2)
        )
        
    def generate(self, noise):
        """Generate fake samples"""
        return self.generator(noise)
    
    def discriminate(self, data):
        """Discriminate real vs fake"""
        return self.discriminator(data)
    
    def extract_features(self, data):
        """Extract shared features"""
        return self.shared_features(data)

class AdaptiveLossBalancer:
    """Dynamically balance generator and discriminator training"""
    def __init__(self):
        self.gen_loss_history = deque(maxlen=10)
        self.disc_loss_history = deque(maxlen=10)
        self.balance_factor = 1.0
        
    def update(self, gen_loss, disc_loss):
        self.gen_loss_history.append(gen_loss)
        self.disc_loss_history.append(disc_loss)
        
        if len(self.gen_loss_history) >= 5:
            avg_gen = np.mean(self.gen_loss_history)
            avg_disc = np.mean(self.disc_loss_history)
            
            # Adjust balance based on relative performance
            if avg_gen > avg_disc * 1.5:
                self.balance_factor = min(2.0, self.balance_factor * 1.1)
            elif avg_disc > avg_gen * 1.5:
                self.balance_factor = max(0.5, self.balance_factor * 0.9)
                
    def get_gen_weight(self):
        return self.balance_factor
    
    def get_disc_weight(self):
        return 1.0 / self.balance_factor

def measure_resources():
    process = psutil.Process()
    cpu_time = process.cpu_times().user + process.cpu_times().system
    memory = process.memory_info().rss / 1024 / 1024
    return cpu_time, memory

def load_neighbor_samples(node_id, output_dir, max_samples=200):
    """Load samples from neighboring nodes for collaborative training"""
    neighbor_samples = []
    
    # Look for samples from other nodes
    base_results_dir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results"
    for neighbor_id in range(1, 21):  # Assuming 20 hybrid nodes
        if neighbor_id == node_id:
            continue
            
        neighbor_dir = f"{base_results_dir}/hybrid_node_{neighbor_id}"
        samples_pattern = f"{neighbor_dir}/generated_samples_*.csv"
        sample_files = glob.glob(samples_pattern)
        
        if sample_files:
            # Get the most recent sample file
            latest_file = max(sample_files, key=os.path.getctime)
            try:
                neighbor_df = pd.read_csv(latest_file)
                if len(neighbor_df) > 0:
                    # Sample a subset to avoid overwhelming memory
                    sample_size = min(max_samples // 19, len(neighbor_df))  # Distribute among neighbors
                    sampled_data = neighbor_df.sample(n=sample_size, replace=False) if len(neighbor_df) >= sample_size else neighbor_df
                    neighbor_samples.append(sampled_data.values)
                    logging.info(f"Loaded {len(sampled_data)} samples from neighbor {neighbor_id}")
            except Exception as e:
                logging.warning(f"Could not load samples from neighbor {neighbor_id}: {e}")
    
    if neighbor_samples:
        return np.vstack(neighbor_samples)
    else:
        return np.array([])

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 run_hybrid_node.py <node_id> <sender_ip> [gen_loss] [disc_loss]")
        logging.error("Invalid arguments")
        return 1

    node_id = int(sys.argv[1])
    sender_ip = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    initial_gen_loss = float(sys.argv[3]) if len(sys.argv) > 3 else 9999.0
    initial_disc_loss = float(sys.argv[4]) if len(sys.argv) > 4 else 9999.0

    output_dir = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/hybrid_node_{node_id}"
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()
    start_cpu, start_memory = measure_resources()

    monitor_log_path = f"{output_dir}/resource_log.csv"
    stop_monitor = threading.Event()
    monitor_event = start_resource_monitor(monitor_log_path, interval=1.0, stop_event=stop_monitor)
    
    try:
        # Load DDPG configuration
        ddpg_config_path = f"{output_dir}/ddpg_config.json"
        batch_size = 64
        gen_epochs = 3
        disc_epochs = 5
        
        if os.path.exists(ddpg_config_path):
            try:
                with open(ddpg_config_path, "r") as f:
                    config = json.load(f)
                    batch_size = config.get("batch_size", 64)
                    gen_epochs = config.get("gen_epochs", 3)
                    disc_epochs = config.get("disc_epochs", 5)
                    print(f"Loaded DDPG config: batch_size={batch_size}, gen_epochs={gen_epochs}, disc_epochs={disc_epochs}")
                    logging.info(f"Loaded DDPG config: batch_size={batch_size}, gen_epochs={gen_epochs}, disc_epochs={disc_epochs}")
            except Exception as e:
                print(f"Error reading DDPG config: {e}")
                logging.error(f"Error reading DDPG config: {e}")
        
        # Load dataset
        data_path = f'/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets/discriminator_{node_id}/data.csv'
        if not os.path.exists(data_path):
            # Fallback to a generic dataset
            data_path = f'/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets/discriminator_1/data.csv'
        
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with shape: {df.shape}")
        logging.info(f"Loaded dataset with shape: {df.shape}")
        
        # Hyperparameters
        noise_dim = 64
        data_dim = df.shape[1]
        
        # Initialize hybrid GAN
        hybrid_gan = HybridGAN(noise_dim, data_dim).to(device)
        
        # Load previous state
        model_path = f"{output_dir}/hybrid_gan.pth"
        iteration = 0
        if os.path.exists(model_path):
            hybrid_gan.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded previously trained hybrid GAN")
            logging.info("Loaded previously trained hybrid GAN")
            iteration_file = f"{output_dir}/iteration.txt"
            if os.path.exists(iteration_file):
                with open(iteration_file, "r") as f:
                    iteration = int(f.read().strip())
        
        # Setup optimizers
        gen_optimizer = torch.optim.Adam(hybrid_gan.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        disc_optimizer = torch.optim.Adam(hybrid_gan.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Data preprocessing
        real_data = torch.tensor(df.values, dtype=torch.float32)
        real_data = (real_data - real_data.mean(dim=0)) / (real_data.std(dim=0) + 1e-7)
        real_data = real_data.to(device)
        
        # Load neighbor samples for collaborative training
        neighbor_samples = load_neighbor_samples(node_id, output_dir)
        if len(neighbor_samples) > 0:
            neighbor_data = torch.tensor(neighbor_samples, dtype=torch.float32).to(device)
            # Normalize neighbor data with same statistics as real data
            neighbor_data = (neighbor_data - real_data.mean(dim=0)) / (real_data.std(dim=0) + 1e-7)
            # Combine with real data
            combined_real_data = torch.cat([real_data, neighbor_data], dim=0)
            print(f"Combined real data with {len(neighbor_samples)} neighbor samples")
            logging.info(f"Combined real data with {len(neighbor_samples)} neighbor samples")
        else:
            combined_real_data = real_data
        
        # Initialize adaptive loss balancer
        loss_balancer = AdaptiveLossBalancer()
        
        iteration += 1
        print(f"Starting training iteration {iteration}")
        logging.info(f"Starting training iteration {iteration}")
        
        # Training phase
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        
        # Discriminator training
        hybrid_gan.discriminator.train()
        hybrid_gan.generator.eval()
        
        for disc_iter in range(disc_epochs):
            disc_optimizer.zero_grad()
            
            # Real batch
            real_indices = torch.randint(0, len(combined_real_data), (batch_size,))
            real_batch = combined_real_data[real_indices]
            
            # Fake batch
            noise = torch.randn(batch_size, noise_dim).to(device)
            with torch.no_grad():
                fake_batch = hybrid_gan.generate(noise)
            
            # Discriminator loss with gradient penalty
            real_scores = hybrid_gan.discriminate(real_batch)
            fake_scores = hybrid_gan.discriminate(fake_batch)
            
            gradient_penalty = calculate_gradient_penalty(hybrid_gan.discriminator, real_batch, fake_batch, device)
            disc_loss = -(torch.mean(real_scores) - torch.mean(fake_scores)) + 10 * gradient_penalty
            
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(hybrid_gan.discriminator.parameters(), max_norm=1.0)
            disc_optimizer.step()
            
            disc_loss_total += disc_loss.item()
            
        avg_disc_loss = disc_loss_total / disc_epochs
        
        # Generator training
        hybrid_gan.generator.train()
        hybrid_gan.discriminator.eval()
        
        for gen_iter in range(gen_epochs):
            gen_optimizer.zero_grad()
            
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_samples = hybrid_gan.generate(noise)
            
            # Generator loss
            fake_scores = hybrid_gan.discriminate(fake_samples)
            gen_loss = -torch.mean(fake_scores)
            
            # Apply adaptive weighting
            weighted_gen_loss = gen_loss * loss_balancer.get_gen_weight()
            
            weighted_gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(hybrid_gan.generator.parameters(), max_norm=1.0)
            gen_optimizer.step()
            
            gen_loss_total += gen_loss.item()
        
        avg_gen_loss = gen_loss_total / gen_epochs
        
        # Update loss balancer
        loss_balancer.update(avg_gen_loss, avg_disc_loss)
        
        print(f"Training completed - Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
        logging.info(f"Training completed - Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
        
        # Save hybrid GAN
        torch.save(hybrid_gan.state_dict(), model_path)
        with open(f"{output_dir}/iteration.txt", "w") as f:
            f.write(str(iteration))
        
        # Generate samples
        hybrid_gan.generator.eval()
        with torch.no_grad():
            noise = torch.randn(500, noise_dim).to(device)
            generated_samples = hybrid_gan.generate(noise)
            
            if torch.isnan(generated_samples).any():
                print("Warning: NaN in generated samples. Replacing with zeros.")
                logging.warning("NaN in generated samples")
                generated_samples = torch.nan_to_num(generated_samples, nan=0.0)
            
            generated_samples_np = generated_samples.cpu().numpy()
            samples_path = f"{output_dir}/generated_samples_iter_{iteration}.csv"
            pd.DataFrame(generated_samples_np).to_csv(samples_path, index=False)
            
            # Evaluate quality metrics
            real_scores = hybrid_gan.discriminate(real_data[:200])  # Sample for efficiency
            fake_scores = hybrid_gan.discriminate(generated_samples[:200])
            
            real_score_mean = torch.mean(real_scores).item()
            fake_score_mean = torch.mean(fake_scores).item()
            score_diff = real_score_mean - fake_score_mean
            
            print(f"Quality metrics - Real score: {real_score_mean:.4f}, Fake score: {fake_score_mean:.4f}, Diff: {score_diff:.4f}")
            logging.info(f"Quality metrics - Real score: {real_score_mean:.4f}, Fake score: {fake_score_mean:.4f}, Diff: {score_diff:.4f}")
            
            # Visualize samples
            try:
                real_data_np = combined_real_data.cpu().numpy()
                visualize_samples(generated_samples_np, real_data_np, iteration, output_dir)
            except Exception as e:
                print(f"Could not visualize: {e}")
                logging.error(f"Could not visualize: {e}")
                # Fallback histogram
                plt.figure(figsize=(12, 8))
                for i in range(min(data_dim, 10)):
                    plt.subplot(2, 5, i+1)
                    data_col = generated_samples_np[:, i]
                    if not np.all(np.isnan(data_col)):
                        plt.hist(data_col, bins=30, alpha=0.7, label='Generated')
                        if i < real_data_np.shape[1]:
                            plt.hist(real_data_np[:, i], bins=30, alpha=0.7, label='Real')
                        plt.title(f"Feature {i+1}")
                        plt.legend()
                    else:
                        plt.text(0.5, 0.5, "All NaN", ha='center', va='center')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/comparison_hist_iter_{iteration}.png")
                plt.close()
        
        # Measure end resources and latency
        end_time = time.time()
        end_cpu, end_memory = measure_resources()
        latency = end_time - start_time
        cpu_usage = end_cpu - start_cpu
        memory_usage = end_memory
        
        # Estimate network traffic (samples sent/received)
        network_traffic = 0
        if os.path.exists(samples_path):
            network_traffic += os.path.getsize(samples_path)
        
        # Calculate additional metrics
        diversity_score = np.std(generated_samples_np.flatten())  # Simple diversity measure
        
        # Load or initialize metrics history
        history_path = f"{output_dir}/metrics_history.json"
        history = []
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        
        # Add new metrics with timestamp
        metrics = {
            "node_id": node_id,
            "iteration": iteration,
            "timestamp": start_time,
            "latency": latency,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "network_traffic": network_traffic,
            "gen_loss": avg_gen_loss,
            "disc_loss": avg_disc_loss,
            "real_score_mean": real_score_mean,
            "fake_score_mean": fake_score_mean,
            "score_difference": score_diff,
            "diversity_score": diversity_score,
            "balance_factor": loss_balancer.balance_factor,
            "neighbor_samples_used": len(neighbor_samples) if len(neighbor_samples) > 0 else 0
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
        
        # Save state for DDPG agent
        ddpg_state_path = f"{output_dir}/ddpg_state.json"
        ddpg_state = {
            "gen_loss": avg_gen_loss,
            "disc_loss": avg_disc_loss,
            "score_difference": score_diff,
            "diversity_score": diversity_score,
            "latency": latency,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "iteration": iteration
        }
        
        with open(ddpg_state_path, "w") as f:
            json.dump(ddpg_state, f, indent=2)
        logging.info(f"Saved DDPG state to {ddpg_state_path}")
        
        print(f"Hybrid node {node_id} processing completed, iteration {iteration}")
        logging.info(f"Hybrid node {node_id} processing completed, iteration {iteration}")
        return 0
        
    except Exception as e:
        print(f"Error during hybrid node processing: {str(e)}")
        logging.error(f"Error during hybrid node processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        stop_monitor.set()
        end_time = time.time()
        end_cpu, end_memory = measure_resources()
        latency = end_time - start_time
        cpu_usage = end_cpu - start_cpu
        memory_usage = end_memory
        logging.info(f"Hybrid node {node_id} completed in {latency:.2f}s with CPU {cpu_usage:.2f}s and memory {memory_usage:.2f}MB")
        return 0

if __name__ == "__main__":
    sys.exit(main())