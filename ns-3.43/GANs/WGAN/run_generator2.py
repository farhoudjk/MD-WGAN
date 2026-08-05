# run_generator2.py (modified)
import sys
import os
import numpy as np
import torch
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

from discriminator import Discriminator
from generator import Generator
from utils import visualize_samples, start_resource_monitor

# Setup logging
logging.basicConfig(filename='/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/generator.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def measure_resources():
    process = psutil.Process()
    cpu_time = process.cpu_times().user + process.cpu_times().system
    memory = process.memory_info().rss / 1024 / 1024
    return cpu_time, memory

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_generator2.py <node_id> [sender_ip]")
        logging.error("Invalid arguments")
        return 1

    node_id = int(sys.argv[1])
    sender_ip = sys.argv[2] if len(sys.argv) > 2 else "unknown"

    output_dir = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/generator_{node_id}"
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
        
        # Load training configuration
        config_path = f"{output_dir}/training_config.json"
        E_G = 3  # Default value
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    E_G = config.get("E_G", 3)
                    print(f"Loaded training config: E_G={E_G}")
                    logging.info(f"Loaded training config: E_G={E_G}")
            except Exception as e:
                print(f"Error reading training config: {e}")
                logging.error(f"Error reading training config: {e}")
        
        # Load RL agent recommendations
        rl_recommendation_path = f"{output_dir}/rl_recommendations.json"
        selected_discriminators = []
        if os.path.exists(rl_recommendation_path):
            try:
                with open(rl_recommendation_path, "r") as f:
                    rl_data = json.load(f)
                    selected_discriminators = rl_data.get("selected_discriminators", [])
                    print(f"RL agent selected discriminators: {selected_discriminators}")
                    logging.info(f"RL selected discriminators: {selected_discriminators}")
            except Exception as e:
                print(f"Error reading RL recommendations: {e}")
                logging.error(f"Error reading RL recommendations: {e}")
        
        # Fallback: Select discriminators with lowest recent loss
        if not selected_discriminators:
            print("No RL recommendations found, selecting discriminators based on recent loss")
            logging.warning("No RL recommendations, using fallback")
            disc_losses = {}
            base_disc_id = (node_id - 1) * 5
            for local_disc_id in range(1, 6):
                global_disc_id = base_disc_id + local_disc_id
                feedback_path = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/discriminator_{global_disc_id}/feedback_for_generator.json"
                if os.path.exists(feedback_path):
                    with open(feedback_path, "r") as f:
                        feedback = json.load(f)
                        disc_losses[local_disc_id] = feedback.get("loss", float('inf'))
                else:
                    disc_losses[local_disc_id] = float('inf')
                    logging.warning(f"No feedback for discriminator {global_disc_id}")
            selected_discriminators = [k for k, v in sorted(disc_losses.items(), key=lambda x: x[1])[:3]]
            print(f"Fallback selected discriminators: {selected_discriminators}")
            logging.info(f"Fallback selected discriminators: {selected_discriminators}")
        
        # Map to global discriminator IDs
        base_disc_id = (node_id - 1) * 5
        disc_ids = [base_disc_id + disc_id for disc_id in selected_discriminators]
        
        # Load sample dataset
        sample_data_path = f'/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets/discriminator_{base_disc_id + 1}/data.csv'
        sample_df = pd.read_csv(sample_data_path)
        data_dim = sample_df.shape[1]
        
        # Hyperparameters
        noise_dim = 64
        batch_size = 64
        
        # Initialize generator
        gen = Generator(noise_dim, data_dim).to(device)
        
        # Load previous state
        generator_path = f"{output_dir}/generator.pth"
        iteration = 0
        if os.path.exists(generator_path):
            gen.load_state_dict(torch.load(generator_path, map_location=device))
            print("Loaded previously trained generator")
            logging.info("Loaded previously trained generator")
            iteration_file = f"{output_dir}/iteration.txt"
            if os.path.exists(iteration_file):
                with open(iteration_file, "r") as f:
                    iteration = int(f.read().strip())
        
        # Setup optimizer
        gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Load a discriminator for loss computation
        disc = Discriminator(data_dim).to(device)
        disc_path = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/discriminator_{disc_ids[0]}/discriminator_{disc_ids[0]}.pth"
        if os.path.exists(disc_path):
            disc.load_state_dict(torch.load(disc_path, map_location=device))
            print(f"Loaded discriminator {disc_ids[0]} for loss computation")
            logging.info(f"Loaded discriminator {disc_ids[0]}")
        else:
            print(f"Warning: Discriminator {disc_ids[0]} model not found, initializing new")
            logging.warning(f"Discriminator {disc_ids[0]} model not found")
        
        # Generate fake samples for selected discriminators
        with torch.no_grad():
            for disc_id in disc_ids:
                noise = torch.randn(500, noise_dim).to(device)
                fake_samples = gen(noise)
                
                if torch.isnan(fake_samples).any():
                    print(f"Warning: NaN values in samples for discriminator {disc_id}. Replacing with zeros.")
                    logging.warning(f"NaN values in samples for discriminator {disc_id}")
                    fake_samples = torch.nan_to_num(fake_samples, nan=0.0)
                
                fake_samples_np = fake_samples.cpu().numpy()
                samples_path = f"{output_dir}/samples_for_disc_{disc_id}.csv"
                pd.DataFrame(fake_samples_np).to_csv(samples_path, index=False)
                print(f"Generated samples for discriminator {disc_id} (shape: {fake_samples_np.shape})")
                logging.info(f"Generated samples for discriminator {disc_id}")
        
        # Wait for discriminator feedback
        print("Waiting for discriminator feedback...")
        logging.info("Waiting for discriminator feedback")
        timeout = 30
        start_feedback_time = time.time()
        discriminator_feedback = []
        while len(discriminator_feedback) < len(disc_ids) and time.time() - start_feedback_time < timeout:
            for disc_id in disc_ids:
                feedback_path = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/discriminator_{disc_id}/feedback_for_generator.json"
                if os.path.exists(feedback_path):
                    if not any(d.get('id') == disc_id for d in discriminator_feedback):
                        try:
                            with open(feedback_path, "r") as f:
                                feedback = json.load(f)
                                feedback['id'] = disc_id
                                discriminator_feedback.append(feedback)
                                print(f"Received feedback from discriminator {disc_id}")
                                logging.info(f"Received feedback from discriminator {disc_id}: {feedback}")
                        except Exception as e:
                            logging.error(f"Error reading feedback for discriminator {disc_id}: {e}")
            if len(discriminator_feedback) < len(disc_ids):
                time.sleep(1)
        
        print(f"Collected feedback from {len(discriminator_feedback)} discriminators")
        logging.info(f"Collected feedback from {len(discriminator_feedback)} discriminators")
        
        # Train generator
        proxy_loss_value = 0.0  # Default value if training fails
        if discriminator_feedback:
            iteration += 1
            print(f"Training generator (iteration {iteration})")
            logging.info(f"Training generator (iteration {iteration})")
            for _ in range(E_G):
                gen_optimizer.zero_grad()
                noise = torch.randn(batch_size, noise_dim).to(device)
                fake_samples = gen(noise)
                
                if torch.isnan(fake_samples).any():
                    print("Warning: NaN values in fake samples. Replacing with zeros.")
                    logging.warning("NaN values in fake samples")
                    fake_samples = torch.nan_to_num(fake_samples, nan=0.0)
                
                # Compute loss using discriminator
                disc.eval()
                disc_scores = disc(fake_samples)
                proxy_loss = -torch.mean(disc_scores)  # Maximize discriminator scores
                proxy_loss_value = proxy_loss.item()
                
                print(f"Generator proxy loss: {proxy_loss_value:.4f}, requires_grad: {proxy_loss.requires_grad}, grad_fn: {proxy_loss.grad_fn}")
                logging.info(f"Generator proxy loss: {proxy_loss_value:.4f}, requires_grad: {proxy_loss.requires_grad}, grad_fn: {proxy_loss.grad_fn}")
                
                if torch.isnan(proxy_loss):
                    print("Warning: NaN in proxy_loss. Skipping iteration.")
                    logging.warning("NaN in proxy_loss")
                    continue
                
                proxy_loss.backward()
                gen_optimizer.step()
                disc.train()
            
            # Save generator
            torch.save(gen.state_dict(), generator_path)
            with open(f"{output_dir}/iteration.txt", "w") as f:
                f.write(str(iteration))
            
            # Generate and visualize samples
            with torch.no_grad():
                noise = torch.randn(500, noise_dim).to(device)
                generated_samples = gen(noise)
                
                if torch.isnan(generated_samples).any():
                    print("Warning: NaN in final samples. Replacing with zeros.")
                    logging.warning("NaN in final samples")
                    generated_samples = torch.nan_to_num(generated_samples, nan=0.0)
                
                generated_samples_np = generated_samples.cpu().numpy()
                samples_path = f"{output_dir}/generated_samples_iter_{iteration}.csv"
                pd.DataFrame(generated_samples_np).to_csv(samples_path, index=False)
                
                try:
                    real_data = pd.read_csv(sample_data_path).values
                    real_data = (real_data - np.mean(real_data, axis=0)) / (np.std(real_data, axis=0) + 1e-7)
                    visualize_samples(generated_samples_np, real_data, iteration, output_dir)
                except Exception as e:
                    print(f"Could not visualize: {e}")
                    logging.error(f"Could not visualize: {e}")
                    plt.figure(figsize=(10, 8))
                    for i in range(data_dim):
                        plt.subplot(5, 2, i+1)
                        data_col = generated_samples_np[:, i]
                        if not np.all(np.isnan(data_col)):
                            plt.hist(data_col, bins=30)
                            plt.title(f"Feature {i+1}")
                        else:
                            plt.text(0.5, 0.5, "All NaN", ha='center', va='center')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/generated_hist_iter_{iteration}.png")
                    plt.close()
        
        # Measure end resources and latency
        end_time = time.time()
        end_cpu, end_memory = measure_resources()
        latency = end_time - start_time
        cpu_usage = end_cpu - start_cpu
        memory_usage = end_memory
        
        # Estimate network traffic
        network_traffic = 0
        for disc_id in disc_ids:
            samples_path = f"{output_dir}/samples_for_disc_{disc_id}.csv"
            feedback_path = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/discriminator_{disc_id}/feedback_for_generator.json"
            if os.path.exists(samples_path):
                network_traffic += os.path.getsize(samples_path)
            if os.path.exists(feedback_path):
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
            "iteration": iteration,
            "timestamp": start_time,  # Use start time as the reference timestamp
            "latency": latency,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "network_traffic": network_traffic,
            "avg_disc_score": np.mean([f.get('mean_score', 0) for f in discriminator_feedback]) if discriminator_feedback else 0.0,
            "proxy_loss": proxy_loss_value
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
        
        print(f"Generator processing completed, iteration {iteration}")
        logging.info(f"Generator processing completed, iteration {iteration}")
        return 0
        
    except Exception as e:
        print(f"Error during generator processing: {str(e)}")
        logging.error(f"Error during generator processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        stop_monitor.set()
        end_time = time.time()
        end_cpu, end_memory = measure_resources()
        latency = end_time - start_time
        cpu_usage = end_cpu - start_cpu
        memory_usage = end_memory
        logging.info(f"Generator {node_id} completed in {latency:.2f}s with CPU {cpu_usage:.2f}s and memory {memory_usage:.2f}MB")
        return 0

if __name__ == "__main__":
    sys.exit(main())