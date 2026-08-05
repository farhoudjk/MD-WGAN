#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environment
import matplotlib.pyplot as plt
import time

# Make sure we can import from other files in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your GAN modules
from generator import Generator
from discriminator import Discriminator
from utils import calculate_gradient_penalty, visualize_samples

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_gan.py <node_id> [sender_ip]")
        return 1
    
    node_id = int(sys.argv[1])
    sender_ip = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    
    print(f"Starting GAN training for node {node_id}, packet from {sender_ip}")
    
    # Create output directory for this node
    output_dir = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/node_{node_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file path
    log_file = f"{output_dir}/training_log.txt"
    
    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load dataset
        data_path = '/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets/discriminator_1/data.csv'
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Hyperparameters
        noise_dim = 64
        data_dim = df.shape[1]
        batch_size = 64
        n_epochs = 50  # Reduced for quicker simulation
        
        # Initialize models
        gen = Generator(noise_dim, data_dim).to(device)
        disc = Discriminator(data_dim).to(device)
        
        # Setup optimizers
        gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
        disc_optimizer = torch.optim.Adam(disc.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Data preprocessing
        real_data = torch.tensor(df.values, dtype=torch.float32)
        real_data = (real_data - real_data.mean(dim=0)) / (real_data.std(dim=0) + 1e-7)
        real_data = real_data.to(device)
        
        # Training loop
        for epoch in range(n_epochs):
            # Training code (shortened version for NS3 simulation)
            for _ in range(5):  # Discriminator updates
                disc_optimizer.zero_grad()
                
                # Real samples
                indices = torch.randint(0, real_data.size(0), (batch_size,))
                real_samples = real_data[indices].to(device)
                
                # Fake samples
                noise = torch.randn(batch_size, noise_dim).to(device)
                fake_samples = gen(noise).detach()
                
                # WGAN-GP loss
                gradient_penalty = calculate_gradient_penalty(disc, real_samples, fake_samples, device)
                disc_loss = -(torch.mean(disc(real_samples)) - torch.mean(disc(fake_samples))) + 10 * gradient_penalty
                
                disc_loss.backward()
                disc_optimizer.step()
            
            # Generator update
            gen_optimizer.zero_grad()
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_samples = gen(noise)
            gen_loss = -torch.mean(disc(fake_samples))
            
            gen_loss.backward()
            gen_optimizer.step()
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | D Loss: {disc_loss.item():.4f} | G Loss: {gen_loss.item():.4f}")
                
                # Generate samples for visualization
                if epoch == n_epochs - 1:  # Only visualize at the end
                    with torch.no_grad():
                        test_noise = torch.randn(500, noise_dim).to(device)
                        generated_samples = gen(test_noise).cpu().numpy()
                        
                        # Save generated samples
                        pd.DataFrame(generated_samples).to_csv(
                            f"{output_dir}/generated_samples.csv", index=False)
                        
                        # Visualize samples
                        visualize_samples(generated_samples, real_data.cpu().numpy(), 
                                         epoch+1, output_dir)
        
        # Save final model
        torch.save(gen.state_dict(), f"{output_dir}/generator.pth")
        torch.save(disc.state_dict(), f"{output_dir}/discriminator.pth")
        
        print(f"Training completed successfully! Results saved to {output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error during GAN training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
