import torch
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from generator import Generator
from discriminator import Discriminator
from utils import calculate_gradient_penalty, visualize_samples

# Hyperparameters
noise_dim = 64  # Reduced noise dimension
data_dim = 17
lr = 0.0001  # Increased learning rate
n_epochs = 100  # More epochs
batch_size = 128  # Larger batch size
lambda_gp = 10  # Gradient penalty weight
n_critic = 5  # Number of critic updates per generator update

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize models
gen = Generator(noise_dim, data_dim).to(device)
disc = Discriminator(data_dim).to(device)

# Setup optimizers (using Adam instead of RMSprop)
gen_optimizer = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
disc_optimizer = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

# Load and preprocess data
data_path = '/home/yorku/Downloads/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets/discriminator_1/data.csv'
df = pd.read_csv(data_path)
print(f"Loaded dataset with shape: {df.shape}")

# Robust normalization (using median and IQR)
real_data = torch.tensor(df.values, dtype=torch.float32)
medians = torch.median(real_data, dim=0)[0]
q1 = torch.quantile(real_data, 0.25, dim=0)
q3 = torch.quantile(real_data, 0.75, dim=0)
iqr = q3 - q1
# Clip outliers and normalize
real_data = torch.clamp(real_data, medians - 3*iqr, medians + 3*iqr)
real_data = (real_data - real_data.mean(dim=0)) / (real_data.std(dim=0) + 1e-7)
real_data = real_data.to(device)
print("Data normalized and ready for training")

# Create directory for results
results_dir = "./gan_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Training loop
print(f"Starting training for {n_epochs} epochs")
disc_losses = []
gen_losses = []

for epoch in range(n_epochs):
    epoch_disc_losses = []
    epoch_gen_losses = []
    
    for _ in range(n_critic):  # Discriminator update steps
        disc_optimizer.zero_grad()
        
        # Real samples
        indices = torch.randint(0, real_data.size(0), (batch_size,))
        real_samples = real_data[indices].to(device)
        
        # Fake samples
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_samples = gen(noise).detach()
        
        # WGAN-GP loss
        gradient_penalty = calculate_gradient_penalty(disc, real_samples, fake_samples, device)
        disc_loss = -(torch.mean(disc(real_samples)) - torch.mean(disc(fake_samples))) + lambda_gp * gradient_penalty
        
        disc_loss.backward()
        disc_optimizer.step()
        epoch_disc_losses.append(disc_loss.item())

    # Generator update
    gen_optimizer.zero_grad()
    noise = torch.randn(batch_size, noise_dim).to(device)
    fake_samples = gen(noise)
    gen_loss = -torch.mean(disc(fake_samples))
    
    gen_loss.backward()
    gen_optimizer.step()
    epoch_gen_losses.append(gen_loss.item())

    # Print progress
    if (epoch + 1) % 10 == 0:
        avg_disc_loss = sum(epoch_disc_losses) / len(epoch_disc_losses)
        avg_gen_loss = sum(epoch_gen_losses) / len(epoch_gen_losses)
        disc_losses.append(avg_disc_loss)
        gen_losses.append(avg_gen_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Disc Loss: {avg_disc_loss:.4f} | Gen Loss: {avg_gen_loss:.4f}")
        
        # Generate and visualize samples
        with torch.no_grad():
            test_noise = torch.randn(1000, noise_dim).to(device)
            generated_samples = gen(test_noise).cpu().numpy()
            visualize_samples(generated_samples, real_data.cpu().numpy(), epoch+1, results_dir)

# Save models
torch.save(gen.state_dict(), f"{results_dir}/generator.pth")
torch.save(disc.state_dict(), f"{results_dir}/discriminator.pth")
print(f"Models saved to {results_dir}")

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(0, n_epochs, 10), disc_losses, label='Discriminator Loss')
plt.plot(range(0, n_epochs, 10), gen_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.savefig(f"{results_dir}/loss_curves.png")
plt.close()

print("Training complete!")
