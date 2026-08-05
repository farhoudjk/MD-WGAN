#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from scipy import linalg

# Ensure compatibility with your environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom feature extractor (MLP) for 17D vectors
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=128, output_dim=64):
        super(FeatureExtractor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Function to calculate FID with custom features
def calculate_fid(real_data, generated_data, batch_size=64, device=device):
    # Initialize feature extractor
    feature_model = FeatureExtractor().to(device)
    feature_model.eval()

    # Preprocess data (normalize to [0, 1] for consistency)
    def preprocess_data(data):
        data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        data_tensor = (data_tensor - data_tensor.min()) / (data_tensor.max() - data_tensor.min() + 1e-7)
        return data_tensor

    # Process real data
    real_data_tensor = preprocess_data(real_data)
    real_features = []
    with torch.no_grad():
        for i in range(0, len(real_data_tensor), batch_size):
            batch = real_data_tensor[i:i + batch_size].to(device)
            # Add batch dimension and ensure 2D input [batch_size, 17]
            batch = batch.view(batch.size(0), -1)
            feature = feature_model(batch)
            real_features.append(feature.cpu().numpy())
    real_features = np.concatenate(real_features, axis=0)
    mu_r = np.mean(real_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)

    # Process generated data
    generated_data_tensor = preprocess_data(generated_data)
    gen_features = []
    with torch.no_grad():
        for i in range(0, len(generated_data_tensor), batch_size):
            batch = generated_data_tensor[i:i + batch_size].to(device)
            # Add batch dimension and ensure 2D input [batch_size, 17]
            batch = batch.view(batch.size(0), -1)
            feature = feature_model(batch)
            gen_features.append(feature.cpu().numpy())
    gen_features = np.concatenate(gen_features, axis=0)
    mu_g = np.mean(gen_features, axis=0)
    sigma_g = np.cov(gen_features, rowvar=False)

    # Calculate FID
    diff = mu_r - mu_g
    covmean = linalg.sqrtm(sigma_r.dot(sigma_g))
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # Handle numerical instability
    fid = np.sum(diff ** 2) + np.trace(sigma_r + sigma_g - 2 * covmean)

    return fid

# Example usage with your dataset
def main():
    # Load real data from CSV
    data_path = '/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets/discriminator_1/data.csv'
    df = pd.read_csv(data_path)
    real_data = df.values  # 17 features per sample

    # Load or generate fake data (from your trained generator)
    output_dir = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/generator"  # Adjust node_id
    generated_samples = pd.read_csv(f"{output_dir}/generated_samples_iter_1.csv").values

    # Ensure data shapes match (truncate if necessary)
    min_samples = min(real_data.shape[0], generated_samples.shape[0])
    real_data = real_data[:min_samples]
    generated_data = generated_samples[:min_samples]

    # Calculate FID
    fid_score = calculate_fid(real_data, generated_data)
    print(f"FID Score: {fid_score:.4f}")

    # Save FID score to a file
    with open(f"{output_dir}/fid_score.txt", "w") as f:
        f.write(f"FID Score: {fid_score:.4f}\n")

if __name__ == "__main__":
    main()
