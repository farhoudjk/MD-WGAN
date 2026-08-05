#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from scipy import linalg
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import euclidean_distances

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def preprocess_data(data, clip_percentile=99.5):  # Increased to 99.5
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    # Compute percentile-based threshold per feature
    thresholds = np.percentile(np.abs(data), clip_percentile, axis=0)
    # Clip extreme values per feature
    for i in range(data_tensor.size(1)):
        data_tensor[:, i] = torch.clamp(data_tensor[:, i], min=-thresholds[i], max=thresholds[i])
    # Standardization: zero mean, unit variance
    mean = data_tensor.mean(dim=0, keepdim=True)
    std = data_tensor.std(dim=0, keepdim=True) + 1e-7  # Avoid division by zero
    data_tensor = (data_tensor - mean) / std
    processed_data = data_tensor.cpu().numpy()
    print(f"Processed Real Data Min: {np.min(processed_data)}, Max: {np.max(processed_data)}")
    return processed_data

def calculate_fid(real_data, generated_data, batch_size=64, device=device):
    feature_model = FeatureExtractor().to(device)
    feature_model.eval()

    def preprocess_for_fid(data):
        data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        thresholds = np.percentile(np.abs(data), 99.5, axis=0)
        for i in range(data_tensor.size(1)):
            data_tensor[:, i] = torch.clamp(data_tensor[:, i], min=-thresholds[i], max=thresholds[i])
        mean = data_tensor.mean(dim=0, keepdim=True)
        std = data_tensor.std(dim=0, keepdim=True) + 1e-7
        data_tensor = (data_tensor - mean) / std
        return data_tensor

    real_data_tensor = preprocess_for_fid(real_data)
    real_features = []
    with torch.no_grad():
        for i in range(0, len(real_data_tensor), batch_size):
            batch = real_data_tensor[i:i + batch_size].to(device)
            batch = batch.view(batch.size(0), -1)
            feature = feature_model(batch)
            real_features.append(feature.cpu().numpy())
    real_features = np.concatenate(real_features, axis=0)
    mu_r = np.mean(real_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)

    generated_data_tensor = preprocess_for_fid(generated_data)
    gen_features = []
    with torch.no_grad():
        for i in range(0, len(generated_data_tensor), batch_size):
            batch = generated_data_tensor[i:i + batch_size].to(device)
            batch = batch.view(batch.size(0), -1)
            feature = feature_model(batch)
            gen_features.append(feature.cpu().numpy())
    gen_features = np.concatenate(gen_features, axis=0)
    mu_g = np.mean(gen_features, axis=0)
    sigma_g = np.cov(gen_features, rowvar=False)

    diff = mu_r - mu_g
    covmean = linalg.sqrtm(sigma_r.dot(sigma_g))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum(diff ** 2) + np.trace(sigma_r + sigma_g - 2 * covmean)

    return fid

def calculate_mse(real_data, generated_data):
    real_data_processed = preprocess_data(real_data)
    generated_data_processed = preprocess_data(generated_data)
    min_samples = min(real_data_processed.shape[0], generated_data_processed.shape[0])
    real_data_processed = real_data_processed[:min_samples]
    generated_data_processed = generated_data_processed[:min_samples]
    if np.any(np.isnan(real_data_processed)) or np.any(np.isinf(real_data_processed)) or \
       np.any(np.isnan(generated_data_processed)) or np.any(np.isinf(generated_data_processed)):
        raise ValueError("Processed data contains NaNs or infinities")
    mse = np.mean((real_data_processed - generated_data_processed) ** 2)
    return mse

def calculate_wasserstein_distance(real_data, generated_data):
    real_data_processed = preprocess_data(real_data)
    generated_data_processed = preprocess_data(generated_data)
    min_samples = min(real_data_processed.shape[0], generated_data_processed.shape[0])
    real_data_processed = real_data_processed[:min_samples]
    generated_data_processed = generated_data_processed[:min_samples]
    if np.any(np.isnan(real_data_processed)) or np.any(np.isinf(real_data_processed)) or \
       np.any(np.isnan(generated_data_processed)) or np.any(np.isinf(generated_data_processed)):
        raise ValueError("Processed data contains NaNs or infinities")
    w_dist = 0
    for i in range(real_data_processed.shape[1]):  # Per feature
        w_dist += wasserstein_distance(real_data_processed[:, i], generated_data_processed[:, i])
    w_dist /= real_data_processed.shape[1]  # Average over 17 features
    return w_dist

def calculate_diversity(generated_data):
    generated_data_processed = preprocess_data(generated_data)
    if np.any(np.isnan(generated_data_processed)) or np.any(np.isinf(generated_data_processed)):
        return np.nan
    distances = euclidean_distances(generated_data_processed)
    np.fill_diagonal(distances, np.inf)  # Ignore self-distances
    diversity = np.var(distances[np.isfinite(distances)])  # Handle potential nans
    return diversity if not np.isnan(diversity) else 0.0

def main():
    data_path = '/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets/discriminator_1/data.csv'
    df = pd.read_csv(data_path)
    real_data = df.values

    output_dir = f"/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/generator"
    generated_samples = pd.read_csv(f"{output_dir}/generated_samples_iter_7.csv").values

    min_samples = min(real_data.shape[0], generated_samples.shape[0])
    real_data = real_data[:min_samples]
    generated_data = generated_samples[:min_samples]

    # Print raw data ranges for debugging
    print(f"Real Data Min: {np.min(real_data)}, Max: {np.max(real_data)}")
    print(f"Generated Data Min: {np.min(generated_data)}, Max: {np.max(generated_data)}")

    # Calculate metrics
    fid_score = calculate_fid(real_data, generated_data)
    print(f"FID Score: {fid_score:.4f}")

    mse_score = calculate_mse(real_data, generated_data)
    print(f"MSE Score: {mse_score:.4f}")

    w_dist_score = calculate_wasserstein_distance(real_data, generated_data)
    print(f"Wasserstein Distance: {w_dist_score:.4f}")

    diversity_score = calculate_diversity(generated_data)
    print(f"Diversity Score: {diversity_score:.4f}")

    # Save all scores to a file
    with open(f"{output_dir}/metrics_score.txt", "w") as f:
        f.write(f"FID Score: {fid_score:.4f}\n")
        f.write(f"MSE Score: {mse_score:.4f}\n")
        f.write(f"Wasserstein Distance: {w_dist_score:.4f}\n")
        f.write(f"Diversity Score: {diversity_score:.4f}\n")

    # Baseline comparison with random data
    random_data = np.random.rand(*real_data.shape)
    fid_random = calculate_fid(real_data, random_data)
    mse_random = calculate_mse(real_data, random_data)
    w_dist_random = calculate_wasserstein_distance(real_data, random_data)
    diversity_random = calculate_diversity(random_data)
    print(f"FID Random: {fid_random:.4f}")
    print(f"MSE Random: {mse_random:.4f}")
    print(f"Wasserstein Random: {w_dist_random:.4f}")
    print(f"Diversity Random: {diversity_random:.4f}")
    with open(f"{output_dir}/metrics_score.txt", "a") as f:
        f.write(f"FID Random: {fid_random:.4f}\n")
        f.write(f"MSE Random: {mse_random:.4f}\n")
        f.write(f"Wasserstein Random: {w_dist_random:.4f}\n")
        f.write(f"Diversity Random: {diversity_random:.4f}\n")

if __name__ == "__main__":
    main()
