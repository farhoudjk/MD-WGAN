# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
evaluate_gan_tabular.py

Compute:
  1. A Frechet-style distance (FID-like) between real and generated tabular data
  2. Precision & Recall for generated samples via K-means cluster coverage

Usage:
  python3 evaluate_gan_tabular.py \
      --real path/to/real.csv \
      --fake path/to/generated.csv \
      --clusters 10
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm


def compute_fid_like(real, fake):
    """Compute FID-like distance between real and fake data."""
    mu_real = np.mean(real, axis=0)
    mu_fake = np.mean(fake, axis=0)

    sigma_real = np.cov(real, rowvar=False)
    sigma_fake = np.cov(fake, rowvar=False)

    cov_sqrt = sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    mean_term = np.sum((mu_real - mu_fake) ** 2)
    fid_score = mean_term + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return float(fid_score)


def compute_precision_recall(real, fake, n_clusters=10):
    """Estimate generator precision and recall via KMeans clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(real)

    real_labels = set(kmeans.predict(real))
    fake_labels = set(kmeans.predict(fake))

    common = real_labels & fake_labels
    precision = len(common) / len(fake_labels) if fake_labels else 0.0
    recall = len(common) / len(real_labels) if real_labels else 0.0
    return precision, recall


def main():
    parser = argparse.ArgumentParser(description="Evaluate GAN outputs on tabular data")
    parser.add_argument("--real", required=True, type=str, help="Path to real data CSV")
    parser.add_argument("--fake", required=True, type=str, help="Path to generated data CSV")
    parser.add_argument("--clusters", type=int, default=10, help="Number of KMeans clusters")
    args = parser.parse_args()

    # Load CSVs
    if not os.path.exists(args.real):
        raise FileNotFoundError(f"Real file not found: {args.real}")
    if not os.path.exists(args.fake):
        raise FileNotFoundError(f"Fake file not found: {args.fake}")

    real_df = pd.read_csv(args.real)
    fake_df = pd.read_csv(args.fake)

    if real_df.shape[1] != fake_df.shape[1]:
        raise ValueError(f"Mismatch in number of features: real={real_df.shape[1]}, fake={fake_df.shape[1]}")

    # Normalize
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_df.values)
    fake_scaled = scaler.transform(fake_df.values)

    # Compute metrics
    fid_score = compute_fid_like(real_scaled, fake_scaled)
    precision, recall = compute_precision_recall(real_scaled, fake_scaled, n_clusters=args.clusters)

    # Output
    print(f"FID-like Score      : {fid_score:.4f}")
    print(f"Precision (clusters): {precision:.4f}")
    print(f"Recall (clusters)   : {recall:.4f}")


if __name__ == "__main__":
    main()
