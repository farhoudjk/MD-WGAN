import pandas as pd

# Load and preprocess VeReMi or NGSIM dataset
raw_data = pd.read_csv('raw_traffic_data.csv')

# Split data for each discriminator vehicle
for i in range(1, 1):  # Example: 5 discriminators
    discriminator_data = raw_data.sample(n=1000)
    discriminator_data.to_csv(f'/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets/discriminator_{i}/data.csv', index=False)

