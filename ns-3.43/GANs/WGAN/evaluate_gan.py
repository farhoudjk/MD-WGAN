import matplotlib.pyplot as plt
import torch
import pandas as pd  # <-- add this import
from generator import Generator

gen = Generator(17, 17)
gen.load_state_dict(torch.load('generator.pt'))

noise = torch.randn(100, 17)
fake_data = gen(noise).detach()

real_data = pd.read_csv('raw_traffic_data.csv').values

plt.scatter(fake_data[:, 0], fake_data[:, 1], label='Generated Data', alpha=0.7)
plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, label='Real Data')

plt.legend()
plt.title("GAN Generated Data vs Real Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()

