#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import logging

# Setup logging
logging.basicConfig(filename='/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/rl_agent.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class GameTheoryOptimizer:
    def __init__(self):
        self.alpha = 0.05  # Discriminator learning rate parameter
        self.beta = 0.1   # Generator learning rate parameter
        self.c_D = 0.3    # Cost per epoch for discriminator (seconds)
        self.c_G = 0.8    # Cost per epoch for generator (seconds)
        self.lambda_D = 0.05  # Weight for discriminator resource cost
        self.lambda_G = 0.05 # Weight for generator resource cost
        self.gamma = 0.7     # Penalty for opponent's effectiveness
        self.E_D_range = list(range(45, 76, 5))  # Possible E_D values: 5, 10, ..., 40
        self.E_G_range = list(range(2, 16))     # Possible E_G values: 1, 2, ..., 10
    
    def effectiveness_D(self, E_D):
        return 1 - np.exp(-self.alpha * E_D)
    
    def effectiveness_G(self, E_G):
        return 1 - np.exp(-self.beta * E_G)
    
    def cost_D(self, E_D):
        return self.c_D * E_D
    
    def cost_G(self, E_G):
        return self.c_G * E_G
    
    def payoff_D(self, E_D, E_G):
        eff_D = self.effectiveness_D(E_D)
        eff_G = self.effectiveness_G(E_G)
        cost = self.cost_D(E_D)
        return eff_D - self.lambda_D * cost - self.gamma * eff_G
    
    def payoff_G(self, E_D, E_G):
        eff_D = self.effectiveness_D(E_D)
        eff_G = self.effectiveness_G(E_G)
        cost = self.cost_G(E_G)
        return eff_G - self.lambda_G * cost - self.gamma * eff_D
    
    def find_nash_equilibrium(self):
        best_E_D = self.E_D_range[0]
        best_E_G = self.E_G_range[0]
        best_payoff_D = -float('inf')
        best_payoff_G = -float('inf')
        
        # For each possible E_D, find the best E_G response
        for E_D in self.E_D_range:
            max_G_payoff = -float('inf')
            best_E_G_for_D = self.E_G_range[0]
            for E_G in self.E_G_range:
                payoff_G = self.payoff_G(E_D, E_G)
                if payoff_G > max_G_payoff:
                    max_G_payoff = payoff_G
                    best_E_G_for_D = E_G
            # Compute discriminator payoff for this pair
            payoff_D = self.payoff_D(E_D, best_E_G_for_D)
            if payoff_D > best_payoff_D:
                best_payoff_D = payoff_D
                best_E_D = E_D
                best_E_G = best_E_G_for_D
        
        # For each possible E_G, find the best E_D response
        for E_G in self.E_G_range:
            max_D_payoff = -float('inf')
            best_E_D_for_G = self.E_D_range[0]
            for E_D in self.E_D_range:
                payoff_D = self.payoff_D(E_D, E_G)
                if payoff_D > max_D_payoff:
                    max_D_payoff = payoff_D
                    best_E_D_for_G = E_D
            # Compute generator payoff for this pair
            payoff_G = self.payoff_G(best_E_D_for_G, E_G)
            if payoff_G > best_payoff_G:
                best_payoff_G = payoff_G
                best_E_G = E_G
                best_E_D = best_E_D_for_G
        
        logging.info(f"Nash Equilibrium: E_D={best_E_D}, E_G={best_E_G}, Payoff_D={best_payoff_D:.4f}, Payoff_G={best_payoff_G:.4f}")
        return best_E_D, best_E_G

class RLAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Add game theory optimizer
        self.game_optimizer = GameTheoryOptimizer()
    
    def act(self, state):
        if random.random() < self.epsilon:
            action = random.sample(range(1, self.action_size + 1), k=3)
            logging.info(f"Random action selected: {action}")
            return action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        _, indices = torch.topk(q_values, k=3)
        action = [idx.item() + 1 for idx in indices[0]]
        logging.info(f"DQN action selected: {action}")
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            logging.info("Not enough memory for replay")
            return
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([t[0] for t in minibatch]).to(self.device)
        actions = torch.LongTensor([t[1][0] - 1 for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in minibatch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        q_values = self.model(states)
        next_q_values = self.model(next_states)
        
        targets = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        loss = self.criterion(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        logging.info(f"Replay completed, loss: {loss.item():.4f}, epsilon: {self.epsilon:.4f}")
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info(f"Saved model to {path}")
    
    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            logging.info(f"Loaded model from {path}")
        else:
            logging.warning(f"Model file not found at {path}")
    
    def optimize_epochs(self):
        return self.game_optimizer.find_nash_equilibrium()

def normalize(x, min_x, max_x):
    return (x - min_x) / (max_x - min_x + 1e-7)

def compute_reward(metrics, prev_metrics):
    w1, w2, w3, w4, w5 = 0.2, 0.2, 0.2, 0.2, 0.2
    loss_improvement = 0.0
    if prev_metrics and prev_metrics.get('proxy_loss', 0) > 0:
        loss_improvement = (prev_metrics['proxy_loss'] - metrics.get('proxy_loss', 0)) / prev_metrics['proxy_loss']
    
    latency = normalize(metrics.get('latency', 0), 0, 60)
    network_traffic = normalize(metrics.get('network_traffic', 0), 0, 1e6)
    cpu_usage = normalize(metrics.get('cpu_usage', 0), 0, 10)
    memory_usage = normalize(metrics.get('memory_usage', 0), 0, 1000)
    
    reward = w1 * loss_improvement - w2 * latency - w3 * network_traffic - w4 * cpu_usage - w5 * memory_usage
    logging.info(f"Computed reward: {reward:.4f}, loss_improvement: {loss_improvement:.4f}")
    return reward

def main():
    logging.info("Starting RL agent")
    base_dir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results"
    num_generators = 4
    num_disc_per_gen = 5
    state_size = 7
    action_size = num_disc_per_gen
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    agents = [RLAgent(state_size, action_size, device) for _ in range(num_generators)]
    
    # Optimize local epochs using game theory
    E_D, E_G = agents[0].optimize_epochs()
    
    # Save optimized epochs for use by generator and discriminator scripts
    for gen_id in range(1, num_generators + 1):
        gen_dir = f"{base_dir}/generator_{gen_id}"
        os.makedirs(gen_dir, exist_ok=True)
        with open(f"{gen_dir}/training_config.json", "w") as f:
            json.dump({"E_D": E_D, "E_G": E_G}, f, indent=2)
        logging.info(f"Saved training config for generator {gen_id}: E_D={E_D}, E_G={E_G}")
    
    for gen_id in range(1, num_generators + 1):
        gen_dir = f"{base_dir}/generator_{gen_id}"
        model_path = f"{gen_dir}/rl_model.pth"
        agents[gen_id-1].load(model_path)
        
        # Collect state
        state = []
        prev_metrics = {}
        metrics_path = f"{gen_dir}/metrics.json"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    prev_metrics = json.load(f)
            except Exception as e:
                logging.error(f"Error reading metrics for generator {gen_id}: {e}")
        
        # Gather discriminator metrics
        disc_metrics = []
        base_disc_id = (gen_id - 1) * num_disc_per_gen
        for local_disc_id in range(1, num_disc_per_gen + 1):
            global_disc_id = base_disc_id + local_disc_id
            disc_dir = f"{base_dir}/discriminator_{global_disc_id}"
            disc_metrics_path = f"{disc_dir}/metrics.json"
            if os.path.exists(disc_metrics_path):
                try:
                    with open(disc_metrics_path, "r") as f:
                        disc_metrics.append(json.load(f))
                except Exception as e:
                    logging.error(f"Error reading metrics for discriminator {global_disc_id}: {e}")
        
        # State: [avg_loss, avg_mean_score, latency, cpu, memory, network_traffic, avg_disc_score]
        avg_loss = np.mean([m.get('loss', 0) for m in disc_metrics]) if disc_metrics else 0
        avg_mean_score = np.mean([m.get('mean_score', 0) for m in disc_metrics]) if disc_metrics else 0
        latency = prev_metrics.get('latency', 0)
        cpu_usage = prev_metrics.get('cpu_usage', 0)
        memory_usage = prev_metrics.get('memory_usage', 0)
        network_traffic = prev_metrics.get('network_traffic', 0)
        avg_disc_score = prev_metrics.get('avg_disc_score', 0)
        
        state = [avg_loss, avg_mean_score, latency, cpu_usage, memory_usage, network_traffic, avg_disc_score]
        logging.info(f"State for generator {gen_id}: {state}")
        
        # Select action
        action = agents[gen_id-1].act(state)
        selected_discriminators = action
        
        # Save recommendations
        recommendation_path = f"{gen_dir}/rl_recommendations.json"
        try:
            with open(recommendation_path, "w") as f:
                json.dump({"selected_discriminators": selected_discriminators}, f, indent=2)
            logging.info(f"Saved recommendations for generator {gen_id}: {selected_discriminators}")
        except Exception as e:
            logging.error(f"Error saving recommendations for generator {gen_id}: {e}")
        
        # Compute reward and update model
        metrics = prev_metrics or {}
        reward = compute_reward(metrics, prev_metrics)
        next_state = state
        agents[gen_id-1].remember(state, action, reward, next_state, False)
        agents[gen_id-1].replay()
        agents[gen_id-1].save(model_path)
        
        print(f"Generator {gen_id}: Selected discriminators {selected_discriminators}")
    logging.info("RL agent completed")

if __name__ == "__main__":
    main()