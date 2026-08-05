#!/usr/bin/env python3
# ddpg_agent.py - Deep Deterministic Policy Gradient for Hybrid GAN optimization
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import logging

# Setup logging
logging.basicConfig(filename='/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/ddpg_agent.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for action exploration"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = self.mu.copy()
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class Actor(nn.Module):
    """Actor network for DDPG"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # Use tanh to bound actions between -1 and 1, then scale
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    """Critic network for DDPG"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = torch.cat([xs, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    """DDPG Agent for continuous control"""
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.001  # Soft update parameter
        self.batch_size = 128
        self.buffer_size = 100000
        self.learning_rate_actor = 1e-4
        self.learning_rate_critic = 1e-3
        
        # Networks
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic)
        
        # Noise process
        self.noise = OrnsteinUhlenbeckNoise(action_size)
        
        # Replay buffer
        self.memory = ReplayBuffer(self.buffer_size)
        
        # Initialize target networks
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        
        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
        
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
        
        # Scale actions to appropriate ranges
        action = self.scale_action(action[0])
        return action
    
    def scale_action(self, action):
        """Scale normalized action to actual parameter ranges"""
        # Action vector: [batch_size_scale, gen_epochs_scale, disc_epochs_scale, learning_rate_scale]
        scaled_action = np.zeros(4)
        
        # Batch size: 32 to 128
        scaled_action[0] = 32 + (action[0] + 1) * 48  # Scale from [-1,1] to [32, 128]
        
        # Generator epochs: 1 to 10
        scaled_action[1] = 1 + (action[1] + 1) * 4.5  # Scale from [-1,1] to [1, 10]
        
        # Discriminator epochs: 1 to 15
        scaled_action[2] = 1 + (action[2] + 1) * 7  # Scale from [-1,1] to [1, 15]
        
        # Learning rate scale: 0.0001 to 0.001
        scaled_action[3] = 0.0001 + (action[3] + 1) * 0.00045  # Scale from [-1,1] to [0.0001, 0.001]
        
        # Ensure integer values where needed
        scaled_action[0] = int(np.clip(scaled_action[0], 32, 128))
        scaled_action[1] = int(np.clip(scaled_action[1], 1, 10))
        scaled_action[2] = int(np.clip(scaled_action[2], 1, 15))
        scaled_action[3] = np.clip(scaled_action[3], 0.0001, 0.001)
        
        return scaled_action
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn"""
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update Critic
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # Update Actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        
        # Store losses for monitoring
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        logging.info(f"Training step - Actor loss: {actor_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, path):
        """Save agent models"""
        torch.save({
            'actor_local': self.actor_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }, path)
        logging.info(f"Saved DDPG agent to {path}")
    
    def load(self, path):
        """Load agent models"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.actor_local.load_state_dict(checkpoint['actor_local'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_local.load_state_dict(checkpoint['critic_local'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.actor_losses = checkpoint.get('actor_losses', [])
            self.critic_losses = checkpoint.get('critic_losses', [])
            logging.info(f"Loaded DDPG agent from {path}")
        else:
            logging.warning(f"DDPG model file not found at {path}")

class MultiObjectiveReward:
    """Multi-objective reward function for hybrid GAN optimization"""
    def __init__(self):
        self.weights = {
            'loss_improvement': 0.3,
            'score_difference': 0.2,
            'diversity': 0.15,
            'efficiency': 0.2,
            'stability': 0.15
        }
        
    def compute_reward(self, current_metrics, previous_metrics=None):
        """Compute multi-objective reward"""
        reward = 0.0
        
        # Loss improvement reward
        if previous_metrics:
            gen_loss_improvement = (previous_metrics.get('gen_loss', 1) - current_metrics.get('gen_loss', 1))
            disc_loss_improvement = (previous_metrics.get('disc_loss', 1) - current_metrics.get('disc_loss', 1))
            loss_improvement = (gen_loss_improvement + disc_loss_improvement) / 2
            reward += self.weights['loss_improvement'] * loss_improvement
        
        # Score difference reward (want real and fake scores to be close)
        score_diff = abs(current_metrics.get('score_difference', 1))
        score_reward = 1 / (1 + score_diff)  # Inverse relationship
        reward += self.weights['score_difference'] * score_reward
        
        # Diversity reward
        diversity = current_metrics.get('diversity_score', 0)
        diversity_reward = min(diversity / 10.0, 1.0)  # Normalize and cap
        reward += self.weights['diversity'] * diversity_reward
        
        # Efficiency reward (inverse of resource usage)
        latency = current_metrics.get('latency', 1)
        cpu_usage = current_metrics.get('cpu_usage', 1)
        memory_usage = current_metrics.get('memory_usage', 100)
        
        efficiency = 1 / (1 + latency + cpu_usage/10 + memory_usage/1000)
        reward += self.weights['efficiency'] * efficiency
        
        # Stability reward (based on balance factor)
        balance_factor = current_metrics.get('balance_factor', 1)
        stability = 1 / (1 + abs(balance_factor - 1))
        reward += self.weights['stability'] * stability
        
        logging.info(f"Reward components - Loss: {loss_improvement if previous_metrics else 0:.4f}, "
                    f"Score: {score_reward:.4f}, Diversity: {diversity_reward:.4f}, "
                    f"Efficiency: {efficiency:.4f}, Stability: {stability:.4f}, Total: {reward:.4f}")
        
        return reward

def normalize_state(state):
    """Normalize state values to [0, 1] range"""
    normalized = np.zeros_like(state)
    
    # Define normalization ranges for each state component
    ranges = {
        0: (0, 10),      # gen_loss
        1: (0, 10),      # disc_loss  
        2: (-5, 5),      # score_difference
        3: (0, 20),      # diversity_score
        4: (0, 120),     # latency
        5: (0, 20),      # cpu_usage
        6: (0, 2000),    # memory_usage
        7: (0.5, 2.0)    # balance_factor
    }
    
    for i, (min_val, max_val) in ranges.items():
        if i < len(state):
            normalized[i] = np.clip((state[i] - min_val) / (max_val - min_val), 0, 1)
    
    return normalized

def main():
    logging.info("Starting DDPG agent")
    base_dir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results"
    num_hybrid_nodes = 20
    state_size = 8  # [gen_loss, disc_loss, score_diff, diversity, latency, cpu, memory, balance_factor]
    action_size = 4  # [batch_size, gen_epochs, disc_epochs, learning_rate]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize DDPG agent
    agent = DDPGAgent(state_size, action_size, device)
    reward_function = MultiObjectiveReward()
    
    # Load existing model if available
    global_model_path = f"{base_dir}/ddpg_global_model.pth"
    agent.load(global_model_path)
    
    # Process each hybrid node
    total_reward = 0
    nodes_processed = 0
    
    for node_id in range(1, num_hybrid_nodes + 1):
        node_dir = f"{base_dir}/hybrid_node_{node_id}"
        
        if not os.path.exists(node_dir):
            continue
            
        # Load current state
        state_path = f"{node_dir}/ddpg_state.json"
        if not os.path.exists(state_path):
            continue
            
        try:
            with open(state_path, "r") as f:
                current_metrics = json.load(f)
                
            # Load previous metrics for comparison
            history_path = f"{node_dir}/metrics_history.json"
            previous_metrics = None
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history = json.load(f)
                    if len(history) >= 2:
                        previous_metrics = history[-2]  # Second to last entry
            
            # Construct state vector
            state = np.array([
                current_metrics.get('gen_loss', 0),
                current_metrics.get('disc_loss', 0),
                current_metrics.get('score_difference', 0),
                current_metrics.get('diversity_score', 0),
                current_metrics.get('latency', 0),
                current_metrics.get('cpu_usage', 0),
                current_metrics.get('memory_usage', 0),
                current_metrics.get('balance_factor', 1.0)
            ])
            
            # Normalize state
            normalized_state = normalize_state(state)
            
            # Get action from agent
            action = agent.act(normalized_state, add_noise=True)
            
            # Compute reward
            reward = reward_function.compute_reward(current_metrics, previous_metrics)
            total_reward += reward
            
            # Store experience if we have previous data
            if previous_metrics is not None:
                prev_state = np.array([
                    previous_metrics.get('gen_loss', 0),
                    previous_metrics.get('disc_loss', 0),
                    previous_metrics.get('score_difference', 0),
                    previous_metrics.get('diversity_score', 0),
                    previous_metrics.get('latency', 0),
                    previous_metrics.get('cpu_usage', 0),
                    previous_metrics.get('memory_usage', 0),
                    previous_metrics.get('balance_factor', 1.0)
                ])
                prev_normalized_state = normalize_state(prev_state)
                
                # Assume previous action is stored or use current action as approximation
                prev_action = action  # Simplified for now
                
                agent.step(prev_normalized_state, prev_action, reward, normalized_state, False)
            
            # Save configuration for hybrid node
            config = {
                "batch_size": int(action[0]),
                "gen_epochs": int(action[1]),
                "disc_epochs": int(action[2]),
                "learning_rate": float(action[3])
            }
            
            config_path = f"{node_dir}/ddpg_config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            logging.info(f"Node {node_id}: State={normalized_state[:4]}, Action={action}, Reward={reward:.4f}")
            print(f"Node {node_id}: Configured with batch_size={config['batch_size']}, "
                  f"gen_epochs={config['gen_epochs']}, disc_epochs={config['disc_epochs']}, "
                  f"lr={config['learning_rate']:.6f}, reward={reward:.4f}")
            
            nodes_processed += 1
            
        except Exception as e:
            logging.error(f"Error processing node {node_id}: {e}")
            continue
    
    # Save updated global model
    if nodes_processed > 0:
        agent.save(global_model_path)
        avg_reward = total_reward / nodes_processed
        
        # Save global statistics
        global_stats = {
            "total_nodes_processed": nodes_processed,
            "average_reward": avg_reward,
            "total_reward": total_reward,
            "actor_loss_history": agent.actor_losses[-100:],  # Last 100 entries
            "critic_loss_history": agent.critic_losses[-100:]
        }
        
        stats_path = f"{base_dir}/ddpg_global_stats.json"
        with open(stats_path, "w") as f:
            json.dump(global_stats, f, indent=2)
        
        logging.info(f"DDPG agent completed. Processed {nodes_processed} nodes with average reward {avg_reward:.4f}")
        print(f"DDPG optimization completed for {nodes_processed} hybrid nodes. Average reward: {avg_reward:.4f}")
    else:
        logging.warning("No nodes were processed by DDPG agent")
        print("No hybrid nodes found for DDPG optimization")

if __name__ == "__main__":
    main()