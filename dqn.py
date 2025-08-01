import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  # Larger network
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularization
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, num_actions, device, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=100_000, batch_size=64, target_update_freq=1000):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = device

        self.model = DQN(state_dim, num_actions).to(device)
        self.target_model = DQN(state_dim, num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Adam instead of NAdam
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for more stable training

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.train_steps = 0
        self.target_update_freq = target_update_freq

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(state_tensor).argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, float(done)))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.model(states).gather(1, actions).squeeze(1)
        
        # Double DQN: use main network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Tighter gradient clipping
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Slower epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle both old format (direct state_dict) and new format (dict with keys)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format
            model_state_dict = checkpoint['model_state_dict']
            
            # Load optimizer state and training progress if available
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("Could not load optimizer state - continuing with fresh optimizer")
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            if 'train_steps' in checkpoint:
                self.train_steps = checkpoint['train_steps']
        else:
            # Old format (direct state_dict)
            model_state_dict = checkpoint
            print("Loaded model in old format - training progress not restored")
        
        # Try to load the state dict, handle architecture mismatches
        try:
            self.model.load_state_dict(model_state_dict)
            self.target_model.load_state_dict(self.model.state_dict())
            print("Model weights loaded successfully")
        except RuntimeError as e:
            print(f"Architecture mismatch detected: {e}")
            print("Starting with fresh weights due to network architecture change")
            # Reset training progress since we can't use old weights
            self.epsilon = 1.0
            self.train_steps = 0

    def save_replay_buffer(self, path):
        """Save replay buffer to file"""
        with open(path, 'wb') as f:
            pickle.dump(list(self.replay_buffer), f)

    def load_replay_buffer(self, path):
        """Load replay buffer from file"""
        try:
            with open(path, 'rb') as f:
                buffer_data = pickle.load(f)
                self.replay_buffer = deque(buffer_data, maxlen=self.replay_buffer.maxlen)
                print(f"Loaded replay buffer with {len(self.replay_buffer)} experiences")
        except Exception as e:
            print(f"Could not load replay buffer: {e}")
            self.replay_buffer = deque(maxlen=self.replay_buffer.maxlen)