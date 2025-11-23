#!/usr/bin/env python3
"""
Reinforcement Learning Agent - Self-Contained Module
DQN/PPO agent for learning optimal engine weight combinations.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pickle
import random
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DQNNetwork(nn.Module if TORCH_AVAILABLE else object):
    """Deep Q-Network for learning engine weights with GPU support."""
    
    def __init__(self, state_size: int = 50, action_size: int = 3, use_gpu: bool = True):
        if TORCH_AVAILABLE:
            super().__init__()
            self.fc1 = nn.Linear(state_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, action_size)
            self.relu = nn.ReLU()
            self.use_gpu = use_gpu and CUDA_AVAILABLE
            if self.use_gpu:
                self.to(DEVICE)
        else:
            self.weights = np.random.randn(state_size, action_size) * 0.01
            self.use_gpu = False
    
    def forward(self, x):
        """Forward pass."""
        if not TORCH_AVAILABLE:
            return np.dot(x, self.weights)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
    def predict(self, state):
        """Predict action values with GPU support."""
        if not TORCH_AVAILABLE:
            return np.dot(state, self.weights)
        
        self.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if self.use_gpu:
                state = state.to(DEVICE)
            q_values = self.forward(state)
            if self.use_gpu:
                q_values = q_values.cpu()
            return q_values.numpy()


class RLAgent:
    """Reinforcement Learning Agent for engine weight optimization with GPU support."""
    
    def __init__(self, state_size: int = 50, action_size: int = 3, learning_rate: float = 0.001, use_gpu: bool = True):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95  # Discount factor
        self.use_gpu = use_gpu
        
        if TORCH_AVAILABLE:
            self.q_network = DQNNetwork(state_size, action_size, use_gpu=use_gpu)
            self.target_network = DQNNetwork(state_size, action_size, use_gpu=use_gpu)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.update_target_network()
        else:
            self.q_network = DQNNetwork(state_size, action_size, use_gpu=False)
            self.target_network = None
    
    def update_target_network(self):
        """Update target network with Q-network weights."""
        if TORCH_AVAILABLE:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training: bool = True):
        """Choose action using epsilon-greedy policy."""
        if training and random.random() <= self.epsilon:
            # Random action (exploration)
            return random.randrange(self.action_size)
        
        # Greedy action (exploitation)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values)
    
    def get_engine_weights(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Get optimal engine weights for given state. Returns (statistical, technical, ml) weights."""
        action = self.act(state, training=False)
        
        # Map action to weight combination
        # Action 0: Favor statistical (0.6, 0.2, 0.2)
        # Action 1: Favor technical (0.2, 0.6, 0.2)
        # Action 2: Favor ML (0.2, 0.2, 0.6)
        # Action 3+: Balanced combinations
        
        weight_combinations = [
            (0.6, 0.2, 0.2),  # Statistical
            (0.2, 0.6, 0.2),  # Technical
            (0.2, 0.2, 0.6),  # ML
            (0.4, 0.3, 0.3),  # Balanced 1
            (0.33, 0.33, 0.34),  # Balanced 2
        ]
        
        if action < len(weight_combinations):
            return weight_combinations[action]
        else:
            return (0.33, 0.33, 0.34)  # Default balanced
    
    def replay(self, batch_size: int = 32):
        """Train on a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        if not TORCH_AVAILABLE:
            # Simple Q-learning update
            for state, action, reward, next_state, done in batch:
                q_values = self.q_network.predict(state)
                if done:
                    q_values[action] = reward
                else:
                    next_q_values = self.q_network.predict(next_state)
                    q_values[action] = reward + self.gamma * np.max(next_q_values)
                # Update weights (simplified)
                self.q_network.weights += self.learning_rate * (q_values - self.q_network.predict(state))
            return
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        # Move to GPU if available
        if self.use_gpu and CUDA_AVAILABLE:
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            rewards = rewards.to(DEVICE)
            next_states = next_states.to(DEVICE)
            dones = dones.to(DEVICE)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, prediction_accuracy: float, sharpe_ratio: float = 0.0) -> float:
        """Calculate reward based on prediction accuracy and risk metrics."""
        # Reward is primarily based on accuracy
        base_reward = prediction_accuracy * 10  # Scale to 0-10
        
        # Bonus for good risk-adjusted returns
        if sharpe_ratio > 1.0:
            base_reward += 2.0
        elif sharpe_ratio > 0.5:
            base_reward += 1.0
        
        return base_reward
    
    def save(self, filepath: str):
        """Save agent to file."""
        try:
            if TORCH_AVAILABLE:
                torch.save({
                    'q_network': self.q_network.state_dict(),
                    'epsilon': self.epsilon,
                    'memory': list(self.memory)
                }, filepath)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'weights': self.q_network.weights,
                        'epsilon': self.epsilon,
                        'memory': list(self.memory)
                    }, f)
        except Exception:
            pass
    
    def load(self, filepath: str):
        """Load agent from file."""
        try:
            if TORCH_AVAILABLE and os.path.exists(filepath):
                checkpoint = torch.load(filepath)
                self.q_network.load_state_dict(checkpoint['q_network'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.memory = deque(checkpoint.get('memory', []), maxlen=10000)
                self.update_target_network()
            elif os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.q_network.weights = data['weights']
                    self.epsilon = data.get('epsilon', self.epsilon)
                    self.memory = deque(data.get('memory', []), maxlen=10000)
        except Exception:
            pass


# Global RL agent instance
_rl_agent = None


def get_rl_agent(state_size: int = 50, action_size: int = 3, use_gpu: bool = True) -> RLAgent:
    """Get or create global RL agent instance with GPU support."""
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = RLAgent(state_size, action_size, use_gpu=use_gpu)
        # Try to load saved agent
        agent_path = os.path.join(os.path.dirname(__file__), "..", "memory", "rl_agent.pkl")
        _rl_agent.load(agent_path)
    return _rl_agent
