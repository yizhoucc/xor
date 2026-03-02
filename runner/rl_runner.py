"""RL experiment runner for DQN with InnerNet activation.

Supports multi-seed training with replay buffer, epsilon-greedy, and target network.
"""
import os
import random
import logging
import pickle
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger('exp')


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def pretrain_inner_net(inner_net, device, num_steps=500, lr=1e-2):
    """Pretrain InnerNet on Gaussian target function."""
    optimizer = optim.Adam(inner_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    xv, yv = np.meshgrid(x, y)
    inputs = torch.tensor(
        np.vstack([xv.reshape(-1), yv.reshape(-1)]).T,
        dtype=torch.float32
    ).to(device)
    targets = torch.exp(-(inputs[:, 0]**2 + inputs[:, 1]**2)).view(-1, 1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = criterion(inner_net(inputs), targets)
        loss.backward()
        optimizer.step()

    return inner_net.state_dict()


class RLRunner:
    """Runner for DQN reinforcement learning experiments."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        self.save_dir = config.save_dir

        # RL hyperparameters
        rl = config.rl
        self.env_name = rl.env_name
        self.gamma = rl.get('gamma', 0.99)
        self.batch_size = rl.get('batch_size', 64)
        self.lr = rl.get('lr', 1e-3)
        self.epsilon_start = rl.get('epsilon_start', 1.0)
        self.epsilon_end = rl.get('epsilon_end', 0.01)
        self.epsilon_decay = rl.get('epsilon_decay', 0.996)
        self.memory_size = rl.get('memory_size', 100000)
        self.target_update = rl.get('target_update', 10)
        self.num_episodes = rl.get('num_episodes', 1000)
        self.num_seeds = rl.get('num_seeds', 10)
        self.log_interval = rl.get('log_interval', 50)

        # Model config
        self.model_name = config.model.name
        self.is_innernet = self.model_name == 'InnerNetDQN'

    def _make_env(self):
        import gymnasium as gym
        return gym.make(self.env_name)

    def _make_model(self, state_dim, action_dim):
        from model.dqn import InnerNetDQN, BaselineDQN
        if self.is_innernet:
            inner_hidden = self.config.model.get('inner_hidden', 32)
            return InnerNetDQN(state_dim, action_dim, inner_hidden)
        else:
            return BaselineDQN(state_dim, action_dim)

    def train(self):
        """Train DQN across multiple seeds."""
        env = self._make_env()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()

        # Pretrain InnerNet once (shared across seeds)
        gaussian_weights = None
        if self.is_innernet:
            logger.info("Pretraining InnerNet on Gaussian target...")
            from model.dqn import InnerNetDQNActivation
            temp_inner = InnerNetDQNActivation(
                hidden_dim=self.config.model.get('inner_hidden', 32)
            ).to(self.device)
            gaussian_weights = pretrain_inner_net(temp_inner, self.device)
            logger.info("InnerNet pretrained.")

        seeds = list(range(42, 42 + self.num_seeds))
        all_scores = []

        for seed in seeds:
            logger.info(f"[Seed {seed}] Training {self.model_name}...")
            scores = self._train_single_seed(seed, state_dim, action_dim, gaussian_weights)
            all_scores.append(scores)
            logger.info(f"[Seed {seed}] Final avg(last 50): {np.mean(scores[-50:]):.2f}")

        # Save results
        results = {
            'model_name': self.model_name,
            'env_name': self.env_name,
            'seeds': seeds,
            'all_scores': all_scores,
            'mean_scores': np.mean(all_scores, axis=0).tolist(),
            'std_scores': np.std(all_scores, axis=0).tolist(),
        }
        results_path = os.path.join(self.save_dir, 'rl_results.p')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        self._mark_stage('COMPLETED')
        logger.info(f"All seeds done. Results saved to {results_path}")

    def _train_single_seed(self, seed, state_dim, action_dim, gaussian_weights):
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        env = self._make_env()
        policy_net = self._make_model(state_dim, action_dim).to(self.device)
        target_net = self._make_model(state_dim, action_dim).to(self.device)

        # Load pretrained InnerNet weights
        if self.is_innernet and gaussian_weights is not None:
            policy_net.inner_net.load_state_dict(gaussian_weights)
            target_net.inner_net.load_state_dict(gaussian_weights)

        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=self.lr)
        memory = ReplayBuffer(self.memory_size)
        epsilon = self.epsilon_start
        scores = []
        criterion = nn.MSELoss()

        for i_episode in range(self.num_episodes):
            state, _ = env.reset(seed=seed + i_episode)
            score = 0
            done = False

            while not done:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        q_values = policy_net(state_t)
                        action = q_values.argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                memory.push(state, action, reward, next_state, done)
                state = next_state
                score += reward

                # Train on batch
                if len(memory) > self.batch_size:
                    s, a, r, ns, d = memory.sample(self.batch_size)
                    s_b = torch.FloatTensor(np.array(s)).to(self.device)
                    a_b = torch.LongTensor(a).to(self.device).unsqueeze(1)
                    r_b = torch.FloatTensor(r).to(self.device).unsqueeze(1)
                    ns_b = torch.FloatTensor(np.array(ns)).to(self.device)
                    d_b = torch.FloatTensor(d).to(self.device).unsqueeze(1)

                    q_eval = policy_net(s_b).gather(1, a_b)
                    with torch.no_grad():
                        q_next = target_net(ns_b).max(1)[0].unsqueeze(1)
                        q_target = r_b + (self.gamma * q_next * (1 - d_b))

                    loss = criterion(q_eval, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
            if i_episode % self.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            scores.append(score)

            if (i_episode + 1) % self.log_interval == 0:
                avg_score = np.mean(scores[-self.log_interval:])
                logger.info(f"  Ep {i_episode+1}/{self.num_episodes}: "
                          f"Avg Score = {avg_score:.2f} (eps: {epsilon:.3f})")

        env.close()
        return scores

    def test(self):
        """Test is not applicable for RL in the same way. Results are saved during train."""
        logger.info("RL test: results were saved during training.")

    def _mark_stage(self, stage_name):
        marker = os.path.join(self.save_dir, stage_name)
        with open(marker, 'w') as f:
            f.write('')
