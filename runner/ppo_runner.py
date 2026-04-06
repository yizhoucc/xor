"""PPO experiment runner with InnerNet/ReLU/SwiGLU activation comparison.

Implements standard PPO with GAE, clipped surrogate objective, value function loss.
"""
import os
import random
import logging
import pickle

import numpy as np
import torch
import torch.optim as optim

logger = logging.getLogger('exp_logger')


class PPORunner:
    """Runner for PPO reinforcement learning experiments."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        self.save_dir = config.save_dir

        rl = config.rl
        self.env_name = rl.env_name
        self.gamma = rl.get('gamma', 0.99)
        self.gae_lambda = rl.get('gae_lambda', 0.95)
        self.clip_eps = rl.get('clip_eps', 0.2)
        self.lr = rl.get('lr', 3e-4)
        self.num_epochs = rl.get('ppo_epochs', 4)
        self.batch_size = rl.get('batch_size', 64)
        self.num_steps = rl.get('num_steps', 2048)  # steps per rollout
        self.num_updates = rl.get('num_updates', 200)  # total PPO updates
        self.num_seeds = rl.get('num_seeds', 10)
        self.vf_coef = rl.get('vf_coef', 0.5)
        self.ent_coef = rl.get('ent_coef', 0.01)
        self.max_grad_norm = rl.get('max_grad_norm', 0.5)
        self.log_interval = rl.get('log_interval', 10)

        self.model_name = config.model.name

    def _make_env(self):
        import gymnasium as gym
        return gym.make(self.env_name)

    def _make_model(self, state_dim, action_dim):
        from model.ppo import InnerNetPPO, BaselinePPO, SwiGLUPPO
        hidden_dim = self.config.model.get('hidden_dim', 128 if self.model_name != 'BaselinePPO' else 64)
        if self.model_name == 'InnerNetPPO':
            inner_hidden = self.config.model.get('inner_hidden', 32)
            return InnerNetPPO(state_dim, action_dim, hidden_dim=hidden_dim, inner_hidden=inner_hidden)
        elif self.model_name == 'SwiGLUPPO':
            return SwiGLUPPO(state_dim, action_dim, hidden_dim=hidden_dim)
        else:
            return BaselinePPO(state_dim, action_dim, hidden_dim=hidden_dim)

    def train(self):
        """Train PPO across multiple seeds."""
        logger.info(f"PPO on {self.env_name} with {self.model_name}")
        env = self._make_env()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()

        seeds = list(range(42, 42 + self.num_seeds))
        all_scores = []

        for si, seed in enumerate(seeds):
            logger.info(f"[Seed {seed}] ({si+1}/{len(seeds)})")
            scores = self._train_single_seed(seed, state_dim, action_dim)
            all_scores.append(scores)
            logger.info(f"[Seed {seed}] Done. Final avg(last 20): {np.mean(scores[-20:]):.2f}")

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
        logger.info(f"Results saved to {results_path}")

    def _train_single_seed(self, seed, state_dim, action_dim):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        env = self._make_env()
        model = self._make_model(state_dim, action_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, eps=1e-5)

        eval_scores = []

        for update in range(self.num_updates):
            # Collect rollout
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
            state, _ = env.reset(seed=seed + update)

            for step in range(self.num_steps):
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, log_prob, value = model.act(state_t)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                states.append(state)
                actions.append(action.item())
                log_probs.append(log_prob.item())
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())

                state = next_state
                if done:
                    state, _ = env.reset()

            # Compute GAE advantages
            with torch.no_grad():
                next_value = model._critic_forward(
                    torch.FloatTensor(state).unsqueeze(0).to(self.device)
                ).item() if hasattr(model, '_critic_forward') else \
                    model.forward(torch.FloatTensor(state).unsqueeze(0).to(self.device))[1].item()

            advantages = np.zeros(self.num_steps)
            last_gae = 0
            for t in reversed(range(self.num_steps)):
                next_val = next_value if t == self.num_steps - 1 else values[t + 1]
                next_nonterminal = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_val * next_nonterminal - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae

            returns = advantages + np.array(values)

            # Convert to tensors
            b_states = torch.FloatTensor(np.array(states)).to(self.device)
            b_actions = torch.LongTensor(actions).to(self.device)
            b_log_probs = torch.FloatTensor(log_probs).to(self.device)
            b_advantages = torch.FloatTensor(advantages).to(self.device)
            b_returns = torch.FloatTensor(returns).to(self.device)

            # Normalize advantages
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # PPO update epochs
            indices = np.arange(self.num_steps)
            for _ in range(self.num_epochs):
                np.random.shuffle(indices)
                for start in range(0, self.num_steps, self.batch_size):
                    end = start + self.batch_size
                    mb_idx = indices[start:end]

                    new_log_probs, entropy, new_values = model.evaluate(
                        b_states[mb_idx], b_actions[mb_idx])

                    ratio = torch.exp(new_log_probs - b_log_probs[mb_idx])
                    surr1 = ratio * b_advantages[mb_idx]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages[mb_idx]

                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = ((new_values - b_returns[mb_idx]) ** 2).mean()
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()

            # Evaluate
            if (update + 1) % self.log_interval == 0 or update == 0:
                eval_score = self._evaluate(model, seed)
                eval_scores.append(eval_score)
                logger.info(f"  Update {update+1}/{self.num_updates}: eval={eval_score:.1f}")

        env.close()
        return eval_scores

    def _evaluate(self, model, seed, n_episodes=10):
        """Evaluate policy without exploration."""
        env = self._make_env()
        total = 0
        for ep in range(n_episodes):
            state, _ = env.reset(seed=seed * 1000 + ep)
            done = False
            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    probs = model._actor_forward(state_t) if hasattr(model, '_actor_forward') else \
                        model.forward(state_t)[0]
                    action = probs.argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += reward
        env.close()
        return total / n_episodes

    def _mark_stage(self, stage_name):
        marker = os.path.join(self.save_dir, stage_name)
        with open(marker, 'w') as f:
            f.write('')

    def test(self):
        logger.info("PPO test not implemented (eval is done during training)")
