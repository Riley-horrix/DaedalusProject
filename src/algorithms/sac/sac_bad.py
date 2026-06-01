import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os

# -----------------------------------------
# Neural Network Architectures
# -----------------------------------------

class LayerNormHiddenBlock(nn.Module):
    """
    Implements the hidden unit topology from the paper:
    Linear combination -> Layer Normalization -> ReLU.
    """
    def __init__(self, in_dim, out_dim):
        super(LayerNormHiddenBlock, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.layer_norm(x)
        return F.relu(x)

class DoubleCritic(nn.Module):
    """
    Double Q-function critic network.
    """
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super(DoubleCritic, self).__init__()

        # Q1 Architecture
        self.q1_l1 = LayerNormHiddenBlock(obs_dim + action_dim, hidden_sizes[0])
        self.q1_l2 = LayerNormHiddenBlock(hidden_sizes[0], hidden_sizes[1])
        self.q1_out = nn.Linear(hidden_sizes[1], 1)

        # Q2 Architecture
        self.q2_l1 = LayerNormHiddenBlock(obs_dim + action_dim, hidden_sizes[0])
        self.q2_l2 = LayerNormHiddenBlock(hidden_sizes[0], hidden_sizes[1])
        self.q2_out = nn.Linear(hidden_sizes[1], 1)

        # Xavier Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)

        q1 = self.q1_l1(sa)
        q1 = self.q1_l2(q1)
        q1 = self.q1_out(q1)

        q2 = self.q2_l1(sa)
        q2 = self.q2_l2(q2)
        q2 = self.q2_out(q2)
        return q1, q2

class Actor(nn.Module):
    """
    Stochastic policy network outputting mean and log standard deviation.
    """
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super(Actor, self).__init__()

        self.l1 = LayerNormHiddenBlock(obs_dim, hidden_sizes[0])
        self.l2 = LayerNormHiddenBlock(hidden_sizes[0], hidden_sizes[1])

        self.mean_out = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std_out = nn.Linear(hidden_sizes[1], action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = self.l1(state)
        x = self.l2(x)

        mean = self.mean_out(x)
        log_std = self.log_std_out(x)
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t

        # Enforce action bounds and compute log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean)

# -----------------------------------------
# SAC Agent Wrapper
# -----------------------------------------

class SACAgent:
    def __init__(self, obs_dim, action_dim, num_envs, device, config):
        self.device = device
        self.config = config
        self.action_dim = action_dim

        hidden_sizes = config('hidden_sizes', [64, 64])
        self.gamma = config('gamma', 0.99)
        self.tau = config('tau', 0.005)
        self.batch_size = config('batch_size', 256)

        # Initialize Networks
        self.actor = Actor(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.critic = DoubleCritic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.critic_target = DoubleCritic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        lr = config('learning_rate', 3e-4)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic Temperature (Entropy) Tuning (Paper uses \eta)
        self.target_entropy = config('target_entropy', -action_dim)
        self.log_eta = torch.zeros(1, requires_grad=True, device=self.device)
        self.eta_optimizer = optim.Adam([self.log_eta], lr=lr)

    def act(self, obs, add_noise=True):
        """
        Called by train.py. If add_noise is False (evaluation), uses the deterministic mean.
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action, _, deterministic_action = self.actor.sample(obs)
            if not add_noise:
                return deterministic_action
            return action

    def update(self, buffer, num_updates, log):
        """
        Performs gradient steps based on batches sampled from the replay buffer.
        """
        if buffer.size < self.batch_size:
            return

        actor_loss_acc = 0.0
        critic_loss_acc = 0.0
        eta_loss_acc = 0.0

        for _ in range(num_updates):
            state, action, reward, next_state, done = buffer.sample(self.batch_size)

            # 1. Update Critic
            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2) - self.log_eta.exp() * next_log_prob
                target_q = reward.unsqueeze(-1) + (1.0 - done.unsqueeze(-1).float()) * self.gamma * target_q

            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 2. Update Actor
            new_action, log_prob, _ = self.actor.sample(state)
            q1_new, q2_new = self.critic(state, new_action)
            q_new = torch.min(q1_new, q2_new)

            eta = self.log_eta.exp().detach()
            actor_loss = (eta * log_prob - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 3. Update Temperature (\eta)
            eta_loss = -(self.log_eta * (log_prob + self.target_entropy).detach()).mean()

            self.eta_optimizer.zero_grad()
            eta_loss.backward()
            self.eta_optimizer.step()

            # 4. Soft Update Target Networks
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            actor_loss_acc += actor_loss.item()
            critic_loss_acc += critic_loss.item()
            eta_loss_acc += eta_loss.item()

        # Update logging struct
        log.log['loss/actor'] = actor_loss_acc / num_updates
        log.log['loss/critic'] = critic_loss_acc / num_updates
        log.log['loss/eta'] = eta_loss_acc / num_updates
        log.log['train/eta'] = self.log_eta.exp().item()

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_eta': self.log_eta,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'eta_optimizer': self.eta_optimizer.state_dict()
        }, path)

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.log_eta = checkpoint['log_eta']
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.eta_optimizer.load_state_dict(checkpoint['eta_optimizer'])
        else:
            print(f"Warning: Checkpoint not found at {path}")