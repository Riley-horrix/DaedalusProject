import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.configs.config import Config
from src.utils.logging import LoggingStruct
from src.algorithms.base_agent import BaseAgent
from src.utils.replay_buffer import ReplayBuffer

class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        q1_value = self.q1(x)
        q2_value = self.q2(x)
        return q1_value, q2_value


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Squash output to [-1, 1] for control surfaces
        )

        self.apply(self._orthogonal_init)

    def _orthogonal_init(self, module):
        if isinstance(module, nn.Linear):
            # For the final layer, initialize weights with a smaller gain to keep initial actions near zero
            if module == self.net[-2]:
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
            else:
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting a deterministic action."""
        return self.net(obs)


class TD3Agent(BaseAgent):
    def __init__(self, obs_dim: int, action_dim: int, num_envs: int, device: torch.device, config: Config):
        super().__init__(obs_dim, action_dim, num_envs, device, config)

        self.gamma = config('gamma')
        self.tau = config('tau')
        self.batch_size = config('batch_size')

        # TD3 Specific Hyperparameters
        self.exploration_noise = config('exploration_noise', 0.1)
        self.target_noise = config('target_noise', 0.2)
        self.noise_clip = config('noise_clip', 0.5)
        self.policy_delay = config('policy_delay', 2)

        self.total_it = 0 # Step counter for delayed updates

        # Actor and Target Actor
        self.actor = DeterministicActor(obs_dim, action_dim, config('actor_hidden_dim')).to(device)
        self.actor_target = DeterministicActor(obs_dim, action_dim, config('actor_hidden_dim')).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic and Target Critic
        self.critic = DoubleQCritic(obs_dim, action_dim, config('critic_hidden_dim')).to(device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim, config('critic_hidden_dim')).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(),
            lr=config('actor_learning_rate'),
            weight_decay=config('actor_weight_decay')
        )

        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(),
            lr=config('critic_learning_rate'),
            weight_decay=config('critic_weight_decay')
        )

    def act(self, obs: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """Select actions using the deterministic policy, with optional exploration noise."""
        with torch.no_grad():
            action = self.actor(obs)
            if add_noise:
                # Add Gaussian exploration noise
                noise = torch.randn_like(action) * self.exploration_noise
                action = (action + noise).clamp(-1.0, 1.0)
        return action

    def update(self, replay_buffer: ReplayBuffer, epochs: int, log: LoggingStruct):
        for epoch in range(epochs):
            self.total_it += 1

            # Sample a batch from the replay buffer
            obs, actions, rewards, next_obs, dones = replay_buffer.sample(self.batch_size)

            obs = obs[..., :self.obs_dim].view(-1, self.obs_dim)
            actions = actions[..., :self.action_dim].view(-1, self.action_dim)
            next_obs = next_obs[..., :self.obs_dim].view(-1, self.obs_dim)

            rewards = rewards.view(-1, 1)
            dones = dones.view(-1, 1).float()

            # ---------------- Critic Update ---------------- #
            with torch.no_grad():
                # Target Policy Smoothing: Add clipped noise to the target action
                noise = (torch.randn_like(actions) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
                next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)

                # Get target Q values
                target_q1, target_q2 = self.critic_target(next_obs, next_actions)
                min_target_q = torch.min(target_q1, target_q2)

                # Bellman backup
                target_value = rewards + (1.0 - dones) * self.gamma * min_target_q

            # Get current Q values
            current_q1, current_q2 = self.critic(obs, actions)

            # Compute critic loss
            critic_1_loss = F.mse_loss(current_q1, target_value)
            critic_2_loss = F.mse_loss(current_q2, target_value)
            critic_loss = critic_1_loss + critic_2_loss

            # Optimize Critic
            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optim.step()

            # ---------------- Delayed Actor Update ---------------- #
            actor_loss = torch.tensor(0.0, device=self.device) # Fallback for logging if not updated

            if self.total_it % self.policy_delay == 0:
                # Actor loss is derived from the first critic's output
                q1_new, _ = self.critic(obs, self.actor(obs))
                actor_loss = -q1_new.mean()

                # Optimize Actor
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optim.step()

                # Target network soft updates (Polyak Averaging)
                with torch.no_grad():
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            # Log metrics on the final epoch cycle
            if epoch == epochs - 1:
                log_dict = {
                    "loss/critic_1": critic_1_loss.item(),
                    "loss/critic_2": critic_2_loss.item(),
                }
                # Only log actor loss if it isn't the dummy zero tensor
                if actor_loss.item() != 0.0:
                    log_dict["loss/actor"] = actor_loss.item()

                log.update(log_dict)

    def save(self, path: str):
        data = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'optimizer_state_dict': {
                'actor': self.actor_optim.state_dict(),
                'critic': self.critic_optim.state_dict(),
            }
        }
        torch.save(data, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data['actor_state_dict'])
        self.actor_target.load_state_dict(data['actor_target_state_dict'])
        self.critic.load_state_dict(data['critic_state_dict'])
        self.critic_target.load_state_dict(data['critic_target_state_dict'])
        self.actor_optim.load_state_dict(data['optimizer_state_dict']['actor'])
        self.critic_optim.load_state_dict(data['optimizer_state_dict']['critic'])