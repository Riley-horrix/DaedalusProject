import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Q-network.

        Args:
            obs (torch.Tensor): Batch of observations with shape [batch_size, obs_dim].
            action (torch.Tensor): Batch of actions with shape [batch_size, action_dim].
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Q-values with shape [batch_size, 1].
        """
        x = torch.cat([obs, action], dim=-1)
        q1_value = self.q1(x)
        q2_value = self.q2(x)
        return q1_value, q2_value


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim=256):
        super().__init__()
        self.x1 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # Bounds needed for log_std to prevent numerical issues with very large or small standard deviations
        self.LOG_STD_MAX = 2.0
        self.LOG_STD_MIN = -20.0

    def _orthogonal_init(self, module):
        """Orthogonal initialization for linear layers, with special handling for the final action output layer to ensure initial actions are near zero.

        Args:
            module (nn.Linear): The linear layer to initialize.
        """
        if isinstance(module, nn.Linear):
            # For the final layer, initialise weights with a smaller gain to keep initial actions near zero
            if module == self.mean_layer:
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
            else:
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs):
        """Forward pass through the actor network.

        Args:
            obs (torch.Tensor): Batch of observations with shape [batch_size, obs_dim].
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (mean, log_std) each with shape [batch_size, action_dim].
        """
        x1 = self.x1(obs)
        mean = self.mean_layer(x1)
        log_std = self.log_std_layer(x1)

        # Clamp log_std
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # This implements the reparameterization trick to allow gradients to flow through the sampling process
        normal = Normal(mean, std)
        x_t = normal.rsample()

        # Squash output to [-1, 1] for plane control surfaces
        y_t = torch.tanh(x_t)
        action = y_t

        # Calculate log probability
        log_prob = normal.log_prob(x_t)

        # Enforcing action bounds (see appendix C of SAC paper for details)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class SACAgent(BaseAgent):
    def __init__(self, obs_dim: int, action_dim: int, num_envs: int, device: torch.device, config: Config):
        super().__init__(obs_dim, action_dim, num_envs, device, config)

        self.gamma = config('gamma')
        self.tau = config('tau')
        self.batch_size = config('batch_size')
        self.init_alpha = config('init_alpha')

        self.target_entropy = -action_dim
        self.total_epochs = 0

        self.log_alpha = torch.tensor(np.log(self.init_alpha), requires_grad=True, device=device)

        self.actor = GaussianActor(obs_dim, action_dim, config('actor_hidden_dim')).to(device)
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

        self.alpha_optim = torch.optim.Adam(
            [self.log_alpha],
            lr=config('alpha_learning_rate'),
            weight_decay=config('alpha_weight_decay')
        )

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, _, _ = self.actor.sample(obs)
        return action

    def update(self, replay_buffer: ReplayBuffer, epochs: int, log: LoggingStruct):
        for epoch in range(epochs):
            # Sample a batch from the replay buffer
            # This is of shape [batch_size, num_envs, ...], so we need to reshape it to [batch_size * num_envs, ...] for processing
            obs, actions, rewards, next_obs, dones = replay_buffer.sample(self.batch_size)

            obs = obs.view(self.batch_size * self.num_envs, self.obs_dim)
            actions = actions.view(self.batch_size * self.num_envs, self.action_dim)
            next_obs = next_obs.view(self.batch_size * self.num_envs, self.obs_dim)

            rewards = rewards.view(self.batch_size * self.num_envs, 1)
            dones = dones.view(self.batch_size * self.num_envs, 1).float()

            # Critic update
            with torch.no_grad():
                # Get next actions and entropy from current policy
                next_actions, next_log_probs, _ = self.actor.sample(next_obs)

                # Get target Q values from target critics
                target_q1, target_q2 = self.critic_target(next_obs, next_actions)

                # Take the minimum to prevent overestimation bias, and subtract the entropy penalty
                min_target_q = torch.min(target_q1, target_q2) - torch.exp(self.log_alpha) * next_log_probs

                # Calculate the Bellman backup (y)
                # If done=True (1.0), the future value is 0.
                target_value = rewards + (1.0 - dones.float()) * self.gamma * min_target_q

            # Get current Q values from current critics
            current_q1, current_q2 = self.critic(obs, actions)

            # Calculate mean squared error for both critics
            critic_1_loss = F.mse_loss(current_q1, target_value)
            critic_2_loss = F.mse_loss(current_q2, target_value)
            critic_loss = critic_1_loss + critic_2_loss

            # Optimize Critic
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Actor update
            # Sample NEW actions from the current state to evaluate the policy
            new_actions, log_probs, _ = self.actor.sample(obs)

            # Get Q values for these new actions from the current critics
            q1_new, q2_new = self.critic(obs, new_actions)
            min_q_new = torch.min(q1_new, q2_new)

            # Actor wants to maximize Q-value and maximize entropy.
            # PyTorch minimizes by default, so we minimize: (alpha * log_prob) - Q
            actor_loss = (self.log_alpha.exp().detach() * log_probs - min_q_new).mean()

            # Optimize Actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Alpha update
            # We want to tune alpha so that the policy's entropy matches our target_entropy (-4.0)
            # We have to detach the log probs so gradients don't flow back into the Actor again
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()

            # Optimize Alpha
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            # Target network soft update (Polyak Averaging)
            # Slowly blend the current critic weights into the target critic
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            # Finally, log any useful metrics on the final epoch cycle
            if epoch == epochs - 1:
                log.update({
                    "loss/critic_1": critic_1_loss.item(),
                    "loss/critic_2": critic_2_loss.item(),
                    "loss/actor": actor_loss.item(),
                    "loss/alpha": alpha_loss.item(),
                    "metric/alpha_value": self.log_alpha.exp().item(),
                    "metric/batch_entropy": -log_probs.mean().item()
                })

    def save(self, path: str):
        data = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'optimizer_state_dict': {
                'actor': self.actor_optim.state_dict(),
                'critic': self.critic_optim.state_dict(),
                'alpha': self.alpha_optim.state_dict()
            },
            'log_alpha': self.log_alpha
        }
        torch.save(data, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data['actor_state_dict'])
        self.critic.load_state_dict(data['critic_state_dict'])
        self.critic_target.load_state_dict(data['critic_target_state_dict'])
        self.actor_optim.load_state_dict(data['optimizer_state_dict']['actor'])
        self.critic_optim.load_state_dict(data['optimizer_state_dict']['critic'])
        self.alpha_optim.load_state_dict(data['optimizer_state_dict']['alpha'])
        self.log_alpha = data['log_alpha']