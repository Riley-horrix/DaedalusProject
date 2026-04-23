import torch

import src.utils.rate_limited_printer as rlp

class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, num_envs: int, device: torch.device):
        """A circular buffer implementation using torch Tensors to keep replay buffer data on the GPU.

        Args:
            obs_dim (int): Dimensionality of the observation space
            action_dim (int): Dimensionality of the action space
            capacity (int): Maximum number of transitions to store in the buffer (total across all environments),
                this should be set to at least (episode_length * num_envs) to ensure you can capture full episodes without overwriting early data.
            num_envs (int): Number of parallel environments (used to determine how to store batched data)
            device (torch.device): Device on which to store the replay buffer tensors (e.g., torch.device('cuda'))
        """
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self.ptr = 0
        self.size = 0
        self.actual_size = 0
        self.max_overrun = 0

        self.obs: torch.Tensor = torch.zeros((self.capacity, num_envs, obs_dim), device=device)
        self.actions: torch.Tensor = torch.zeros((self.capacity, num_envs, action_dim), device=device)
        self.rewards: torch.Tensor = torch.zeros((self.capacity, num_envs, 1), device=device)
        self.next_obs: torch.Tensor = torch.zeros((self.capacity, num_envs, obs_dim), device=device)
        self.dones: torch.Tensor = torch.zeros((self.capacity, num_envs, 1), device=device, dtype=torch.bool)

        print(f"Total VRAM usage by replay buffer: {self.obs.element_size() * self.obs.nelement() * 5 / (1024 ** 3):.2f} GB")

        self.printer = rlp.RateLimitedPrinter(interval=5.0)

    def push(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, done: torch.Tensor):
        """Push batched replay data into buffer.

        Data is expected to be in shape [num_envs, ...] and will be stored in a circular manner.

        Args:
            obs (torch.Tensor): Batched observations of shape [num_envs, obs_dim]
            action (torch.Tensor): Batched actions of shape [num_envs, action_dim]
            reward (torch.Tensor): Batched rewards of shape [num_envs]
            next_obs (torch.Tensor): Batched next observations of shape [num_envs, obs_dim]
            done (torch.Tensor): Batched done flags of shape [num_envs]
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward.unsqueeze(-1)
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done.unsqueeze(-1)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.actual_size += 1

        if self.size == self.capacity:
            self.max_overrun = max(self.max_overrun, self.actual_size - self.capacity)
            self.printer(f"Replay buffer is full - oldest data will be overwritten.\nConsider increasing capacity or reducing episode length to capture early episode data.\nMax overrun so far: {self.max_overrun} episodes")

    def sample(self, batch_size: int):
        """Sample a batch of random samples from the replay buffer.

        Args:
            batch_size (int): Number of samples to return.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): Tuple of (obs, actions, rewards, next_obs, dones) each of shape [batch_size, num_envs, ...]
        """
        indices = torch.randint(0, self.size, (batch_size,))
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices]
        )

    def clear(self):
        """Clears the replay buffer."""
        self.ptr = 0
        self.size = 0
        self.actual_size = 0