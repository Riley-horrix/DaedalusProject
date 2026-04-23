# algorithms/base_agent.py
from abc import ABC, abstractmethod
import torch

# from src.utils.rate_limited_printer import RateLimitedPrinter
from src.utils.replay_buffer import ReplayBuffer
from src.utils.logging import LoggingStruct
from src.configs.config import Config

class BaseAgent(ABC):
    def __init__(self, obs_dim: int, action_dim: int, num_envs: int, device: torch.device, config: Config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.device = device
        self.config = config

    @abstractmethod
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Given a batch of observations, return a batch of actions.

        Args:
            obs (torch.Tensor): Batch of observations with shape [num_envs, obs_dim].

        Returns:
            torch.Tensor: Batch of actions with shape [num_envs, action_dim].
        """
        pass

    @abstractmethod
    def update(self, replay_buffer: ReplayBuffer, epochs: int, log: LoggingStruct):
        """Samples from the replay buffer and performs learning updates on the agent's policy and value networks.

        Args:
            replay_buffer (rb.ReplayBuffer): The replay buffer to sample from for learning updates.
            epochs (int): Number of epochs to perform learning updates for.
            log (logging.LoggingStruct): Logging structure to log training metrics (e.g., losses) during updates.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Saves the agent to the specified path.

        Args:
            path (str): Path to save the agent's networks (e.g., a directory or file prefix).
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """Loads the agent's networks from the specified path.

        Args:
            path (str): Path to load the agent's networks from (e.g., a directory or file prefix).
        """
        pass