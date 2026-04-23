import torch

from src.configs.config import Config

class BaseEnv:
    def __init__(self, config: Config, num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        self.config = config
        self.device = device
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim

def env_from_config(config: Config, device: torch.device) -> BaseEnv:
    if config('env_name') == "neural_plane":
        from src.envs.neural_plane import create_neuralplane_env
        return create_neuralplane_env(config, device)
    else:
        raise ValueError(f"Unsupported environment name: {config('env_name')}")