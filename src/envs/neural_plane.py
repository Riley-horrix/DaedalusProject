import torch
import sys
import os

# Ensure the NeuralPlane library is accessible just like in your test script
current_dir = os.path.dirname(os.path.abspath(__file__))
neural_plane_path = os.path.join(current_dir, './lib/NeuralPlane')
if neural_plane_path not in sys.path:
    sys.path.append(os.path.abspath(neural_plane_path))

from lib.NeuralPlane.envs.control_env import ControlEnv

from src.configs.config import Config
from src.envs.base_env import BaseEnv

class NeuralPlaneEnv(BaseEnv):
    def __init__(self, config: Config, device: torch.device):
        super().__init__(config, num_envs=config('num_envs', default=1), obs_dim=22, action_dim=4, device=device)

        if config('env_type') == 'control':
            self.env = ControlEnv(num_envs=self.num_envs, config='tracking', model='F16', device=device)
        elif config('env_type') == 'attitude':
            self.env = ControlEnv(num_envs=self.num_envs, config='control', model='F16', device=device)
        else:
            raise ValueError(f"Unsupported env_type: {config('env_type')}")


        # Setup environment parameters based on config
        self.env.model.min_altitude = config('min_init_altitude', default=1000.0)
        self.env.model.max_altitude = config('max_init_altitude', default=10000.0)
        self.env.model.min_vt = config('min_init_v', default=150.0)
        self.env.model.max_vt = config('max_init_v', default=600.0)

    def reset(self):
        """
        Resets all parallel environments simultaneously.
        Returns: Batched observation tensor [num_envs, obs_dim]
        """
        obs = self.env.reset()
        return obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Pipes the neural network's batched actions directly into the FDM.

        Args:
            actions(torch.Tensor): Tensor of shape [num_envs, 4] bounded between [-1, 1]
        Returns:
            obs(torch.Tensor): Batched observations [num_envs, obs_dim]
            reward(torch.Tensor): Batched rewards [num_envs]
            done(torch.Tensor): Batched done flags [num_envs]
            bad_done(torch.Tensor): Batched bad done flags [num_envs]
            timeout(torch.Tensor): Batched timeout flags [num_envs]
            info(dict): List of info dicts for each environment
        """
        actions = actions.to(self.device).contiguous()
        obs, reward, done, bad_done, timeout, info = self.env.step(actions)

        """
        NeuralPlane observation units:

        observation(dim 22):
            0. ego_delta_npos      (unit: km)
            1. ego_delta_epos       (unit km)
            2. ego_delta_altitude            (unit: km)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego_vt                  (unit: mh)
            9. ego_alpha_sin
            10. ego_alpha_cos
            11. ego_beta_sin
            12. ego_beta_cos
            13. ego_P                  (unit: rad/s)
            14. ego_Q                  (unit: rad/s)
            15. ego_R                  (unit: rad/s)
            16. ego_T                  (unit: %)
            17. ego_el                 (unit: %)
            18. ego_ail                (unit: %)
            19. ego_rud                (unit: %)
            20. ego_lef                (unit: %)
            21. EAS2TAS
        """
        return obs, reward, done, bad_done, timeout, info


def create_neuralplane_env(config: Config, device: torch.device) -> BaseEnv:
    """Factory function to create a NeuralPlane environment instance based on the provided configuration.

    Args:
        config (Config): Configuration object containing environment parameters and settings.
        device (torch.device): PyTorch device on which to run the environment.

    Returns:
        BaseEnv: An instance of NeuralPlaneEnv initialized with the specified configuration.
    """
    return NeuralPlaneEnv(config, device=device)