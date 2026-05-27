import torch

from src.algorithms.sac.sac_agent import SACAgent
from src.algorithms.sac.attitude_agent import AttitudeAgent
from src.algorithms.td3.td3_agent import TD3Agent

class LayeredSACAgent(SACAgent):
    """
    This agent splits the control into layers.
    The outer loop (SAC) generates desired targets (pitch, roll, beta, velocity).
    The inner loop (AttitudeAgent) maps these targets and current states into physical control surface commands.
    """
    def __init__(self, obs_dim, action_dim, num_envs, device, config, attitude_config):
        super().__init__(obs_dim, action_dim, num_envs, device, config)
        if attitude_config("name") == "attitude_agent":
            self.attitude_agent = AttitudeAgent(14, 4, num_envs, device, attitude_config)
        elif attitude_config("name") == "td3_agent":
            self.attitude_agent = TD3Agent(14, 4, num_envs, device, attitude_config)
        else:
            raise ValueError(f"Unsupported inner agent algorithm: {attitude_config('name')}")

        self.last_outer_action = None

    def _wrap_pi(self, angles):
        return (angles + torch.pi) % (2 * torch.pi) - torch.pi

    def act(self, obs):
        outer_action = super().act(obs)
        self.last_outer_action = outer_action.clone()

        target_pitch = outer_action[:, 0] * (torch.pi / 2)
        target_roll = outer_action[:, 1] * torch.pi
        target_beta = outer_action[:, 2] * 0.5
        target_vt = (outer_action[:, 3] + 1.0) * 350.0 + 300.0

        roll = torch.atan2(obs[:, 4], obs[:, 5])
        pitch = torch.atan2(obs[:, 6], obs[:, 7])
        beta = torch.atan2(obs[:, 11], obs[:, 12])

        vt_fts = obs[:, 8] * 340.0 / 0.3048

        e_beta = self._wrap_pi(target_beta - beta)
        e_pitch = self._wrap_pi(target_pitch - pitch)
        e_roll = self._wrap_pi(target_roll - roll)
        delta_vt_fts = vt_fts - target_vt

        c_beta = (6.0 / torch.pi) * 4.0
        c_pitch = (6.0 / torch.pi) * 1.0
        c_roll = (6.0 / torch.pi) * 1.0

        weighted_e_beta = e_beta * c_beta
        weighted_e_pitch = e_pitch * c_pitch
        weighted_e_roll = e_roll * c_roll

        norm_el = obs[:, 17]
        norm_ail = obs[:, 18]
        norm_rud = obs[:, 19]
        norm_P = obs[:, 13]
        norm_Q = obs[:, 14]
        norm_R = obs[:, 15]
        norm_vt = obs[:, 8]
        alpha_sin = obs[:, 9]
        alpha_cos = obs[:, 10]
        eas2tas = obs[:, 21]

        inner_obs = torch.stack([
            weighted_e_beta,
            weighted_e_pitch,
            weighted_e_roll,
            norm_el,
            norm_ail,
            norm_rud,
            norm_P,
            norm_Q,
            norm_R,
            norm_vt,
            alpha_sin,
            alpha_cos,
            eas2tas,
            delta_vt_fts
        ], dim=-1)

        action = self.attitude_agent.act(inner_obs)
        return action

    def load_inner_agent(self, path: str):
        """Helper function to load pre-trained weights specifically for the inner loop."""
        self.attitude_agent.load(path)