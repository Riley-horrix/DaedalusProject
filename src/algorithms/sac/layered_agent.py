import torch
import torch.nn as nn

from src.algorithms.sac.sac_agent import SACAgent
from src.algorithms.sac.attitude_agent import AttitudeAgent
from src.algorithms.td3.td3_agent import TD3Agent

class LayeredSACAgent(SACAgent):
    """
    Cascaded Controller.
    Outer Loop (SAC): Generates desired aerodynamic targets [-1, 1].
    Inner Loop (Frozen): Maps physical targets to control surface deflections.
    """
    def __init__(self, obs_dim, action_dim, num_envs, device, config, attitude_config):
        super().__init__(obs_dim, action_dim, num_envs, device, config)

        # The AttitudeTask defines exactly 11 observation dimensions.
        inner_obs_dim = 11
        inner_action_dim = 4 # el, ail, rud, thr

        if attitude_config("name") == "attitude_agent":
            self.attitude_agent = AttitudeAgent(inner_obs_dim, inner_action_dim, num_envs, device, attitude_config)
        elif attitude_config("name") == "sac_agent":
            self.attitude_agent = SACAgent(inner_obs_dim, inner_action_dim, num_envs, device, attitude_config)
        elif attitude_config("name") == "td3_agent":
            self.attitude_agent = TD3Agent(inner_obs_dim, inner_action_dim, num_envs, device, attitude_config)
        else:
            raise ValueError(f"Unsupported inner agent algorithm: {attitude_config('name')}")

        # Freeze the inner agent completely to prevent catastrophic forgetting
        for param in self.attitude_agent.actor.parameters():
            param.requires_grad = False

        self.last_outer_action = None

        # Physical limits to un-squash outer loop actions
        self.max_pitch_target = 0.5  # radians (~28 degrees)
        self.max_roll_target = 1.0   # radians (~57 degrees)
        self.max_vt_target_delta = 100.0 # ft/s
        self.max_velocities_u_increment = 100.0 # From AttitudeTask

    def _wrap_pi(self, angles):
        return (angles + torch.pi) % (2 * torch.pi) - torch.pi

    def act(self, obs, deterministic=False):
        # 1. Get [-1, 1] normalized targets from the Outer Policy
        outer_action = super().act(obs, deterministic=deterministic)
        self.last_outer_action = outer_action.clone()

        # 2. Un-squash actions to physical targets
        # SAC outputs [-1, 1]. We must scale these to actual flight envelopes.
        target_beta = outer_action[:, 0] * 0.0 # Usually force 0 for coordinated flight
        target_pitch = outer_action[:, 1] * self.max_pitch_target
        target_roll = outer_action[:, 2] * self.max_roll_target
        target_vt_delta = outer_action[:, 3] * self.max_vt_target_delta

        # 3. Extract raw states from the Outer Observation tensor
        # NOTE: Ensure these indices match your actual OuterLoopTask observation space!
        roll = torch.atan2(obs[:, 4], obs[:, 5])
        pitch = torch.atan2(obs[:, 6], obs[:, 7])
        beta = torch.atan2(obs[:, 11], obs[:, 12])

        vt_fts = obs[:, 8] * 340.0 / 0.3048 # Conversion logic you provided
        target_vt = vt_fts + target_vt_delta

        # Extract control surfaces and rates (Ensure indices are correct for your env)
        norm_P = obs[:, 13]
        norm_Q = obs[:, 14]
        norm_R = obs[:, 15]
        norm_el = obs[:, 17]
        norm_ail = obs[:, 18]
        norm_rud = obs[:, 19]
        norm_thr = obs[:, 16] # Added throttle to match the 11-dim inner obs!

        # 4. Calculate tracking errors
        e_beta = self._wrap_pi(target_beta - beta)
        e_pitch = self._wrap_pi(target_pitch - pitch)
        e_roll = self._wrap_pi(target_roll - roll)
        delta_vt_fts = vt_fts - target_vt

        # 5. Apply AttitudeTask weighting scalars
        weighted_e_beta = e_beta * (6.0 / torch.pi) * 4.0
        weighted_e_pitch = e_pitch * (6.0 / torch.pi) * 1.0
        weighted_e_roll = e_roll * (6.0 / torch.pi) * 1.0
        weighted_e_vel = delta_vt_fts * ((6.0 / torch.pi) * 0.5) / self.max_velocities_u_increment

        # 6. Construct the exact 11-dimensional Inner Observation
        inner_obs = torch.stack([
            weighted_e_beta,   # 0
            weighted_e_pitch,  # 1
            weighted_e_roll,   # 2
            weighted_e_vel,    # 3
            norm_el,           # 4
            norm_ail,          # 5
            norm_rud,          # 6
            norm_thr,          # 7 (Was missing in your draft)
            norm_P,            # 8
            norm_Q,            # 9
            norm_R             # 10
        ], dim=-1)

        # 7. Query the pre-trained Inner Agent for physical control surfaces
        with torch.no_grad():
            action = self.attitude_agent.act(inner_obs, deterministic=True)

        return action

    def load_inner_agent(self, path: str):
        """Helper function to load pre-trained weights specifically for the inner loop."""
        self.attitude_agent.load(path)