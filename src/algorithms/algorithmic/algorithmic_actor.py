import torch
import torch.nn as nn

from src.algorithms.sac.sac_agent import SACAgent
from src.algorithms.sac.attitude_agent import AttitudeAgent
from src.algorithms.td3.td3_agent import TD3Agent

class AlgorithmicLayeredAgent:
    """
    Cascaded Controller with an Algorithmic (Geometric) Outer Loop.
    Outer Loop (Classical): Algebraically generates desired aerodynamic targets [-1, 1].
    Inner Loop (Frozen): Maps physical targets to control surface deflections via a pre-trained policy.
    """
    # FIX 1: Lower kp_roll to 0.4 and drop smoothing_alpha to 0.05 for aggressive low-pass filtering
    def __init__(self, num_envs, device, attitude_config, kp_roll=0.4, kp_pitch=1.2, target_mach=1.1, smoothing_alpha=0.05):
        self.num_envs = num_envs
        self.device = device

        self.kp_roll = kp_roll
        self.kp_pitch = kp_pitch
        self.target_mach = target_mach

        self.smoothing_alpha = smoothing_alpha
        self.smoothed_outer_action = None

        inner_obs_dim = 11
        inner_action_dim = 4

        if attitude_config("name") == "attitude_agent":
            self.attitude_agent = AttitudeAgent(inner_obs_dim, inner_action_dim, num_envs, device, attitude_config)
        elif attitude_config("name") == "sac_agent":
            self.attitude_agent = SACAgent(inner_obs_dim, inner_action_dim, num_envs, device, attitude_config)
        elif attitude_config("name") == "td3_agent":
            self.attitude_agent = TD3Agent(inner_obs_dim, inner_action_dim, num_envs, device, attitude_config)
        else:
            raise ValueError(f"Unsupported inner agent algorithm: {attitude_config('name')}")

        for param in self.attitude_agent.actor.parameters():
            param.requires_grad = False

        self.last_outer_action = None

        self.max_pitch_increment = getattr(attitude_config, 'max_pitch_increment', 0.3)
        self.max_roll_increment = getattr(attitude_config, 'max_roll_increment', 0.3)
        self.max_velocities_u_increment = getattr(attitude_config, 'max_velocities_u_increment', 100.0)
        self.noise_scale = getattr(attitude_config, 'noise_scale', 0.01)

    def _wrap_pi(self, angles):
        return (angles + torch.pi) % (2 * torch.pi) - torch.pi

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        obs = obs.to(self.device)
        batch_size = obs.shape[0]

        # 1. Extract outer tracking parameters from tracking state vector s_track
        d_ego = obs[:, 0]
        delta_psi_az = obs[:, 1]
        delta_theta_el = obs[:, 2]
        norm_EAS = obs[:, 8]

        # 2. Compute Raw Classical Outer-Loop Targets bounded within [-1, 1]
        raw_outer_action = torch.zeros((batch_size, 4), device=self.device)
        raw_outer_action[:, 0] = 0.0

        raw_outer_action[:, 1] = torch.clamp(self.kp_pitch * delta_theta_el, min=-1.0, max=1.0)

        roll_cmd = -self.kp_roll * delta_psi_az
        near_field_mask = d_ego < 0.2
        roll_cmd = torch.where(near_field_mask, roll_cmd * (d_ego / 0.2), roll_cmd)

        # FIX 2: Restrict the maximum requested roll target to 60% of max capacity during transient flight
        # This keeps the inner loop far away from its highly non-linear saturation limits
        raw_outer_action[:, 2] = torch.clamp(roll_cmd, min=-0.6, max=0.6)

        v_error = self.target_mach - norm_EAS
        v_cmd = 5.0 * v_error
        turn_severity = torch.abs(raw_outer_action[:, 2])
        v_cmd = v_cmd * (1.0 - 0.2 * turn_severity)
        raw_outer_action[:, 3] = torch.clamp(v_cmd, min=-0.2, max=1.0)

        # 3. Apply the heavily damped Low-Pass Filter path
        if self.smoothed_outer_action is None or self.smoothed_outer_action.shape[0] != batch_size:
            self.smoothed_outer_action = raw_outer_action.clone()
        else:
            self.smoothed_outer_action = (self.smoothing_alpha * raw_outer_action +
                                          (1.0 - self.smoothing_alpha) * self.smoothed_outer_action)

        outer_action = self.smoothed_outer_action.clone()
        self.last_outer_action = outer_action.clone()

        # 4. Un-squash smoothed outer actions
        target_beta = outer_action[:, 0] * 0.0
        target_pitch = outer_action[:, 1] * self.max_pitch_increment
        target_roll = outer_action[:, 2] * self.max_roll_increment
        target_vt_delta = outer_action[:, 3] * self.max_velocities_u_increment

        roll = torch.atan2(obs[:, 4], obs[:, 5])
        pitch = torch.atan2(obs[:, 6], obs[:, 7])
        beta = torch.atan2(obs[:, 11], obs[:, 12])

        delta_vt_fts = -target_vt_delta

        norm_P = obs[:, 13]
        norm_Q = obs[:, 14]
        norm_R = obs[:, 15]

        norm_thr = obs[:, 16]
        norm_el = obs[:, 17]
        norm_ail = obs[:, 18]
        norm_rud = obs[:, 19]

        e_beta = self._wrap_pi(target_beta - beta)
        e_pitch = self._wrap_pi(target_pitch - pitch)
        e_roll = self._wrap_pi(target_roll - roll)

        # 5. Generate the 11-Dimensional Inner State Vector
        weighted_e_beta = e_beta * (6.0 / torch.pi) * 4.0
        weighted_e_pitch = e_pitch * (6.0 / torch.pi) * 1.0
        weighted_e_roll = e_roll * (6.0 / torch.pi) * 1.0
        weighted_e_vel = delta_vt_fts * ((6.0 / torch.pi) * 0.5) / self.max_velocities_u_increment

        inner_obs = torch.stack([
            weighted_e_beta,   # 0
            weighted_e_pitch,  # 1
            weighted_e_roll,   # 2
            weighted_e_vel,    # 3
            norm_el,           # 4
            norm_ail,          # 5
            norm_rud,          # 6
            torch.zeros_like(norm_rud),          # 7
            norm_P,            # 8
            norm_Q,            # 9
            norm_R             # 10
        ], dim=-1)

        inner_obs = inner_obs + torch.randn_like(inner_obs) * self.noise_scale

        # 6. Process through frozen downstream architecture
        with torch.no_grad():
            if isinstance(self.attitude_agent, SACAgent):
                action = self.attitude_agent.act(inner_obs, deterministic=True)
            else:
                action = self.attitude_agent.act(inner_obs, add_noise=False)

        return action

    def load_inner_agent(self, path: str):
        self.attitude_agent.load(path)