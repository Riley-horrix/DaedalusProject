import torch

from src.algorithms.sac.sac_agent import SACAgent
from src.algorithms.td3.td3_agent import TD3Agent

class AttitudeAgent(SACAgent):
    """
    TD3Agent for attitude control task. The observation is the 13-dimensional format of the inner-loop agent.
    The action is the 4-dimensional control surface deflection, with thrust coming from a PID.
    """
    def __init__(self, obs_dim, action_dim, num_envs, device, config):
        super().__init__(obs_dim, action_dim, num_envs, device, config)

        self.kp = config('kp', default=0.05)
        self.ki = config('ki', default=0.001)
        self.kd = config('kd', default=0.01)
        self.dt = config('dt', default=0.01)

        # State tensors for I and D components
        self.integral_error = torch.zeros(num_envs, device=device)
        self.prev_error = torch.zeros(num_envs, device=device)

    def act(self, obs):
        """
        Override the act function to add a PID thrust component to the action.

        Args:
            obs: observation tensor of shape (num_envs, obs_dim)

        Returns:
            action: tensor of shape (num_envs, action_dim)

        Expects the following obs format (dim 14):
            0. weighted_error_beta
            1. weighted_error_pitch
            2. weighted_error_roll
            3. ego_el (elevator deflection)
            4. ego_ail (aileron deflection)
            5. ego_rud (rudder deflection)
            6. ego_P (roll rate)
            7. ego_Q (pitch rate)
            8. ego_R (yaw rate)
            9. vt (velocity)
            10. sin_alpha (angle of attack)
            11. cos_alpha (angle of attack)
            12. EAS2TAS
            13. delta_vt_fts (velocity error in ft/s)
        """
        full_action = torch.hstack((super().act(obs[:, :-1]), self.get_thrust(obs[:, -1])))
        return full_action.clamp(-1.0, 1.0)

    def reset_pid_states(self, done_mask: torch.Tensor):
        """
        MUST be called from main training loop whenever environments reset!
        Args:
            done_mask: Boolean tensor of shape (num_envs,)
        """
        self.integral_error[done_mask] = 0.0
        self.prev_error[done_mask] = 0.0

    def get_thrust(self, error: torch.Tensor) -> torch.Tensor:
        """
        Calculate the thrust command for the current time step.
        Args:
            error: Velocity error in ft/s (shape: [num_envs])
        Returns:
            thrust: Tensor of shape [num_envs, 1] clamped between [-1.0, 1.0]
        """
        p_term = self.kp * error

        self.integral_error += error * self.dt
        # Prevent the integral term from accumulating to infinity during long stalls
        self.integral_error = torch.clamp(self.integral_error, min=-50.0, max=50.0)
        i_term = self.ki * self.integral_error

        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative

        # Save current error for next step's derivative calculation
        self.prev_error = error.clone()

        # Combine terms
        pid_output = p_term + i_term + d_term
        thrust = torch.clamp(pid_output, -1.0, 1.0)

        return thrust.reshape(-1, 1)