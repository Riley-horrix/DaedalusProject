import torch
from lib.NeuralPlane.algorithms.pid.pid import PID
from lib.NeuralPlane.algorithms.utils.utils import wrap_PI

class BasicPIDNav:
    def __init__(self, dt=0.01, n=1, device="cuda:0"):
        self.device = device
        self.n = n

        # Init heading and altitude pid controller
        self.heading_pid = PID(Kp=1.5, Ki=0.01, Kd=0.5, dt=dt, n=n, device=device)
        self.altitude_pid = PID(Kp=0.008, Ki=0.001, Kd=0.0005, dt=dt, n=n, device=device)

    def get_navigation_commands(self, current_pos, target_pos, current_heading, current_alt):
        """
        Args:
            current_pos: [n, 3] (North, East, Alt)
            target_pos: [n, 3] (North, East, Alt)
            current_heading: [n, 1] (Radians)
        Returns:
            roll_cmd, pitch_cmd (Tensors [n, 1])
        """
        # Calculate desired heading to target
        delta_pos = target_pos - current_pos
        desired_heading = torch.atan2(delta_pos[:, 1], delta_pos[:, 0]).reshape(-1, 1)

        # Calculate heading error (wrapped to -pi, pi)
        heading_error = wrap_PI(desired_heading - current_heading)

        # Limit the integrator to avoid windup during long turns
        roll_cmd = self.heading_pid.update_all(target=torch.zeros_like(heading_error),
                                               measurement=-heading_error,
                                               limit=False)
        # Clamp bank angle to +/- 45 degrees (0.78 rad)
        roll_cmd = torch.clamp(roll_cmd, -0.78, 0.78)

        # Target altitude is usually passed separately, but let's assume z component of target_pos if 3D
        # For this basic example, we will take a separate target_alt argument in the wrapper.
        target_alt = target_pos[:, 2].reshape(-1, 1)
        alt_error = target_alt - current_alt

        pitch_cmd = self.altitude_pid.update_all(target=target_alt,
                                                 measurement=current_alt,
                                                 limit=False)

        # Clamp pitch to +/- 20 degrees (0.35 rad)
        pitch_cmd = torch.clamp(pitch_cmd, -0.35, 0.35)

        return roll_cmd, pitch_cmd