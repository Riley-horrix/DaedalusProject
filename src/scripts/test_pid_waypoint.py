import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
neural_plane_path = os.path.join(current_dir, '../../lib/NeuralPlane')

sys.path.append(os.path.abspath(neural_plane_path))

from lib.NeuralPlane.envs.control_env import ControlEnv
from lib.NeuralPlane.algorithms.pid.controller import Controller

device = torch.device("cpu")

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda:0")
else:
    print("Using CPU")

obs_buffer = []

def plot_flight_telemetry(obs_history):
    """
    Plots flight angles from a history of NeuralPlane observations.

    Args:
        obs_history: A list of numpy arrays or a 2D numpy array of shape (timesteps, 22)
    """
    # Ensure it's a 2D numpy array
    obs_array = np.array(obs_history)

    # If the array has an extra batch dimension like (timesteps, 1, 22), squeeze it
    if obs_array.ndim == 3:
        obs_array = np.squeeze(obs_array, axis=1)

    # 1. Extract radians using arctan2(sin, cos)
    # Roll: obs[4]=sin, obs[5]=cos
    roll_rad = np.arctan2(obs_array[:, 4], obs_array[:, 5])
    # Pitch: obs[6]=sin, obs[7]=cos
    pitch_rad = np.arctan2(obs_array[:, 6], obs_array[:, 7])
    # Alpha (Angle of Attack): obs[9]=sin, obs[10]=cos
    alpha_rad = np.arctan2(obs_array[:, 9], obs_array[:, 10])
    # Beta (Sideslip): obs[11]=sin, obs[12]=cos
    beta_rad = np.arctan2(obs_array[:, 11], obs_array[:, 12])

    # Deltas are already in radians
    delta_pitch_rad = obs_array[:, 0]
    delta_heading_rad = obs_array[:, 1]

    # 2. Convert radians to degrees
    roll_deg = np.degrees(roll_rad)
    pitch_deg = np.degrees(pitch_rad)
    alpha_deg = np.degrees(alpha_rad)
    beta_deg = np.degrees(beta_rad)
    delta_pitch_deg = np.degrees(delta_pitch_rad)
    delta_heading_deg = np.degrees(delta_heading_rad)

    # 3. Create the plots
    steps = np.arange(len(obs_array))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Posture (Roll & Pitch)
    axes[0].plot(steps, roll_deg, label="Roll", color='blue', linewidth=2)
    axes[0].plot(steps, pitch_deg, label="Pitch", color='red', linewidth=2)
    axes[0].axhline(90, color='black', linestyle='--', alpha=0.5, label="+90 Gimbal Lock")
    axes[0].axhline(-90, color='black', linestyle='--', alpha=0.5)
    axes[0].set_ylabel("Degrees")
    axes[0].set_title("Aircraft Posture (Roll & Pitch)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True)

    # Subplot 2: Aerodynamic Angles (Alpha & Beta)
    axes[1].plot(steps, alpha_deg, label="Alpha (Angle of Attack)", color='purple', linewidth=2)
    axes[1].plot(steps, beta_deg, label="Beta (Sideslip)", color='green', linewidth=2)
    axes[1].set_ylabel("Degrees")
    axes[1].set_title("Aerodynamic Angles")
    axes[1].legend(loc="upper left")
    axes[1].grid(True)

    # Subplot 3: Target Tracking Errors
    axes[2].plot(steps, delta_pitch_deg, label="Delta Pitch to Target", color='orange')
    axes[2].plot(steps, delta_heading_deg, label="Delta Heading to Target", color='cyan')
    axes[2].set_xlabel("Simulation Step")
    axes[2].set_ylabel("Degrees")
    axes[2].set_title("Target Tracking Error")
    axes[2].legend(loc="upper left")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

def test_pid():
    env = ControlEnv(num_envs=1, config='control', model='F16', device=device)

    env.controller = Controller(dt=env.model.dt, n=env.n, device=env.device)

    env.model.min_altitude = 5000.0
    env.model.max_altitude = 5000.0
    env.model.min_vt = 280.0
    env.model.max_vt = 280.0

    original_termination = env.task.get_termination
    def lenient_termination(env, info):
        done, bad_done, timeout, info = original_termination(env, info)
        return torch.zeros_like(done), torch.zeros_like(bad_done), timeout, info
    env.task.get_termination = lenient_termination

    obs = env.reset()

    start_waypoint = torch.tensor([[-1.0, -1.0, 5000.0]], device=device)
    target_waypoint = torch.tensor([[10000.0, 10000.0, 5000.0]], device=device)

    start_waypoint = start_waypoint.repeat(env.n, 1)
    target_waypoint = target_waypoint.repeat(env.n, 1)

    target_speed = torch.ones((env.n, 1), device=device) * 280.0

    print("Starting PID Waypoint Test...")

    for i in range(5000):
        state = env.model.get_state()
        estate = env.model.get_extended_state()
        eas2tas = env.model.get_EAS2TAS().reshape(-1, 1)

        env.controller.update_waypoint(
            start_waypoint[:, :2],
            target_waypoint[:, :2],
            100.0,
            state,
            estate,
            eas2tas
        )

        target_alt = target_waypoint[:, 2].reshape(-1, 1)
        env.controller.cal_pitch_throttle(target_alt, target_speed, env)

        env.controller.stabilize(env)

        pid_action = env.controller.get_action()

        obs, reward, done, bad_done, exceed_time_limit, info = env.step(pid_action)

        if i % 50 == 0:
            env.render(count=i, filename="./logs/flightpath")

        if i % 5 == 0:
            obs_buffer.append(obs[0].cpu().numpy())

        # Logging
        if i % 100 == 0:
            npos, epos, alt = env.model.get_position()
            dist = torch.norm(target_waypoint[:,:2] - torch.hstack((npos.reshape(-1,1), epos.reshape(-1,1))), dim=1)
            vt = env.model.get_vt()
            print(f"Step {i} | Dist to Target: {dist[0].item():.1f}m | Alt: {alt[0].item():.1f}m | Speed: {vt[0].item():.1f}m/s")

    print("Saving replay... Check ./logs/flightpath*.acmi")

if __name__ == "__main__":
    try:
        test_pid()
    except KeyboardInterrupt:
        print("Test interrupted. Plotting telemetry...")
        plot_flight_telemetry(obs_buffer[0:])