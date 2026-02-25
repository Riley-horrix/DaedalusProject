import sys
import os
import torch
import numpy as np

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

def test_pid():
    env = ControlEnv(num_envs=1, config='tracking', model='F16', device=device)

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

    for i in range(2500):
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

        obs, _, _, _, _, info = env.step(pid_action)

        if i % 5 == 0:
            env.render(count=i, filename="./logs/flightpath")

        # Logging
        if i % 100 == 0:
            npos, epos, alt = env.model.get_position()
            dist = torch.norm(target_waypoint[:,:2] - torch.hstack((npos.reshape(-1,1), epos.reshape(-1,1))), dim=1)
            vt = env.model.get_vt()
            print(f"Step {i} | Dist to Target: {dist[0].item():.1f}m | Alt: {alt[0].item():.1f}m | Speed: {vt[0].item():.1f}m/s")

    print("Saving replay... Check ./logs/flightpath*.acmi")

if __name__ == "__main__":
    test_pid()