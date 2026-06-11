# evaluate.py
import os
import math
import torch
import sys
import wandb
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.sac.sac_agent import SACAgent
from src.algorithms.sac.layered_agent import LayeredSACAgent
from src.algorithms.algorithmic.algorithmic_actor import AlgorithmicLayeredAgent
from src.configs.config import Config
from src.envs.base_env import env_from_config
from src.utils.math import enu_to_geodetic

def export_batch_acmi(filename, step_idx, dt, npos, epos, alt, roll, pitch, yaw, t_npos=None, t_epos=None, t_alt=None, mask=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    mode = 'w' if step_idx == 0 else 'a'
    current_time = step_idx * dt

    with open(filename, mode) as f:
        if step_idx == 0:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")

        f.write(f"#{current_time:.2f}\n")

        num_agents = npos.shape[0]
        for i in range(num_agents):
            if mask is not None and mask[i]:
                obj_id = f"A{100 + i}"
                f.write(f"-{obj_id}\n")
                continue

            n = npos[i].item()
            e = epos[i].item()
            a = alt[i].item()
            lat, lon, h = enu_to_geodetic(e, n, a, 0, 0, 0)

            r = math.degrees(roll[i].item())
            p = math.degrees(pitch[i].item())
            y_ang = math.degrees(yaw[i].item())

            obj_id = f"A{100 + i}"
            if step_idx == 0:
                f.write(f"{obj_id},T={lon}|{lat}|{h}|{r}|{p}|{y_ang},Name=F-16 #{i+1},Type=Air+FixedWing,Color=Red\n")
            else:
                f.write(f"{obj_id},T={lon}|{lat}|{h}|{r}|{p}|{y_ang}\n")

            if t_npos is not None and t_epos is not None and t_alt is not None:
                t_n = t_npos[i].item()
                t_e = t_epos[i].item()
                t_a = t_alt[i].item()
                t_lat, t_lon, t_h = enu_to_geodetic(t_e, t_n, t_a, 0, 0, 0)

                t_id = f"T{100 + i}"
                if step_idx == 0:
                    f.write(f"{t_id},T={t_lon}|{t_lat}|{t_h},Name=Target #{i+1},Type=Waypoint,Color=Blue\n")
                else:
                    f.write(f"{t_id},T={t_lon}|{t_lat}|{t_h}\n")


def run_evaluate(env_config: Config, agent_config: Config, models: list[str], path_base: str, attitude_config: Config = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(models)

    # Init environment
    env = env_from_config(env_config, device=device)
    obs_dim = env.obs_dim
    action_dim = env.action_dim

    # Init models
    algorithm = agent_config('name')
    if algorithm == "sac_agent":
        agents = [SACAgent(obs_dim, action_dim, 10, device, agent_config) for _ in range(len(models))]
    if algorithm == "layered_sac_agent":
        agents = [LayeredSACAgent(obs_dim, action_dim, 10, device, agent_config, attitude_config) for _ in range(len(models))]
        inner_model_path = agent_config('inner_model_path', None)
        if inner_model_path is not None:
            print(f"Loading pre-trained inner attitude agent from {inner_model_path}...")
            for agent in agents:
                agent.load_inner_agent(inner_model_path)
        else:
            ValueError("WARNING: No pre-trained inner agent provided. The outer agent will struggle to learn!")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Load models into agents
    for i in range(len(models)):
        agents[i].load(models[i])

    reward_history = []
    action_history = []
    obs_history = []

    path = path_base

    print("Starting evaluation loop")
    for model_idx in range(len(models)):
        print(f"Evaluating {models[model_idx]}")
        agent = agents[model_idx]
        obs = env.reset()

        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

        model_name = models[model_idx].split("/")[-1]
        for epoch in range(env_config('epochs')):
            with torch.no_grad():
                    action = agent.act(obs, deterministic=True)

            next_obs, reward, done, bad_done, timeout, info = env.step(action)

            reward_history.append(reward_history[-1] + reward.squeeze().detach().cpu().numpy() if len(reward_history) > 0 else 0 + reward.squeeze().detach().cpu().numpy())
            action_history.append(action[0,:].squeeze().detach().cpu().numpy())
            obs_history.append(next_obs[0,:].squeeze().detach().cpu().numpy())

            mask |= done | bad_done | timeout

            raw_npos, raw_epos, raw_alt = env.env.model.get_position()
            roll, pitch, yaw = env.env.model.get_posture()

            t_npos = env.env.task.target_npos
            t_epos = env.env.task.target_epos
            t_alt = env.env.task.target_altitude

            # Export telemetry to Tacview ACMI format
            export_batch_acmi(
                filename=f"{path}evaluation_replay_{model_name}.acmi",
                step_idx=epoch,
                dt=env.env.model.dt,
                # F-16 Model natively runs in feet, so convert to meters
                npos=raw_npos * 0.3048,
                epos=raw_epos * 0.3048,
                alt=raw_alt * 0.3048,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                t_npos=t_npos * 0.3048,
                t_epos=t_epos * 0.3048,
                t_alt=t_alt * 0.3048,
                mask=mask
            )

            obs = next_obs

            if mask.all():
                print(f"All Agents Failed")
                break

            if epoch % 100 == 0:
                print(f"Epoch {epoch} / {env_config('epochs')}")

        print("Generating reward plot...")
        reward_matrix = np.array(reward_history)

        plt.figure(figsize=(12, 6))

        # Check if reward_matrix is 1D or 2D. If 1D (single agent), reshape it.
        if len(reward_matrix.shape) == 1:
            reward_matrix = reward_matrix[:, np.newaxis]

        num_agents_to_plot = reward_matrix.shape[1]
        colors = plt.cm.jet(np.linspace(0, 1, num_agents_to_plot))

        # Plot every single agent's trajectory with its unique color
        for i in range(num_agents_to_plot):
            plt.plot(reward_matrix[:, i], alpha=0.4, color=colors[i], label=f"Agent {i+1}" if i < 10 else None)

        plt.title(f"Individual Agent Rewards Over Time\nModel: {model_name}")
        plt.xlabel("Simulation Steps")
        plt.ylabel("Reward Signal")

        plt.legend()
        plt.grid(True, alpha=0.4)

        plot_path = f"{path}evaluation_rewards_{model_name}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved reward plot to {plot_path}")

        # Now plot each of the 4 action dimensions over time, for the first agent, with each action dimension in a subplot
        action_dimensions = ["Thrust", "Elevator", "Aileron", "Rudder"]
        action_matrix = np.array(action_history)
        plt.subplots(2, 2, figsize=(12, 8))
        for i in range(min(action_matrix.shape[1], 4)): # Safeguard if action_dim < 4
            plt.subplot(2, 2, i+1)
            plt.plot(action_matrix[:, i], alpha=0.4, label=f"Action {action_dimensions[i]}")
            plt.title(f"Action Dimension {i+1} Over Time\nModel: {model_name}")
            plt.xlabel("Simulation Steps")
            plt.ylabel("Control Signal")
            plt.legend()
            plt.grid(True, alpha=0.4)
        # Now also plot a time smoothed curve of the data
        plotted_steps = np.arange(action_matrix.shape[0])
        for i in range(min(action_matrix.shape[1], 4)):
            smoothed = np.convolve(action_matrix[:, i], np.ones(100)/100, mode='valid')
            plt.subplot(2, 2, i+1)
            plt.plot(plotted_steps[:len(smoothed)], smoothed, color='red', label=f"Smoothed Action {action_dimensions[i]}")
            plt.legend()
        plt.tight_layout()
        plot_path = f"{path}evaluation_actions_subplots_{model_name}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved action subplot to {plot_path}")

        # Now also plot flight statistics for first agent
        flight_obs = np.array(obs_history)

        tracking_datas = [
            flight_obs[:, 0],  # Norm Distance
            flight_obs[:, 1],  # Azimuth Error
            flight_obs[:, 2],  # Elevation Error
            flight_obs[:, 3]   # Altitude
        ]

        plot_graph(
            title=f"Tracking Flight Statistics\nModel: {model_name}",
            titles=["Norm Distance from Target", "Azimuth Error", "Elevation Error", "Altitude"],
            ys=["Distance (km)", "Angle (Rad)", "Angle (Rad)", "Altitude (5km)"],
            x="Time step",
            datas=tracking_datas,
            save_path=f"{path}evaluation_tracking_{model_name}.png",
            colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )

        # Reconstruct actual angles (in radians) from the trigonometric states using arctan2(sin, cos)
        roll = np.arctan2(flight_obs[:, 4], flight_obs[:, 5])
        pitch = np.arctan2(flight_obs[:, 6], flight_obs[:, 7])
        alpha = np.arctan2(flight_obs[:, 9], flight_obs[:, 10]) # Angle of Attack
        beta = np.arctan2(flight_obs[:, 11], flight_obs[:, 12]) # Angle of Sideslip

        attitude_datas = [
            roll,
            pitch,
            alpha,
            beta
        ]

        plot_graph(
            title=f"Vehicle Attitude Statistics\nModel: {model_name}",
            titles=["Roll Angle (\u03B8)", "Pitch Angle (\u03C6)", "Angle of Attack (\u03B1)", "Angle of Sideslip (\u03B2)"],
            ys=["Angle (Rad)", "Angle (Rad)", "Angle (Rad)", "Angle (Rad)"],
            x="Time step",
            datas=attitude_datas,
            save_path=f"{path}evaluation_attitude_{model_name}.png",
            colors=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        )


def plot_graph(title: str, titles: list[str], ys: list[str], x: str, datas: list[np.ndarray], save_path: str, colors: list[str] = None):
    num_plots = len(datas)
    cols = 2
    rows = math.ceil(num_plots / cols)

    plt.figure(figsize=(12, 4 * rows))
    plt.suptitle(title, fontsize=16, fontweight='bold')

    for i in range(num_plots):
        plt.subplot(rows, cols, i + 1)
        plot_color = colors[i % len(colors)] if colors else '#1f77b4'
        plt.plot(datas[i], alpha=0.8, color=plot_color)
        plt.title(titles[i])
        plt.xlabel(x)
        plt.ylabel(ys[i])
        plt.grid(True, alpha=0.4)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot '{title}' to {save_path}")

if __name__ == "__main__":
    env_config = Config('evaluate_env')
    agent_config = Config('layered_sac_agent')
    att_config = Config('sac_agent')

    env_config.load_from_file('src/envs/evaluate_env_config.json')
    agent_config.load_from_file('src/algorithms/sac/layered_config.json')
    att_config.load_from_file('src/algorithms/sac/sac_config.json')

    print(f"Evaluating models, {sys.argv[2:]}")

    run_evaluate(env_config, agent_config, sys.argv[2:], sys.argv[1], attitude_config=att_config)