import os
import math
import torch
import sys
import wandb
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.sac.sac_agent import SACAgent
from src.configs.config import Config
from src.envs.base_env import env_from_config
from src.utils.math import enu_to_geodetic

def export_batch_acmi(filename, step_idx, dt, npos, epos, alt, roll, pitch, yaw, t_npos, t_epos, t_alt, mask=None):
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

            t_n = t_npos[i].item()
            t_e = t_epos[i].item()
            t_a = t_alt[i].item()
            t_lat, t_lon, t_h = enu_to_geodetic(t_e, t_n, t_a, 0, 0, 0)

            t_id = f"T{100 + i}"
            if step_idx == 0:
                f.write(f"{t_id},T={t_lon}|{t_lat}|{t_h},Name=Target #{i+1},Type=Waypoint,Color=Blue\n")
            else:
                f.write(f"{t_id},T={t_lon}|{t_lat}|{t_h}\n")


def run_evaluate(env_config: Config, agent_config: Config, models: list[str]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(models)

    # Init environment
    env = env_from_config(env_config, device=device)
    obs_dim = env.obs_dim
    action_dim = env.action_dim

    # Init all models
    algorithm = agent_config('name')
    if algorithm == "sac_agent":
        agents = [SACAgent(obs_dim, action_dim, 1, device, agent_config) for _ in range(len(models))]

        # Load .pt files into agents
        for i in range(len(models)):
            agents[i].load(models[i])
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    reward_history = []
    action_history = []

    path = "./logs/sac/v4/"

    print("Starting evaluation loop")
    for model_idx in range(len(models)):
        print(f"Evaluating {models[model_idx]}")
        agent = agents[model_idx]
        obs = env.reset()

        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

        model_name = models[model_idx].split("/")[-1]
        for epoch in range(env_config('epochs')):
            with torch.no_grad():
                action = agent.act(obs)

            next_obs, reward, done, bad_done, timeout, info = env.step(action)

            reward_history.append(reward_history[-1] if len(reward_history) > 0 else 0 + reward.squeeze().detach().cpu().numpy())
            action_history.append(action[1,:].squeeze().detach().cpu().numpy())

            mask |= done | bad_done | timeout

            raw_npos, raw_epos, raw_alt = env.env.model.get_position()
            roll, pitch, yaw = env.env.model.get_posture()

            # Pull the target coordinates from the task memory
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
        for i in range(action_matrix.shape[1]):
            plt.subplot(2, 2, i+1)
            plt.plot(action_matrix[:, i], alpha=0.4, label=f"Action {action_dimensions[i]}")
            plt.title(f"Action Dimension {i+1} Over Time\nModel: {model_name}")
            plt.xlabel("Simulation Steps")
            plt.ylabel("Control Signal")
            plt.legend()
            plt.grid(True, alpha=0.4)
        # Now also plot a time smoothed curve of the data
        plotted_steps = np.arange(action_matrix.shape[0])
        for i in range(action_matrix.shape[1]):
            smoothed = np.convolve(action_matrix[:, i], np.ones(100)/100, mode='valid')
            plt.subplot(2, 2, i+1)
            plt.plot(plotted_steps[:len(smoothed)], smoothed, color='red', label=f"Smoothed Action {action_dimensions[i]}")
            plt.legend()
        plt.tight_layout()
        plot_path = f"{path}evaluation_actions_subplots_{model_name}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved action subplot to {plot_path}")

if __name__ == "__main__":
    env_config = Config('env')
    agent_config = Config('sac_agent')

    env_config.load_from_file('src/envs/evaluate_env_config.json')
    agent_config.load_from_file('src/algorithms/sac/sac_config.json')

    print(f"Evaluating models, {sys.argv[1:]}")

    run_evaluate(env_config, agent_config, sys.argv[1:])