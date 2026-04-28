import os
import math
import torch
import sys
import wandb

from src.algorithms.sac.sac_agent import SACAgent
from src.configs.config import Config
from src.envs.base_env import env_from_config
from src.utils.math import enu_to_geodetic


def export_batch_acmi(filename, step_idx, dt, npos, epos, alt, roll, pitch, yaw):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # 'w' overwrites the file on the first step, 'a' appends after that
    mode = 'w' if step_idx == 0 else 'a'
    current_time = step_idx * dt

    with open(filename, mode) as f:
        # Write the Tacview Header on the first frame
        if step_idx == 0:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")

        f.write(f"#{current_time:.2f}\n")

        num_agents = npos.shape[0]
        for i in range(num_agents):
            n = npos[i].item()
            e = epos[i].item()
            a = alt[i].item()

            # Convert ENU to Lat/Lon safely (scalar by scalar)
            lat, lon, h = enu_to_geodetic(e, n, a, 0, 0, 0)

            # Convert radians to degrees for Tacview
            r = math.degrees(roll[i].item())
            p = math.degrees(pitch[i].item())
            y_ang = math.degrees(yaw[i].item())

            # Assign a unique Hex ID to each aircraft
            obj_id = f"A{100 + i}"

            # Write the telemetry line
            if step_idx == 0:
                f.write(f"{obj_id},T={lon}|{lat}|{h}|{r}|{p}|{y_ang},Name=F-16 #{i+1},Type=Air+FixedWing\n")
            else:
                f.write(f"{obj_id},T={lon}|{lat}|{h}|{r}|{p}|{y_ang}\n")


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

    print("Starting evaluation loop")
    for model_idx in range(len(models)):
        print(f"Evaluating {models[model_idx]}")
        agent = agents[i]
        obs = env.reset()

        model_name = models[model_idx].split("/")[-1]
        for epoch in range(env_config('epochs')):
            with torch.no_grad():
                action = agent.act(obs)

            next_obs, reward, done, bad_done, timeout, info = env.step(action)

            # Export telemetry to Tacview ACMI format
            export_batch_acmi(
                filename=f"./logs/sac/v1/evaluation_replay_{model_name}.acmi",
                step_idx=epoch,
                dt=env.env.model.dt,
                npos=next_obs[:, 0],
                epos=next_obs[:, 1],
                alt=next_obs[:, 2],
                roll=next_obs[:, 3],
                pitch=next_obs[:, 4],
                yaw=next_obs[:, 5]
            )

            obs = next_obs

            if (done.float().mean() + bad_done.float().mean() + timeout.float().mean()) == 1.0: # All agents are done
                print(f"Agent failed, reason - done: {done.float().mean()}, bad_done: {bad_done.float().mean()}, timeout: {timeout.float().mean()}")
                break

            if epoch % 100 == 0:
                print(f"Epoch {epoch} / {env_config('epochs')}")

if __name__ == "__main__":
    env_config = Config('env')
    agent_config = Config('sac_agent')

    env_config.load_from_file('src/envs/evaluate_env_config.json')
    agent_config.load_from_file('src/algorithms/sac/sac_config.json')

    print(f"Evaluating models, {sys.argv[1:]}")

    run_evaluate(env_config, agent_config, sys.argv[1:])