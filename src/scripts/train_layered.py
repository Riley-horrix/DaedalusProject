import torch
import time
import wandb
import numpy as np
from collections import deque

from src.envs.base_env import env_from_config
from src.utils.replay_buffer import ReplayBuffer
from src.algorithms.sac.layered_agent import LayeredSACAgent
from src.configs.config import Config
from src.utils.logging import LoggingStruct

def train_layered(env_config: Config, agent_config: Config, attitude_config: Config, run: wandb.Run | None, data_path: str, load_model: str | None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_time = round(time.time())

    # Init environment
    env = env_from_config(env_config, device=device)
    obs_dim = env.obs_dim
    num_envs = env_config('num_envs')

    outer_action_dim = 4

    # Init Layered Agent
    print("Initializing Layered SAC Agent...")
    agent = LayeredSACAgent(obs_dim, outer_action_dim, num_envs, device, agent_config, attitude_config)

    inner_model_path = agent_config('inner_model_path', None)
    if inner_model_path is not None:
        print(f"Loading pre-trained inner attitude agent from {inner_model_path}...")
        agent.load_inner_agent(inner_model_path)
    else:
        print("WARNING: No pre-trained inner agent provided. The outer agent will struggle to learn!")

    if load_model is not None:
        print(f"Loading outer model from {load_model}...")
        agent.load(load_model)

    buffer = ReplayBuffer(obs_dim, outer_action_dim, env_config('buffer_capacity'), num_envs, device)

    log = LoggingStruct()

    print("Starting training loop")
    obs = env.reset()

    current_episode_rewards = torch.zeros(num_envs, dtype=torch.float32, device=device)

    recent_scores = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    recent_crashes = deque(maxlen=100)
    recent_timeouts = deque(maxlen=100)

    before_epoch_time = time.time()

    print(f"Observation space: {obs_dim}, Outer Action space: {outer_action_dim}")
    obs_mean = torch.zeros(obs_dim, device=device)
    obs_var = torch.ones(obs_dim, device=device)

    for epoch in range(env_config('epochs')):

        physical_action = agent.act(obs)

        obs_mean = 0.99 * obs_mean + 0.01 * obs.mean(dim=0)
        obs_var = 0.99 * obs_var + 0.01 * obs.var(dim=0)

        next_obs, reward, done, bad_done, timeout, info = env.step(physical_action)

        current_episode_rewards += reward

        valid_mask = ~(torch.isnan(next_obs).any(dim=-1) | torch.isinf(next_obs).any(dim=-1))
        episode_ends = done | bad_done | timeout | ~valid_mask

        if episode_ends.any():
            finished_scores = current_episode_rewards[episode_ends].cpu().numpy()
            recent_scores.extend(finished_scores)

            recent_successes.extend(done[episode_ends].cpu().numpy())
            recent_crashes.extend((bad_done | ~valid_mask)[episode_ends].cpu().numpy())
            recent_timeouts.extend(timeout[episode_ends].cpu().numpy())

            current_episode_rewards[episode_ends] = 0.0

        real_terminal_signal = done | bad_done | ~valid_mask
        buffer.push(obs, agent.last_outer_action, reward, next_obs, real_terminal_signal)

        obs = torch.where(valid_mask.unsqueeze(-1), next_obs, torch.zeros_like(next_obs))

        if epoch > 10:
            agent.update(buffer, 10, log)

        # 7. Logging (Every 100 epochs to reduce console/network spam)
        if (epoch + 1) % 100 == 0:
            now_epoch_time = time.time()
            epoch_time = (now_epoch_time - before_epoch_time) / 100
            before_epoch_time = now_epoch_time

            mean_var_dict = {f"obs_mean/{i}": obs_mean[i].item() for i in range(obs_dim)}
            mean_var_dict.update({f"obs_var/{i}": obs_var[i].item() for i in range(obs_dim)})

            # Safely compute means
            avg_score = np.mean(recent_scores) if len(recent_scores) > 0 else 0.0
            win_rate = np.mean(recent_successes) if len(recent_successes) > 0 else 0.0
            crash_rate = np.mean(recent_crashes) if len(recent_crashes) > 0 else 0.0
            timeout_rate = np.mean(recent_timeouts) if len(recent_timeouts) > 0 else 0.0

            if run is not None:
                log_dict = {
                    "time/average_epoch_time": epoch_time,
                    "env/episodic_reward": avg_score,
                    "env/success_rate": win_rate,
                    "env/crash_rate": crash_rate,
                    "env/timeout_rate": timeout_rate,
                }
                log_dict.update(log.log)
                log_dict.update(mean_var_dict)
                run.log(log_dict, step=epoch)
            else:
                print(f"Epoch {epoch} | Reward: {avg_score:.2f} | Win: {win_rate:.2%} | Crash: {crash_rate:.2%} | Timeout: {timeout_rate:.2%} | Epoch Time: {epoch_time:.4f}s")

        # 8. Save the model
        if epoch % 10000 == 0 and epoch > 0:
            print("Saving model...")
            run_name = wandb.run.name if wandb.run is not None else f"{start_time}"
            path = f"{data_path}{agent_config('name')}_{run_name}_recent.pt"
            agent.save(path)
            if run is not None:
                run.save(path)
                wandb.save(path)
            print("Model Saved.")