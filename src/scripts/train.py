# train.py
import torch
import time
import wandb

from src.envs.base_env import env_from_config
from src.utils.replay_buffer import ReplayBuffer
from src.algorithms.sac.sac_agent import SACAgent
from src.configs.config import Config
from src.utils.logging import LoggingStruct

def train(env_config: Config, agent_config: Config, run: wandb.Run | None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Init environment
    env = env_from_config(env_config, device=device)
    obs_dim = env.obs_dim
    action_dim = env.action_dim

    # Init buffer
    buffer = ReplayBuffer(obs_dim, action_dim, env_config('buffer_capacity'), env_config('num_envs'), device)

    # Init agent
    algorithm = agent_config('name')
    if algorithm == "sac_agent":
        agent = SACAgent(obs_dim, action_dim, env_config('num_envs'), device, agent_config)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Init logging
    log = LoggingStruct()

    print("Starting training loop")

    for episode in range(env_config('episodes')):
        # Reset environment and replay buffer at the start of each episode
        obs = env.reset()
        buffer.clear()

        for step in range(env_config('episode_length')):
            epoch_time = time.time()
            with torch.no_grad():
                action = agent.act(obs)

            env_step_time = time.time()
            next_obs, reward, done, bad_done, timeout, info = env.step(action)
            env_step_time = time.time() - env_step_time

            # Crash Filtering (Preventing NaNs from entering the buffer)
            valid_mask = ~(torch.isnan(next_obs).any(dim=-1) | torch.isinf(next_obs).any(dim=-1))
            real_done = done | ~valid_mask

            # Store transition (Only store valid ones, or zero out broken ones)
            buffer.push(obs, action, reward, next_obs, real_done)

            obs = next_obs

            # Allow agent to train
            agent_update_time = time.time()
            agent.update(buffer, 1, log)
            agent_update_time = time.time() - agent_update_time
            epoch_time = time.time() - epoch_time

            # Log to WandB
            if run is not None:
                log_dict = {
                    "time/env_step_time": env_step_time,
                    "time/agent_update_time": agent_update_time,
                    "time/epoch_time": epoch_time,
                    "env/reward": reward.mean().item(),
                    "env/done": done.float().mean().item(),
                    "env/bad_done": bad_done.float().mean().item(),
                    "env/timeout": timeout.float().mean().item(),
                }
                log_dict.update(log.log)
                run.log(log_dict)
            else:
                print(f"Episode {episode}, Step {step}, Reward: {reward.mean().item():.2f}, Done: {done.float().mean().item():.2f}, Bad Done: {bad_done.float().mean().item():.2f}, Timeout: {timeout.float().mean().item():.2f}, Env Step Time: {env_step_time:.4f}s, Agent Update Time: {agent_update_time:.4f}s, Epoch Time: {epoch_time:.4f}s")


