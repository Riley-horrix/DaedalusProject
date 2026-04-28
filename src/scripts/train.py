# train.py
import torch
import time
import wandb

from src.envs.base_env import env_from_config
from src.utils.replay_buffer import ReplayBuffer
from src.algorithms.sac.sac_agent import SACAgent
from src.configs.config import Config
from src.utils.logging import LoggingStruct


data_route = "/vol/bitbucket/rh1122/DaedalusProject/data/"


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
        print(f"Episode {episode} / {env_config('episodes')} starting...")
        # Reset environment and replay buffer at the start of each episode
        obs = env.reset()
        epoch_time = 0

        for step in range(env_config('episode_length')):
            epoch_time_temp = time.time()
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

            # Log metrics every 10 steps
            if (step + 1) % 10 == 0:
                if run is not None:
                    log_dict = {
                        "time/env_step_time": env_step_time,
                        "time/agent_update_time": agent_update_time,
                        "time/epoch_time": epoch_time,
                        "env/reward": reward.mean().item(),
                        "env/done": done.float().mean().item(),
                        "env/bad_done": bad_done.float().mean().item(),
                        "env/timeout": timeout.float().mean().item()
                    }
                    log_dict.update(log.log)
                    run.log(log_dict, step=(episode * env_config('episode_length') + step))
                else:
                    print(f"Episode {episode}, Step {step}, Reward: {reward.mean().item():.2f}, Done: {done.float().mean().item():.2f}, Bad Done: {bad_done.float().mean().item():.2f}, Timeout: {timeout.float().mean().item():.2f}, Env Step Time: {env_step_time:.4f}s, Agent Update Time: {agent_update_time:.4f}s, Epoch Time: {epoch_time:.4f}s")

            epoch_time = time.time() - epoch_time_temp

            # If all environments are done, break the loop and start a new episode
            if (done | bad_done | timeout).float().mean().item() == 1.0:
                print(f"Episode {episode} completed.")
                break


        # Save the model at the end of each 10 episodes
        print("Episode Complete.")
        if episode % 10 == 0:
            print("Saving model...")
            path = f"{data_route}{agent_config('name')}_episode_{episode}.pt"
            agent.save(path)
            if run is not None:
                run.save(path)
        print("Model Saved.")