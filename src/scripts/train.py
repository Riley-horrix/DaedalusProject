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

    obs = env.reset()

    reward_saved = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    done_saved = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    bad_done_saved = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    timeout_saved = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    before_epoch_time = time.time()

    for epoch in range(env_config('epochs')):
        with torch.no_grad():
            action = agent.act(obs)

        next_obs, reward, done, bad_done, timeout, info = env.step(action)

        # Track done/bad_done/timeout for logging
        reward_saved += reward
        done_saved |= done
        bad_done_saved |= bad_done
        timeout_saved |= timeout

        # Crash Filtering
        valid_mask = ~(torch.isnan(next_obs).any(dim=-1) | torch.isinf(next_obs).any(dim=-1))
        real_done = done | bad_done | ~valid_mask

        # Store transition (Only store valid ones, or zero out broken ones)
        buffer.push(obs, action, reward, next_obs, real_done)

        obs = next_obs

        # Allow agent to train every 10 steps
        if epoch % 10 == 0 and buffer.size >= agent_config('batch_size') * 100:
            agent.update(buffer, 10, log)

        # Log metrics every 100 steps
        if (epoch + 1) % 100 == 0:
            # Calculate epoch time
            now_epoch_time = time.time()
            epoch_time = (now_epoch_time - before_epoch_time) / 100
            before_epoch_time = now_epoch_time

            if run is not None:
                log_dict = {
                    "time/average_epoch_time": epoch_time,
                    "env/reward": (reward_saved / 100).mean().item(),
                    "env/done": done_saved.float().mean().item(),
                    "env/bad_done": bad_done_saved.float().mean().item(),
                    "env/timeout": timeout_saved.float().mean().item()
                }
                log_dict.update(log.log)
                run.log(log_dict, step=epoch)
            else:
                print(f"Epoch {epoch}, Reward: {reward.mean().item():.2f}, Done: {done_saved.float().mean().item():.2f}, Bad Done: {bad_done_saved.float().mean().item():.2f}, Timeout: {timeout_saved.float().mean().item():.2f}, Epoch Time: {epoch_time:.4f}s")

            # Reset done/bad_done/timeout trackers
            done_saved[:] = False
            bad_done_saved[:] = False
            timeout_saved[:] = False

        # Save the model after every 10k steps
        print("Episode Complete.")
        if epoch % 10000 == 0:
            print("Saving model...")
            path = f"{data_route}{agent_config('name')}_epoch_{epoch//1000}k.pt"
            agent.save(path)
            if run is not None:
                run.save(path)
            print("Model Saved.")