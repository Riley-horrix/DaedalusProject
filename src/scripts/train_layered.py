# train_layered.py
import torch
import time
import wandb

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

    # Note: The environment expects 4 control surfaces (el, ail, rud, thrust).
    # Coincidentally, our outer SAC agent also outputs 4 targets (pitch, roll, beta, velocity).
    # We will use the outer agent's action dimension for the Replay Buffer.
    outer_action_dim = 4

    # Init Layered Agent
    print("Initializing Layered SAC Agent...")
    agent = LayeredSACAgent(obs_dim, outer_action_dim, env_config('num_envs'), device, agent_config, attitude_config)

    inner_model_path = agent_config('inner_model_path', None)
    if inner_model_path is not None:
        print(f"Loading pre-trained inner attitude agent from {inner_model_path}...")
        agent.load_inner_agent(inner_model_path)
    else:
        print("WARNING: No pre-trained inner agent provided. The outer agent will struggle to learn if the inner agent cannot fly the plane!")

    if load_model is not None:
        print(f"Loading outer model from {load_model}...")
        agent.load(load_model)

    buffer = ReplayBuffer(obs_dim, outer_action_dim, env_config('buffer_capacity'), env_config('num_envs'), device)

    # Init logging
    log = LoggingStruct()

    print("Starting layered training loop")

    obs = env.reset()

    reward_saved = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    done_saved = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    bad_done_saved = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    timeout_saved = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    before_epoch_time = time.time()

    print(f"Observation space: {obs_dim}, Outer Action space: {outer_action_dim}")
    obs_mean = torch.zeros(obs_dim, device=device)
    obs_var = torch.ones(obs_dim, device=device)

    for epoch in range(env_config('epochs')):
        with torch.no_grad():
            physical_action = agent.act(obs)

        obs_mean = 0.99 * obs_mean + 0.01 * obs.mean(dim=0)
        obs_var = 0.99 * obs_var + 0.01 * obs.var(dim=0)

        next_obs, reward, done, bad_done, timeout, info = env.step(physical_action)

        reward_saved += reward
        done_saved |= done
        bad_done_saved |= bad_done
        timeout_saved |= timeout

        valid_mask = ~(torch.isnan(next_obs).any(dim=-1) | torch.isinf(next_obs).any(dim=-1))
        real_done = done | bad_done | ~valid_mask

        agent.attitude_agent.reset_pid_states(real_done)
        buffer.push(obs, agent.last_outer_action, reward, next_obs, real_done)

        obs = next_obs

        if epoch > 1000 and epoch % 10 == 0:
            agent.update(buffer, 10, log)

        if (epoch + 1) % 100 == 0:
            # Calculate epoch time
            now_epoch_time = time.time()
            epoch_time = (now_epoch_time - before_epoch_time) / 100
            before_epoch_time = now_epoch_time

            mean_var_dict = {f"obs_mean/{i}": obs_mean[i].item() for i in range(obs_dim)}
            mean_var_dict.update({f"obs_var/{i}": obs_var[i].item() for i in range(obs_dim)})

            if run is not None:
                log_dict = {
                    "time/average_epoch_time": epoch_time,
                    "env/reward": (reward_saved / 100).mean().item(),
                    "env/done": done_saved.float().mean().item(),
                    "env/bad_done": bad_done_saved.float().mean().item(),
                    "env/timeout": timeout_saved.float().mean().item(),
                    "velocity/total_error": obs[:, -1].abs().sum().item(),
                    "velocity/first_error": torch.abs(obs[0, -1]).item(),
                }
                log_dict.update(log.log)
                log_dict.update(mean_var_dict)
                run.log(log_dict, step=epoch)
            else:
                print(f"Epoch {epoch}, Reward: {reward.mean().item():.2f}, Done: {done_saved.float().mean().item():.2f}, Bad Done: {bad_done_saved.float().mean().item():.2f}, Epoch Time: {epoch_time:.4f}s")

            reward_saved[:] = 0.0
            done_saved[:] = False
            bad_done_saved[:] = False
            timeout_saved[:] = False

        if epoch % 10000 == 0:
            print("Saving model...")
            run_name = wandb.run.name if wandb.run is not None else f"{start_time}"
            path = f"{data_path}{agent_config('name')}_{run_name}_recent.pt"

            agent.save(path)
            wandb.save(path)

            if run is not None:
                run.save(path)
            print("Model Saved.")