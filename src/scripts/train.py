# train.py
import torch
import time
import wandb

from src.envs.base_env import env_from_config
from src.utils.replay_buffer import ReplayBuffer
from src.algorithms.sac.sac_agent import SACAgent
from src.algorithms.sac.attitude_agent import AttitudeAgent
from src.algorithms.td3.td3_agent import TD3Agent
from src.configs.config import Config
from src.utils.logging import LoggingStruct

from src.scripts.evaluate import run_evaluate

def train(env_config: Config, agent_config: Config, run: wandb.Run | None, data_path: str, load_model: str | None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_time = round(time.time())

    # Init environment
    env = env_from_config(env_config, device=device)
    obs_dim = env.obs_dim
    action_dim = env.action_dim

    num_envs = env_config('num_envs')

    # Init agent
    algorithm = agent_config('name')
    if algorithm == "sac_agent":
        obs_dim = 11
        agent = SACAgent(obs_dim, action_dim, num_envs, device, agent_config)
    elif algorithm == "td3_agent":  # <-- Added TD3 Initialization
        obs_dim = 14
        agent = TD3Agent(obs_dim, action_dim, num_envs, device, agent_config)
    elif algorithm == "attitude_agent":
        obs_dim = 14
        agent = AttitudeAgent(obs_dim, action_dim, num_envs, device, agent_config)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Init buffer
    buffer = ReplayBuffer(obs_dim, action_dim, env_config('buffer_capacity'), num_envs, device)

    # Init logging
    log = LoggingStruct()

    # Load pre-trained model if specified
    if load_model is not None:
        print(f"Loading model from {load_model}...")
        agent.load(load_model)

    print("Starting training loop")

    obs = env.reset()

    reward_saved = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    done_stat = 0
    bad_done_stat = 0
    timeout_stat = 0

    before_epoch_time = time.time()

    print(f"Observation space: {obs_dim}, Action space: {action_dim}")
    obs_mean = torch.zeros(obs_dim, device=device)
    obs_var = torch.ones(obs_dim, device=device)

    # Define warmup steps for TD3 (defaulting to 10k as per the paper)
    warmup_steps = env_config('warmup_steps', 10000)

    for epoch in range(env_config('epochs')):
        with torch.no_grad():
            # <-- Action Selection Logic Updated for TD3 -->
            if algorithm == "td3_agent":
                # Pure exploration for early epochs
                if epoch < warmup_steps and load_model is None:
                    # Generate uniformly random actions in [-1, 1]
                    action = torch.rand((env.num_envs, action_dim), device=device) * 2.0 - 1.0
                else:
                    # Deterministic action + Gaussian noise
                    action = agent.act(obs, add_noise=True)
            else:
                action = agent.act(obs)

        # Update mean and std of observations for logging
        obs_mean = 0.99 * obs_mean + 0.01 * obs.mean(dim=0)
        obs_var = 0.99 * obs_var + 0.01 * obs.var(dim=0)

        next_obs, reward, done, bad_done, timeout, info = env.step(action)

        # Track done/bad_done/timeout for logging
        reward_saved += reward
        done_stat += done.sum().item()
        bad_done_stat += bad_done.sum().item()
        timeout_stat += timeout.sum().item()

        total = done_stat + bad_done_stat + timeout_stat

        # Crash Filtering
        valid_mask = ~(torch.isnan(next_obs).any(dim=-1) | torch.isinf(next_obs).any(dim=-1))
        real_done = done | bad_done | ~valid_mask

        episodic_reward = (reward_saved * real_done.float()).sum().item() / real_done.float().sum().item() if real_done.float().sum().item() > 0 else 0.0
        reward_saved = reward_saved * (~real_done).float()

        if algorithm == "attitude_agent":
            agent.reset_pid_states(real_done)

        # Store transition (Only store valid ones, or zero out broken ones)
        buffer.push(obs, action, reward, next_obs, real_done)

        obs = torch.where(valid_mask.unsqueeze(-1), next_obs, torch.zeros_like(next_obs))

        if epoch > 10:
            agent.update(buffer, num_envs // 10, log)

        # Calculate epoch time
        now_epoch_time = time.time()
        epoch_time = (now_epoch_time - before_epoch_time)
        before_epoch_time = now_epoch_time

        mean_var_dict = {f"obs_mean/{i}": obs_mean[i].item() for i in range(obs_dim)}
        mean_var_dict.update({f"obs_var/{i}": obs_var[i].item() for i in range(obs_dim)})

        if run is not None:
            log_dict = {
                "time/average_epoch_time": epoch_time,
                "env/episodic_reward": episodic_reward,
                "env/done": done_stat / total if total > 0 else 0.0,
                "env/bad_done": bad_done_stat / total if total > 0 else 0.0,
                "env/timeout": timeout_stat / total if total > 0 else 0.0,
                "velocity/total_error": obs[:, -1].abs().sum().item(),
                "velocity/first_error": torch.abs(obs[0, -1]).item(),
            }
            log_dict.update(log.log)
            log_dict.update(mean_var_dict)
            run.log(log_dict, step=epoch)
        else:
            print(f"Epoch {epoch}, Reward: {episodic_reward:.2f}, Done: {done_stat / total if total > 0 else 0.0}, Bad Done: {bad_done_stat / total if total > 0 else 0.0}, Timeout: {timeout_stat / total if total > 0 else 0.0}, Epoch Time: {epoch_time:.4f}s")

        # Reset done/bad_done/timeout trackers
        done_stat = 0
        bad_done_stat = 0
        timeout_stat = 0

        # Save the model after every 50 steps
        if epoch % 50 == 0:
            print("Saving model...")
            run_name = wandb.run.name if wandb.run is not None else f"{start_time}"
            path = f"{data_path}{agent_config('name')}_{run_name}_recent.pt"
            agent.save(path)
            if run is not None:
                run.save(path)
                wandb.save(path)
            print("Model Saved.")