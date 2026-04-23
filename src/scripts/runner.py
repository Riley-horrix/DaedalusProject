import wandb
import argparse

from src.scripts.train import train
from src.configs.config import Config

if __name__ == "__main__":
    env_config = Config('env')
    agent_config = Config('sac_agent')

    env_config.load_from_file('src/envs/env_config.json')
    agent_config.load_from_file('src/algorithms/sac/sac_config.json')

    # Parse command line arguments for logging to wandb
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
    parser.add_argument('--wandb', action='store_true', help="Whether to log training to wandb.")
    args = parser.parse_args()

    if args.wandb:
        print("Logging to wandb enabled.")
        with wandb.init(project="DaedalusProject", config={**env_config.data, **agent_config.data}) as run:
            train(env_config, agent_config, run)
    else:
        print("Logging to wandb disabled. Training will proceed without logging.")
        train(env_config, agent_config, None)