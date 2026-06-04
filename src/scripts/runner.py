import wandb
import argparse

from src.scripts.train import train
from src.configs.config import Config

if __name__ == "__main__":
    env_config = Config('env')
    # agent_config = Config('td3_agent')
    agent_config = Config('sac_agent')

    env_config.load_from_file('src/envs/env_config.json')
    # agent_config.load_from_file('src/algorithms/td3/td3_config.json')
    agent_config.load_from_file('src/algorithms/sac/sac_config.json')

    # Parse command line arguments for logging to wandb
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
    parser.add_argument('--wandb', action='store_true', help="Whether to log training to wandb.")
    parser.add_argument('--data_path', type=str, default="/vol/bitbucket/rh1122/DaedalusProject/data/", help="Path to save training data and logs.")
    parser.add_argument('--load_model', type=str, default=None, help="Path to a saved model to load for training.")

    args = parser.parse_args()

    if args.wandb:
        print("Logging to wandb enabled.")
        with wandb.init(project="DaedalusProject", config={**env_config.data, **agent_config.data}) as run:
            train(env_config, agent_config, run, args.data_path, args.load_model)
    else:
        print("Logging to wandb disabled. Training will proceed without logging.")
        train(env_config, agent_config, None, args.data_path, args.load_model)