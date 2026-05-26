import wandb
import argparse
from src.configs.config import Config
from train_layered import train_layered


if __name__ == "__main__":
    env_config = Config('layered_env')
    env_config.load_from_file('src/env/layered_env_config.json')

    agent_config = Config('layered_sac')
    agent_config.load_from_file('src/algorithms/sac/layered_config.json')

    attitude_config = Config('attitude')
    attitude_config.load_from_file('src/algorithms/sac/attitude_config.json')

    # Parse command line arguments for logging to wandb
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
    parser.add_argument('--wandb', action='store_true', help="Whether to log training to wandb.")
    parser.add_argument('--data_path', type=str, default="/vol/bitbucket/rh1122/DaedalusProject/data/", help="Path to save training data and logs.")
    parser.add_argument('--load_model', type=str, default=None, help="Path to a saved model to load for training.")

    args = parser.parse_args()

    if args.wandb:
        print("Logging to wandb enabled.")
        with wandb.init(project="DaedalusProject", config={**env_config.data, **agent_config.data}) as run:
            train_layered(
                env_config=env_config,
                agent_config=agent_config,
                attitude_config=attitude_config,
                run=run,
                data_path=args.data_path,
                load_model=args.load_model
            )
    else:
        print("Logging to wandb disabled. Training will proceed without logging.")
        train_layered(
            env_config=env_config,
            agent_config=agent_config,
            attitude_config=attitude_config,
            run=None,
            data_path=args.data_path,
            load_model=args.load_model
        )
