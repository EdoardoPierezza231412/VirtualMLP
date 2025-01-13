import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback
from ot2_env_wrapper import OT2Env  # Custom environment wrapper
from clearml import Task  # Import ClearML's Task
import typing_extensions
import os
os.environ["WANDB_FORCE_HARDLINKS"] = "1"
import wandb


# ClearML integration
# 
# task = Task.init(
#     project_name='Mentor Group J/Group 0',  # Replace with your actual project name
#     task_name='RL Training Experiment1'    # Replace with a descriptive task name
#     )
# task.set_base_docker('deanis/2023y2b-rl:latest')  # Set the Docker image
# task.execute_remotely(queue_name="default")       # Execute remotely in the ClearML queue


os.environ['WANDB_API_KEY'] = '4068421fea4a91b66b033f55a01001d7badb12a6'

def main():
    # Argument parser for hyperparameters
    parser = argparse.ArgumentParser(description="Train PPO Model for OT2 Environment with W&B")
    parser.add_argument('--learning_rate', type=float, default=0.0003, help="Learning rate for PPO")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for PPO")
    parser.add_argument('--n_steps', type=int, default=2048, help="Number of steps per rollout")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--total_timesteps', type=int, default=1e6, help="Total timesteps for training")
    parser.add_argument('--max_steps', type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument('--wandb_project', type=str, default="OT2_RL_Project", help="W&B project name")
    args = parser.parse_args()

    # Initialize W&B
    config = {
        "policy_type": "MlpPolicy",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "gamma": args.gamma,
        "total_timesteps": args.total_timesteps,
    }
    run = wandb.init(
        project=args.wandb_project,
        config=config,
        sync_tensorboard=True,  # Sync TensorBoard logs
        monitor_gym=True,       # Auto-upload videos
        save_code=True,         # Save the training script
    )

    # Custom environment setup
    def make_env():
        env = OT2Env(max_steps=args.max_steps)
        env = Monitor(env)  # Record stats like rewards
        return env

    # Create vectorized environment for SB3
    env = DummyVecEnv([make_env])
     
    print("Observation space shape:", env.observation_space.shape)
    print("Action space shape:", env.action_space.shape)


    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",  # Log TensorBoard metrics
    )

    # Ensure the model save directory exists
    os.makedirs(f"models/{run.id}", exist_ok=True)

    # Create W&B callback
    wandb_callback = WandbCallback(
        model_save_freq=1000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    # Train and save models periodically
    timesteps = 100000  # Number of timesteps per chunk
    total_training_steps = int(args.total_timesteps)

    for i in range(total_training_steps // timesteps):
        print(f"Training chunk {i+1}, timesteps: {timesteps}")
        model.learn(
            total_timesteps=timesteps,
            callback=wandb_callback,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"runs/{run.id}",
        )
        # Save the model after each chunk
        model.save(f"models/{run.id}/{timesteps * (i + 1)}")
        print(f"Model saved at timestep {timesteps * (i + 1)}")

    print("Training complete!")

    # Save final model
    final_model_path = f"models/{run.id}/final_model.zip"
    model.save(final_model_path)
    artifact = wandb.Artifact(
        f"ppo_final_model_{run.id}",
        type="model",
        description="Final trained model",
    )
    artifact.add_file(final_model_path)
    wandb.log_artifact(artifact)
    print("Final model saved and logged to W&B.")

    # Finish W&B run
    run.finish()

    # Close environment
    env.close()

if __name__ == "__main__":
    main()

