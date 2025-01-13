import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from ot2_env_wrapper import OT2Env  # Custom environment wrapper

def main():
    # Argument parser for hyperparameters
    parser = argparse.ArgumentParser(description="Train PPO Model for OT2 Environment")
    parser.add_argument('--learning_rate', type=float, default=0.0003, help="Learning rate for PPO")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for PPO")
    parser.add_argument('--n_steps', type=int, default=2048, help="Number of steps per rollout")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--total_timesteps', type=int, default=500, help="Total timesteps for training")
    parser.add_argument('--max_steps', type=int, default=100, help="Maximum steps per episode")
    parser.add_argument('--output_dir', type=str, default="./models", help="Directory to save models")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

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
        tensorboard_log=f"{args.output_dir}/tensorboard",  # Log TensorBoard metrics
    )

    # Train and save models periodically
    timesteps = 100  # Number of timesteps per chunk
    total_training_steps = int(args.total_timesteps)

    for i in range(total_training_steps // timesteps):
        print(f"Training chunk {i+1}, timesteps: {timesteps}")
        model.learn(
            total_timesteps=timesteps,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"run_{i+1}",
        )

        # Save the model after each chunk
        chunk_model_path = os.path.join(args.output_dir, f"model_{timesteps * (i + 1)}.zip")
        model.save(chunk_model_path)
        print(f"Model saved at timestep {timesteps * (i + 1)} to {chunk_model_path}")

    print("Training complete!")

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Close environment
    env.close()

if __name__ == "__main__":
    main()