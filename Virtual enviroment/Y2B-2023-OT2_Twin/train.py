import argparse
from stable_baselines3 import PPO
from ot2_env_wrapper import OT2Env  # Custom environment from Task 10
from clearml import Task


# ClearML integration
task = Task.init(
    project_name='Mentor Group J/Group 0',  # Replace with your actual project name
    task_name='RL Training Experiment1'    # Replace with a descriptive task name
)
task.set_base_docker('deanis/2023y2b-rl:latest')  # Set the Docker image
task.execute_remotely(queue_name="default")       # Execute remotely in the ClearML queue


def main():
    # Parse command-line arguments for hyperparameters
    parser = argparse.ArgumentParser(description="Train PPO Model for OT2Env")
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate for PPO')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for PPO')
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps per update')
    parser.add_argument('--total_timesteps', type=int, default=1e6, help='Total timesteps for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0, 1, 2)')
    args = parser.parse_args()

    # Initialize the custom OT2 environment
    env = OT2Env(render=False, max_steps=1000)

    # Define the PPO model with parsed hyperparameters
    model = PPO(
        "MlpPolicy",         # Policy type
        env,                 # Environment
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        verbose=args.verbose
    )

    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=args.total_timesteps)
    print("Training complete!")

    # Save the trained model
    model.save("ppo_ot2_model")
    print("Model saved as ppo_ot2_model.zip")

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
