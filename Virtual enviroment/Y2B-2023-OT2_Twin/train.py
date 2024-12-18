import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
import wandb
from ot2_env_wrapper import OT2Env  # Your custom environment wrapper
from clearml import Task  # Import ClearML's Task

# ClearML integration
task = Task.init(
    project_name='Mentor Group J/Group 0',  # Replace with your actual project name
    task_name='RL Training Experiment1'    # Replace with a descriptive task name
)
task.set_base_docker('deanis/2023y2b-rl:latest')  # Set the Docker image
task.execute_remotely(queue_name="default")       # Execute remotely in the ClearML queue

def main():
    # Argument parser for hyperparameters
    parser = argparse.ArgumentParser(description="Train PPO Model for OT2 Environment with W&B")
    parser.add_argument('--learning_rate', type=float, default=0.0003, help="Learning rate for PPO")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for PPO")
    parser.add_argument('--n_steps', type=int, default=2048, help="Number of steps per rollout")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--total_timesteps', type=int, default=1e6, help="Total timesteps for training")
    parser.add_argument('--max_steps', type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument('--render', action='store_true', help="Render the environment during training")
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
        env = OT2Env(render=args.render, max_steps=args.max_steps)
        env = Monitor(env)  # Record stats like rewards
        return env

    # Create vectorized environment for SB3
    env = DummyVecEnv([make_env])

    # Optional: Record training videos
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 2000 == 0,
        video_length=200,
    )

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

    # Train the model in chunks with periodic saving
    total_training_steps = args.total_timesteps
    save_frequency = 100000  # Define the frequency to save the model in timesteps

    for i in range(total_training_steps // save_frequency):
        print(f"Training chunk {i+1}, timesteps: {save_frequency}")
        model.learn(
            total_timesteps=save_frequency,
            callback=WandbCallback(
                gradient_save_freq=100,               # Log gradients every 100 steps
                model_save_path=f"models/{run.id}",  # Save models periodically
                verbose=2,                           # Verbosity level for W&B
            ),
            progress_bar=True,
            reset_num_timesteps=False,  # Continue counting timesteps
            tb_log_name=f"runs/{run.id}",
        )
        # Save the model after each chunk
        model.save(f"models/{run.id}/{save_frequency * (i + 1)}")
        print(f"Model saved at timesteps: {save_frequency * (i + 1)}")

    print("Training complete!")

    # Finish W&B run
    run.finish()

    # Save final model
    model.save(f"models/{run.id}/final_model")
    print(f"Final model saved as models/{run.id}/final_model.zip")

    # Close environment
    env.close()

if __name__ == "__main__":
    main()

