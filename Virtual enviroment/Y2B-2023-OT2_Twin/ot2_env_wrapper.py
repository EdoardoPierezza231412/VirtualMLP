import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action space: 3D movement in x, y, z directions
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

        # Define observation space: pipette position (3) + goal position (3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        # Keep track of the number of steps
        self.steps = 0

    def reset(self, seed=None, options=None):
        print("Reset called")
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation
        observation = self.sim.reset(num_agents=1)
        print(f"Reset Observation: {observation}")

        # Dynamically find the robot ID key
        robot_id_key = next(iter(observation.keys()))

        # Extract pipette position
        pipette_position = np.array(observation[robot_id_key]['pipette_position'], dtype=np.float32)

        # Set a random goal position
        self.goal_position = np.random.uniform(
            low=[-0.164, -0.171, 0.169],
            high=[0.253, 0.22, 0.29],
            size=(3,)
        ).astype(np.float32)  # Ensure goal position is float32

        # Combine pipette position with goal position
        full_observation = np.concatenate((pipette_position, self.goal_position), axis=0).astype(np.float32)  # Ensure full observation is float32

        self.steps = 0
        return full_observation, {}


    def step(self, action):
        print(f"Step called with action: {action}")

        # Append a drop command (0) to the action
        action_with_drop = list(action) + [0.0]

        # Execute action in the simulation
        observation = self.sim.run([action_with_drop])

        # Dynamically find the robot ID key
        robot_id_key = next(iter(observation.keys()))

        # Extract pipette position
        pipette_position = np.array(observation[robot_id_key]['pipette_position'], dtype=np.float32)

        # Combine pipette position with goal position
        full_observation = np.concatenate((pipette_position, self.goal_position), axis=0)

        # Compute reward (negative distance to goal)
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        reward = -float(distance_to_goal)  # Ensure reward is a native float

        # Check if the task is complete
        terminated = bool(distance_to_goal < 0.01)  # Ensure terminated is a boolean

        # Check if the episode is truncated
        truncated = bool(self.steps >= self.max_steps)  # Ensure truncated is a boolean

        self.steps += 1
        return full_observation, reward, terminated, truncated, {}


    def render(self, mode='human'):
        # Optional: Add rendering logic if needed
        pass

    def close(self):
        # Clean up the simulation
        self.sim.close()