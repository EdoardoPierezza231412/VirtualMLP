import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
from PIDController import PIDController


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=10000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action space: motor velocities for x, y, and z
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

        # Define observation space: pipette position (3) + goal position (3) + velocity (3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        # Initialize PID controllers for x, y, and z axes
        self.pid_x = PIDController(kp=1.0, ki=0.0, kd=0.1)
        self.pid_y = PIDController(kp=1.0, ki=0.0, kd=0.1)
        self.pid_z = PIDController(kp=1.0, ki=0.0, kd=0.1)

        # Keep track of steps
        self.steps = 0

    def get_current_position(self):
        """
        Retrieve the current position of the robot's pipette.

        Returns:
        - current_position: A numpy array [x, y, z] representing the current pipette position.
        """
        observation = self.sim.run([[0.0, 0.0, 0.0, 0.0]])
        robot_id_key = next(iter(observation.keys()))
        current_position = observation[robot_id_key]["pipette_position"]
        return np.array(current_position, dtype=np.float32)

    def set_goal(self, goal_position):
        """
        Set a custom goal position for the robot.

        Parameters:
        - goal_position: A numpy array [x, y, z].
        """
        self.goal_position = np.array(goal_position, dtype=np.float32)
        print(f"Goal position set to: {self.goal_position}")

    def image(self):
        """
        Capture the current plate image from the simulation.

        Returns:
        - image_path: The path to the saved image of the current plate.
        """
        try:
            image_path = self.sim.get_plate_image()
            if image_path:
                print(f"Image captured and saved at: {image_path}")
                return image_path
            else:
                raise ValueError("Failed to capture the plate image from the simulation.")
        except Exception as e:
            raise RuntimeError(f"Error while capturing image: {e}")

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset simulation and set a random goal position
        observation = self.sim.reset(num_agents=1)
        robot_id_key = next(iter(observation.keys()))
        pipette_position = np.array(
            observation[robot_id_key]["pipette_position"], dtype=np.float32
        )
        self.goal_position = np.random.uniform(
            low=[-0.164, -0.171, 0.169],
            high=[0.253, 0.22, 0.29],
            size=(3,),
        ).astype(np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Initialize velocity

        # Combine pipette position, velocity, and goal position into one observation
        full_observation = np.concatenate((pipette_position, self.velocity, self.goal_position), axis=0)

        # Reset step counter
        self.steps = 0

        # Debug log
        print(f"Reset: Pipette Position {pipette_position}, Goal Position {self.goal_position}")

        return full_observation, {}

    def step(self, _):
        print(f"Step {self.steps + 1} called.")

        # Get the current pipette position
        observation = self.sim.run([[0.0, 0.0, 0.0, 0.0]])
        robot_id_key = next(iter(observation.keys()))
        pipette_position = np.array(observation[robot_id_key]["pipette_position"], dtype=np.float32)

        # Compute errors for each axis
        error_x = self.goal_position[0] - pipette_position[0]
        error_y = self.goal_position[1] - pipette_position[1]
        error_z = self.goal_position[2] - pipette_position[2]

        # Compute velocities using PID controllers
        velocity_x = self.pid_x.compute(error_x)
        velocity_y = self.pid_y.compute(error_y)
        velocity_z = self.pid_z.compute(error_z)

        # Create an action array from PID outputs
        action = np.clip([velocity_x, velocity_y, velocity_z], -1.0, 1.0).astype(np.float32)

        # Execute action in the simulation
        action_with_drop = list(action) + [0.0]
        observation = self.sim.run([action_with_drop])

        # Extract pipette position again after taking the action
        pipette_position = np.array(observation[robot_id_key]["pipette_position"], dtype=np.float32)

        # Compute velocity for the observation
        self.velocity = pipette_position - self.velocity

        # Combine pipette position, velocity, and goal position
        full_observation = np.concatenate((pipette_position, self.velocity, self.goal_position), axis=0)

        # Compute distance to goal
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)

        # Reward function
        max_distance = np.linalg.norm(np.array([0.253, 0.22, 0.29]) - np.array([-0.164, -0.171, 0.169]))
        reward = -(distance_to_goal / max_distance)  # Normalize distance-based reward

        # Add a success bonus
        if distance_to_goal < 0.001:
            reward += 20.0  # Increased bonus for reaching the goal

        # Penalize each step
        reward -= 0.005

        # Check if the task is complete
        terminated = bool(distance_to_goal < 0.001)
        if terminated:
            print(f"Terminated: Reached goal at step {self.steps + 1}")

        # Check if the episode is truncated
        truncated = bool(self.steps >= self.max_steps)
        if truncated:
            print(f"Truncated: Exceeded max steps at step {self.steps}")

        # Increment step count
        self.steps += 1

        return full_observation, reward, terminated, truncated, {}

    def render(self, mode="human"):
        if self.render:
            print(f"Rendering: Goal Position {self.goal_position}")

    def close(self):
        print("Closing environment.")
        self.sim.close()
