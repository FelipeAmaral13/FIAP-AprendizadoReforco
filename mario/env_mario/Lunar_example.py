import gym
import numpy as np

def policy(observation):
    # Define a simple policy to move left if the lander is tilting left
    # and move right if the lander is tilting right.
    if observation[6] > 0:
        return 2  # move right
    else:
        return 0  # move left

def random_policy(observation):
    return np.random.randint(0, 4)

def heuristic_policy(observation):
    pos_x, pos_y, vel_x, vel_y, angle, ang_vel, left_leg, right_leg = observation
    action = 0  # Do nothing by default
    
    # Use heuristics to select actions based on the current observation
    if pos_x < -0.1:
        action = 1  # Move right
    elif pos_x > 0.1:
        action = 2  # Move left
    if angle > 0.1:
        action = 3  # Rotate clockwise
    elif angle < -0.1:
        action = 0  # Rotate counterclockwise
    if left_leg > 0.1 and right_leg > 0.1:
        action = 4  # Fire the engine
    return action

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
print(observation)

for _ in range(1000):
    action = heuristic_policy(observation)  # Call the policy function to get an action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
      observation, info = env.reset()

env.close()

