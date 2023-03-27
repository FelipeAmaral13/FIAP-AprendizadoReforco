import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network policy class that takes observations as input and outputs actions
class NeuralNetworkPolicy(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetworkPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define a function to sample actions from the policy network
def sample_action(observation, policy):
    observation_tensor = torch.from_numpy(observation).float().unsqueeze(0)
    action_probs = torch.softmax(policy(observation_tensor), dim=1)
    action = np.random.choice(np.arange(action_probs.shape[1]), p=action_probs.detach().numpy().ravel())
    return action

# Initialize the environment and policy
env = gym.make("LunarLander-v2", render_mode="human")
policy = NeuralNetworkPolicy(env.observation_space.shape[0], env.action_space.n)

# Define the optimizer and loss function
optimizer = optim.Adam(policy.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train the policy using policy gradients for 1000 episodes
for episode in range(1000):
    observation = env.reset()
    episode_loss = 0.0
    for t in range(1000):
        try:
            action = sample_action(observation, policy)
            next_observation, reward, terminated, truncated, info = env.step(action)
            loss = loss_fn(policy(torch.from_numpy(observation).float().unsqueeze(0)), torch.tensor([action], dtype=torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            episode_loss += loss.item()
            observation = next_observation
        except:
            action = sample_action(observation[0], policy)
            next_observation, reward, terminated, truncated, info = env.step(action)
            loss = loss_fn(policy(torch.from_numpy(observation[0]).float().unsqueeze(0)), torch.tensor([action], dtype=torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            episode_loss += loss.item()
            observation = next_observation
        if terminated or truncated:
            break
    print("Episode {}: Loss {:.4f}".format(episode+1, episode_loss))

# Test the policy for 1000 steps
observation = env.reset()
for t in range(1000):
    action = sample_action(observation, policy)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
      observation, info = env.reset()

# Close the environment
env.close()
