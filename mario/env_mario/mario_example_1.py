import gym
import gym_super_mario_bros

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env.reset()

action = env.action_space.sample() # seleciona uma ação aleatória
observation, reward,  truncated, info = env.step(action)
env.render() # exibe o ambiente

for step in range(5000):
    action = env.action_space.sample() # seleciona uma ação aleatória
    observation, reward, done, info = env.step(action)
    env.render() # exibe o ambiente
    
    if done:
        env.reset()
