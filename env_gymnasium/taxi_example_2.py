"""
Objetivo:
Inicializar um ambiente de aprendizado por Reforço usando a lib Gym da api taxi
Vamos começar com a maneira mais simples de treinar nosso agente para concluir essa tarefa. 
O agente daria passos aleatórios em cada estado até completar a tarefa (pegar o passageiro e deixá-lo no local de desembarque).

"""

import  gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode="human")
state = env.reset()

random_policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

def random_policy_steps_count():
    state = env.reset()
    counter = 0
    reward = None
    while reward != 20:
        state, _, reward, done, info = env.step(env.action_space.sample())  
        counter += 1
        print(f"Counter {counter}, State {state}, Reward {reward}, Done? {done}, info {info}")
    return counter

counts = [random_policy_steps_count() for i in range(10)]

sns.distplot(counts)
plt.title("Distribution of number of steps needed")
plt.show()

print("An agent using Random search takes about an average of " + str(int(np.mean(counts)))
      + " steps to successfully complete its mission.")