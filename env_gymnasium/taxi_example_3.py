# Importa a biblioteca Gym e a biblioteca NumPy
import gym
import numpy as np

# Cria o ambiente "Taxi-v3" e define o estado como o estado inicial
env = gym.make("Taxi-v3")
state = env.reset()

# Define o número de estados e ações disponíveis no ambiente
n_states = env.observation_space.n
n_actions = env.action_space.n

# Inicializa a matriz Q com zeros
Q = np.zeros([n_states, n_actions])

# Define o número de episódios a serem executados e uma lista para armazenar as recompensas obtidas em cada episódio
episodes = 2000
rewardTracker = []

# Define as constantes do algoritmo Q-learning: taxa de aprendizado (alpha) e fator de desconto (gamma)
G = 0
alpha = 0.618

print("Tabela Q original: ", Q)

# Executa um loop sobre o número de episódios definido
for episode in range(1,episodes+1):
    # Reseta o estado e as variáveis de recompensa e término do episódio
    done = False
    G, reward = 0,0
    state = env.reset()

    # Executa um loop enquanto o episódio não termina
    while done != True:
        # Seleciona a ação com o valor Q mais alto para o estado atual
        action = np.argmax(Q[state]) 
        # Executa a ação no ambiente e observa o próximo estado, a recompensa e se o episódio termina
        state2, reward, done, info = env.step(action) 
        # Atualiza o valor Q correspondente ao estado e ação atual usando a equação de atualização Q-learning
        Q[state,action] += alpha * ((reward + (np.max(Q[state2]))  - Q[state,action]))
        # Adiciona a recompensa ao total de recompensa acumulado no episódio e atualiza o estado atual
        G += reward
        state = state2
    
    # Adiciona a recompensa total do episódio à lista de recompensas e imprime o total de recompensa a cada 100 episódios
    rewardTracker.append(G)
    if episode % 100 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G))

print("Tabela Q Final: ", Q)


# Cria o ambiente "Taxi-v3" novamente, desta vez com o modo de renderização definido como "humano"
env = gym.make("Taxi-v3", render_mode="human")

# Reseta o estado e a variável de término do episódio
state = env.reset()
done = None

# Executa um loop enquanto o episódio não termina
while done != True:
    # Seleciona a ação com o valor Q mais alto para o estado atual
    action = np.argmax(Q[int(state)])
    # Executa a ação no ambiente, observa o próximo estado, a recompensa e se o episódio termina, e renderiza a cena
    state, reward, done, info = env.step(action)
    env.render()
