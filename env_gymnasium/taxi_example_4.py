"""
Objetivo:
Inicializar um ambiente de aprendizado por Reforço usando a lib Gym da api taxi
Mostrar o Q-value

"""
# Importar as bibliotecas necessárias
import gym
import numpy as np

# Inicializar o ambiente de aprendizado por Reforço usando a lib Gym
env = gym.make("Taxi-v3", render_mode="human")

# Definir o número de estados e de ações
n_states = env.observation_space.n
n_actions = env.action_space.n

# Inicializar a tabela Q com zeros para todos os estados e ações
q_table = np.zeros([n_states, n_actions])

# Definir os parâmetros do algoritmo de aprendizado
alpha = 0.1
gamma = 0.6
epsilon = 0.1
n_episodes = 3

# Treinar o agente por um número definido de episódios
for episode in range(n_episodes):
    state = env.reset()
    done = False
    # Executar o episódio até o agente alcançar o objetivo ou falhar
    while not done:
        # Escolher uma ação epsilon-greedy (exploração ou explotação)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # exploração
        else:
            action = np.argmax(q_table[state])  # explotação

        # Executar a ação e obter a próxima observação e recompensa
        next_state, reward, done, info = env.step(action)

        # Atualizar o valor Q da ação escolhida no estado atual
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        # Atualizar o estado atual para o próximo estado
        state = next_state
        print(f"Episodio {episode}, Q-old {old_value}, Q-new {new_value}, Recompensa {reward}")

# Testar o agente treinado
total_rewards = 0
state = env.reset()
done = False

# Executar o episódio de teste até o agente alcançar o objetivo ou falhar
while not done:
    # Escolher a ação com maior valor Q para o estado atual
    action = np.argmax(q_table[state])
    # Executar a ação e obter a próxima observação e recompensa
    state, reward, done, info = env.step(action)
    # Adicionar a recompensa do passo ao total de recompensas acumuladas
    total_rewards += reward
    # Renderizar o ambiente para visualização
    env.render()

#Exibir a recompensa total acumulada pelo agente no episódio de teste
print("Total reward: {}".format(total_rewards))
