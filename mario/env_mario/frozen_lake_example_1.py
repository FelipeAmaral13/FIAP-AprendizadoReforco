import gym
import pandas as pd

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
env.reset()


episodio=list()
obs=list()
acao=list()
premio=list()
acabou=list()
informacoes=list()
passos=list()


for i in range(20):
    episodio.append(i)
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        acao.append(action)
        observation, reward, done, _, info = env.step(action)
        obs.append(observation)
        premio.append(reward)
        acabou.append(done)
        informacoes.append(info)
        if done:
            print("Episode terminou depois {} passos".format(t + 1))
            passos.append(t)
            break

env.close()

df = pd.DataFrame(
  {'Episodio':i, 'Obersevacoes': obs, 'Acao': acao, 'Premiacoes': premio, 'Acabou?': acabou, 'Informacoes': informacoes, 'Passos':t}
)
print(df)
df[df['Premiacoes'] > 0.0]