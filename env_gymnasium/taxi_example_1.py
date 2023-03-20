"""
Introdução a biblioteca Gym. 

A biblioteca Gym (abreviação de "OpenAI Gym") é um ambiente de desenvolvimento de software de código aberto para a criação e avaliação de algoritmos de aprendizado por 
reforço (RL). A biblioteca é mantida pela OpenAI e fornece uma variedade de tarefas (ou "ambientes") simulados que podem ser usados para treinar e testar algoritmos de 
aprendizado por reforço.

Esses ambientes incluem jogos, problemas de controle de robôs, simulações de física e muito mais. Cada ambiente fornece uma interface padronizada que permite que os 
algoritmos de RL interajam com o ambiente de maneira consistente, tornando mais fácil para os pesquisadores comparar e avaliar diferentes algoritmos.

Além disso, a biblioteca Gym também inclui ferramentas úteis, como APIs para visualização de desempenho, bibliotecas de registro de dados e uma ampla documentação 
para ajudar os usuários a começar a trabalhar com RL usando a biblioteca.

LINK: https://www.gymlibrary.dev/

https://www.kaggle.com/code/angps95/intro-to-reinforcement-learning-with-openai-gym

"""

"""
Objetivo:
Inicializar um ambiente de aprendizado por Reforço usando a lib Gym da api taxi
Verificar as ações e o ambiente observáveis.
Fazer uma movimentação simples e capturar as informacoes 

"""
import gym

env = gym.make("Taxi-v3", render_mode="human")
env.reset()
env.render()

# Estados observaveis
print(env.observation_space.n)

# Acoes possiveis
print(env.action_space.n)

""" 0: Move south (down)
    1: Move north (up)
    2: Move east (right)
    3: Move west (left)
    4: Pickup passenger
    5: Drop off passenger
"""

# Movimentar o taxi
env.env.s = 114
env.render()

# Operacoes no agente
op = env.step(1)
"""
A cada timestep, o agente escolhe uma ação e o ambiente retorna uma observação e uma recompensa.

 Os 4 elementos retornados são:

     Observação (objeto): o estado em que o ambiente está ou um objeto específico do ambiente que representa sua observação do ambiente.

     Recompensa (float): Recompensa alcançada pela ação anterior.
         +20: Última etapa quando pegamos um passageiro com sucesso e o deixamos no local desejado
         -1: para cada etapa para que o agente tente encontrar a solução mais rápida possível
         -10: toda vez que você pegar ou deixar um passageiro incorretamente

     Concluído (booleano): se é hora de redefinir o ambiente novamente.  
     A maioria (mas não todas) as tarefas são divididas em episódios bem definidos, e quando concluído, True indica que o episódio terminou.  
     (Por exemplo, você perdeu sua última vida.)

     Info (dict): Pode ser ignorado, informações de diagnóstico úteis para depuração.  
     As avaliações oficiais do seu agente não têm permissão para usar isso para aprendizado
"""
print("Observação: ", op[0]," Recompensa: ", op[1], " Concluído? ", op[2], " info: ", op[4])

