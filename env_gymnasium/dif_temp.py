import random

# Define o número de estados e ações possíveis
num_estados = 5
num_acoes = 2

# Define a taxa de aprendizado e o fator de desconto
alpha = 0.1
gamma = 0.5

# Cria uma tabela Q com zeros para todas as entradas
Q = [[0 for _ in range(num_acoes)] for _ in range(num_estados)]

# Define um estado inicial aleatório
estado_atual = random.randint(0, num_estados - 1)

# Define o número de iterações do loop de treinamento
num_iteracoes = 10

# Loop de treinamento
for _ in range(num_iteracoes):
    # Escolhe uma ação aleatória
    acao = random.randint(0, num_acoes - 1)
    
    # Executa a ação e observa o novo estado e a recompensa
    novo_estado = random.randint(0, num_estados - 1)
    recompensa = random.randint(0, 10)
    
    # Calcula a diferença temporal
    diferenca_temporal = recompensa + gamma * max(Q[novo_estado]) - Q[estado_atual][acao]
    
    # Atualiza a tabela Q com a diferença temporal
    Q[estado_atual][acao] += alpha * diferenca_temporal
    
    # Atualiza o estado atual
    estado_atual = novo_estado
    
    # Imprime a tabela Q atualizada
    print(Q)
