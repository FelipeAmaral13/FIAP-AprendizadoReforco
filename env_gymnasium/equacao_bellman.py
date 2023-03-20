"""
Para ilustrar o uso da equação de Bellman, vamos considerar um exemplo simples. 
Suponha que temos um ambiente com três estados: s1, s2 e s3. 
Em cada estado, o agente pode escolher entre duas ações possíveis: a1 e a2. 

A tabela a seguir mostra as recompensas que o agente recebe ao executar uma ação em cada estado:
Estado	Ação a1	Ação a2
s1	       0	1
s2	       2	3
s3	       4	5

Suponha que o agente comece no estado s1 e que o fator de desconto seja γ = 0.9. 
Podemos usar a equação de Bellman para calcular a política ótima, ou seja, a sequência de ações que maximiza o valor total da recompensa ao longo do tempo.
"""


# Define as recompensas para cada estado e ação
rewards = {
    "s1": {"a1": 0, "a2": 1},
    "s2": {"a1": 2, "a2": 3},
    "s3": {"a1": 4, "a2": 5},
}

# Define o fator de desconto
gamma = 0.9

# Inicializa o valor de todos os estados como zero
values = {"s1": 0, "s2": 0, "s3": 0}

# Define a precisão desejada para parar o loop
precision = 0.01

# Repete até a política convergir
while True:
    # Cria uma cópia dos valores atuais dos estados
    old_values = values.copy()

    # Atualiza o valor de cada estado usando a equação de Bellman
    for state in rewards:
        max_value = float('-inf')
        for action in rewards[state]:
            action_value = rewards[state][action] + gamma * old_values[state]
            if action_value > max_value:
                max_value = action_value
        values[state] = max_value

    # Verifica se a política convergiu
    converged = True
    for state in values:
        if abs(values[state] - old_values[state]) > precision:
            converged = False
            break

    if converged:
        break

# Imprime os valores finais dos estados
print("Valores finais:")
for state in values:
    print(f"{state}: {values[state]}")
