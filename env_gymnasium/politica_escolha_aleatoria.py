import random

def random_policy(actions):
    # Escolha uma ação aleatoriamente dentre as ações possíveis
    return random.choice(actions)

# Define as ações possíveis
actions = ["avançar", "virar à direita", "virar à esquerda", "parar"]

# Executa o algoritmo várias vezes
for i in range(10):
    # Escolha uma ação aleatoriamente
    selected_action = random_policy(actions)
    print("Ação selecionada na iteração {}: {}".format(i, selected_action))
