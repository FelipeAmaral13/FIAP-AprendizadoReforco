# Algoritmo de aprendizagem por reforço simples para aprender a jogar o tic-tac-toe

# Value Function: é a medida que considera a probabilidade de todas as possíveis recompensas de um estado.

# A Value Function é uma forma eficiente e rápida de buscar a árvore do jogo em busca do melhor caminho para chegar a recompensa.
# Update rule: V(s) = V(s) + alpha*(V(s') - V(s))
# s  = estado corrente
# s' = próximo estado
# s representa cada estado que nós encontramos em um episódio e por isso precisamos do histórico de estados em um episódio

# Estados terminais não são atualizados, uma vez que não existe próximo estado
# Treinamos o algoritmo sobre vários episódios até encontrar o melhor valor de alpha
# A fórmula acima é similar à fórmula do gradient descent em aprendizagem supervisionada

# Epsilon-greedy policy:
#   action|s = argmax[sobre todas as ações possíveis do estado s]{ V(s) }  if rand > epsilon
#   action|s = selecione ação aleatória de possíveis ações do estado s if rand < epsilon
#
# Algumas dicas de melhorias no algoritmo:

# Atualmente, ambos os agentes usam a mesma estratégia de aprendizagem enquanto jogam um contra o outro.
# E se eles tiverem diferentes taxas de aprendizado?
# E se eles tiverem diferentes epsilons? (probabilidade de explorar)
# Quem convergirá mais rápido?
# E se um agente não aprender?

# Uma pergunta filosófica interessante: se não há ninguém para desafiá-lo, você pode alcançar seu potencial máximo?

# Imports
import numpy as np
import matplotlib.pyplot as plt
from builtins import range, input

# Variável para ajudar a definir o total de estados possíveis
LENGTH = 3

# Classe Agente
class Agent:
    def __init__(self, eps=0.1, alpha=0.5):
        """
        Inicializa um objeto Agente com os parâmetros de aprendizado.

        Args:
            eps (float): Probabilidade de escolher uma ação aleatória em vez de gananciosa (valor entre 0 e 1).
            alpha (float): Taxa de aprendizado para atualização dos valores dos estados (valor entre 0 e 1).

        Atributos:
            eps (float): Probabilidade de ação aleatória.
            alpha (float): Taxa de aprendizado.
            verbose (bool): Define se informações detalhadas serão impressas.
            state_history (list): Lista de estados visitados durante um episódio.

        Retorna:
            Nenhum retorno explícito.
        """
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.state_history = []

    def setV(self, V):
        """
        Define os valores dos estados para o agente.

        Args:
            V (dict): Um dicionário contendo os valores dos estados.

        Retorna:
            Nenhum retorno explícito.
        """
        self.V = V

    def set_symbol(self, sym):
        """
        Define o símbolo (X ou O) do agente.

        Args:
            sym (str): O símbolo a ser usado pelo agente.

        Retorna:
            Nenhum retorno explícito.
        """
        self.sym = sym

    def set_verbose(self, v):
        """
        Define se o agente imprimirá informações detalhadas durante a execução.

        Args:
            v (bool): Verdadeiro para ativar a saída detalhada, Falso caso contrário.

        Retorna:
            Nenhum retorno explícito.
        """
        self.verbose = v

    def reset_history(self):
        """
        Limpa o histórico de estados visitados durante um episódio.

        Retorna:
            Nenhum retorno explícito.
        """
        self.state_history = []

    def take_action(self, env):
        """
        Escolhe uma ação com base na estratégia epsilon-gananciosa.

        Args:
            env (objeto): O ambiente do jogo.

        Retorna:
            Nenhum retorno explícito.
        """
        r = np.random.rand()
        best_state = None

        if r < self.eps:
            # Toma uma ação aleatória
            if self.verbose:
                print("Tomando uma ação aleatória")

            possible_moves = []
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        possible_moves.append((i, j))
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else:
            # Escolhe a melhor ação com base nos valores atuais dos estados
            pos2value = {}
            next_move = None
            best_value = -1

            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        # Qual é o estado se fizermos esse movimento?
                        env.board[i, j] = self.sym
                        state = env.get_state()
                        env.board[i, j] = 0
                        pos2value[(i, j)] = self.V[state]

                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i, j)

            # Se verbose, desenha o tabuleiro com os valores
            if self.verbose:
                print("Tomando uma ação gananciosa")
                for i in range(LENGTH):
                    print("------------------")
                    for j in range(LENGTH):
                        if env.is_empty(i, j):
                            # Imprime o valor
                            print(" %.2f|" % pos2value[(i, j)], end="")
                        else:
                            print("  ", end="")
                            if env.board[i, j] == env.x:
                                print("x  |", end="")
                            elif env.board[i, j] == env.o:
                                print("o  |", end="")
                            else:
                                print("   |", end="")
                    print("")
                print("------------------")

        # Faz o movimento
        env.board[next_move[0], next_move[1]] = self.sym

    def update_state_history(self, s):
        """
        Atualiza o histórico de estados visitados durante um episódio.

        Args:
            s (str): Estado a ser adicionado ao histórico.

        Retorna:
            Nenhum retorno explícito.
        """
        self.state_history.append(s)

    def update(self, env):
        """
        Realiza a atualização dos valores dos estados com base nas recompensas.

        Args:
            env (objeto): O ambiente do jogo.

        Retorna:
            Nenhum retorno explícito.
        """
        # Queremos BACKTRACK sobre os estados para atualizar os valores dos estados
        reward = env.reward(self.sym)
        target = reward

        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha * (target - self.V[prev])
            self.V[prev] = value
            target = value

        self.reset_history()



import numpy as np

# Classe Ambiente
class Environment:

    def __init__(self):
        """
        Inicializa o ambiente do jogo.
        """
        self.board = np.zeros((LENGTH, LENGTH))  # Cria um tabuleiro vazio
        self.x = -1  # Representa um 'x' no tabuleiro, jogador 1
        self.o = 1  # Representa um 'o' no tabuleiro, jogador 2
        self.winner = None  # Armazena o vencedor do jogo
        self.ended = False  # Indica se o jogo terminou
        self.num_states = 3 ** (LENGTH * LENGTH)  # Número de estados possíveis no jogo

    def is_empty(self, i, j):
        """
        Verifica se uma célula no tabuleiro está vazia.
        
        Args:
            i (int): Índice da linha.
            j (int): Índice da coluna.
        
        Returns:
            bool: True se a célula estiver vazia, False caso contrário.
        """
        return self.board[i, j] == 0

    def reward(self, sym):
        """
        Calcula a recompensa para um jogador.

        Args:
            sym (int): Símbolo do jogador (x ou o).

        Returns:
            int: 1 se o jogador ganhou, 0 se empatou, ou None se o jogo não acabou.
        """
        if not self.game_over():
            return 0

        return 1 if self.winner == sym else 0

    def get_state(self):
        """
        Retorna o estado atual do tabuleiro representado como um número inteiro.

        Returns:
            int: Representação do estado atual como um número inteiro.
        """
        k = 0
        h = 0
        for i in range(LENGTH):
            for j in range(LENGTH):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == self.x:
                    v = 1
                elif self.board[i, j] == self.o:
                    v = 2
                h += (3 ** k) * v
                k += 1
        return h

    def game_over(self, force_recalculate=False):
        """
        Verifica se o jogo terminou (um jogador venceu ou é um empate).

        Args:
            force_recalculate (bool): Forçar recálculo, se True.

        Returns:
            bool: True se o jogo terminou, False caso contrário.
        """
        if not force_recalculate and self.ended:
            return self.ended

        # Verifica se há um vencedor
        for i in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[i].sum() == player * LENGTH:
                    self.winner = player
                    self.ended = True
                    return True

        for j in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[:, j].sum() == player * LENGTH:
                    self.winner = player
                    self.ended = True
                    return True

        for player in (self.x, self.o):
            if self.board.trace() == player * LENGTH:
                self.winner = player
                self.ended = True
                return True

            if np.fliplr(self.board).trace() == player * LENGTH:
                self.winner = player
                self.ended = True
                return True

        if np.all((self.board == 0) == False):
            self.winner = None
            self.ended = True
            return True

        self.winner = None
        return False

    def is_draw(self):
        """
        Verifica se o jogo terminou em empate.

        Returns:
            bool: True se o jogo terminou em empate, False caso contrário.
        """
        return self.ended and self.winner is None

    def draw_board(self):
        """
        Exibe o tabuleiro do jogo no console.
        """
        for i in range(LENGTH):
            print("-------------")
            for j in range(LENGTH):
                print("  ", end="")
                if self.board[i, j] == self.x:
                    print("x ", end="")
                elif self.board[i, j] == self.o:
                    print("o ", end="")
                else:
                    print("  ", end="")
            print("")
        print("-------------")

    # Exemplo de tabuleiro
    # -------------
    # | x |   |   |
    # -------------
    # |   |   |   |
    # -------------
    # |   |   | o |
    # -------------

class Human:
    """
    Classe que representa um jogador humano em um jogo.

    Attributes:
        sym (str): O símbolo que o jogador humano utilizará no jogo.
    """

    def __init__(self):
        """
        Inicializa uma instância da classe Humano.
        """
        self.sym = None

    def set_symbol(self, sym):
        """
        Define o símbolo que o jogador utilizará no jogo.

        Args:
            sym (str): O símbolo a ser atribuído ao jogador.
        """
        self.sym = sym

    def take_action(self, env):
        """
        Solicita uma ação (movimento) do jogador humano e atualiza o ambiente.

        Args:
            env (Ambiente): O ambiente no qual o jogador está jogando.

        O jogador humano é solicitado a inserir as coordenadas (i, j) para o próximo movimento.
        Por exemplo, "0,2". O jogador continuará sendo solicitado até que um movimento legal seja feito.
        """
        while True:
            move = input("Insira as coordenadas i, j para o próximo movimento (por exemplo: 0,2): ")
            i, j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty(i, j):
                env.board[i, j] = self.sym
                break

    def update(self, env):
        """
        Atualiza o estado do jogador humano com base no ambiente.

        Args:
            env (Ambiente): O ambiente no qual o jogador está jogando.
        """
        pass  # Este método pode ser implementado para atualizar o estado do jogador com base no ambiente.

    def update_state_history(self, s):
        """
        Atualiza o histórico de estados do jogador humano.

        Args:
            s: O estado a ser adicionado ao histórico.
        """
        pass  # Este método pode ser implementado para manter o histórico de estados do jogador.



# Função recursiva que retornará todos os estados possíveis (como ints) e quem é o vencedor correspondente para esses estados (se houver) 
# (i, j) se refere à próxima célula no tabuleiro para permutar (precisamos tentar -1, 0, 1) 
# jogos impossíveis são ignorados, ou seja, 3x e 3o em uma linha simultaneamente, pois isso nunca acontecerá em um jogo real
def get_state_hash_and_winner(env, i=0, j=0):
    """
    Esta função recursiva calcula o hash do estado atual e determina o vencedor de um jogo no ambiente 'env'.

    Args:
        env: O ambiente do jogo (classe) onde o jogo está sendo jogado.
        i: Índice da linha atual no tabuleiro do jogo.
        j: Índice da coluna atual no tabuleiro do jogo.

    Returns:
        Uma lista de tuplas contendo informações sobre o estado do jogo, o vencedor e se o jogo terminou.
        Cada tupla é composta por:
        - state: O hash do estado atual.
        - winner: O vencedor do jogo (None se o jogo não terminou ou terminou em empate).
        - ended: Um valor booleano indicando se o jogo terminou (True) ou não (False).
    """

    results = []

    for v in (0, env.x, env.o):
        env.board[i, j] = v
        if j == 2:
            if i == 2:
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i + 1, 0)
        else:
            results += get_state_hash_and_winner(env, i, j + 1)

    return results


# Inicializa os estados de x com a função valor
def initialV_x(env, state_winner_triples):
    """
    Função que calcula o valor de estado (V(s)) inicial para o jogador 'x' em um ambiente (env).

    Args:
        env (objeto): O ambiente no qual o jogo está ocorrendo.
        state_winner_triples (lista de tuplas): Uma lista de triplas que contém informações sobre o estado do jogo.
            Cada tripla consiste em: (estado, vencedor, terminado).
            - estado (int): O estado do ambiente.
            - vencedor (str): O jogador que venceu o jogo ('x', 'o' ou None se houve empate).
            - terminado (bool): Indica se o jogo terminou no estado atual.

    Returns:
        numpy.ndarray: Um array contendo os valores de estado para o jogador 'x', onde:
            - Se 'x' vence, V(s) = 1.
            - Se 'x' perde ou ocorre empate, V(s) = 0.
            - Caso contrário, V(s) = 0.5.

    Exemplo de Uso:
    >>> env = MeuAmbiente()
    >>> state_winner_triples = [(0, 'x', True), (1, None, False), (2, 'o', True)]
    >>> valores_iniciais = initialV_x(env, state_winner_triples)
    """
    import numpy as np  # Certifique-se de que a biblioteca numpy está importada.

    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


# Inicializa os estados de o com a função valor
def initialV_o(env, state_winner_triples):
    """
    Inicializa um vetor de valores de estado (V) para o jogador 'o' em um ambiente (env).

    Parâmetros:
        - env: O ambiente de jogo.
        - state_winner_triples: Uma lista de triplas representando o estado, o vencedor e se o jogo terminou.

    Retorna:
        Um vetor de valores de estado (V) para o jogador 'o', onde:
        - 1 é atribuído a estados em que 'o' venceu.
        - 0 é atribuído a estados em que o jogo terminou com um empate ou 'x' venceu.
        - 0.5 é atribuído a estados em que o jogo ainda não terminou.

    Nota:
    A função percorre as triplas de estado_winner_triples e atribui os valores apropriados a cada estado no vetor V.
    """
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


# Loop até o jogo terminar
def play_game(p1, p2, env, draw=False):
    """
    Esta função permite dois jogadores (p1 e p2) competirem em um jogo, tomando alternadamente ações até que o jogo termine.

    Parâmetros:
    p1 (objeto): O primeiro jogador.
    p2 (objeto): O segundo jogador.
    env (objeto): O ambiente do jogo em que a competição ocorre.
    draw (bool, opcional): Indica se o tabuleiro do jogo deve ser desenhado durante a jogada. Pode ter os seguintes valores:
        - False: Nenhum desenho é feito.
        - True: O tabuleiro é desenhado a cada jogada.
        - 1: O tabuleiro é desenhado apenas quando o jogador p1 está ativo.
        - 2: O tabuleiro é desenhado apenas quando o jogador p2 está ativo.

    Retorno:
    Nenhum.

    A função alterna entre os jogadores p1 e p2, com p1 sempre começando primeiro. Cada jogador toma uma ação no ambiente env, atualiza o estado do jogo e o desenha (se necessário) após cada jogada. Após o término do jogo, a função atualiza a função de valor de ambos os jogadores.

    Exemplo de uso:
    play_game(jogador1, jogador2, ambiente)
    play_game(jogador1, jogador2, ambiente, draw=True)
    """
    current_player = None

    while not env.game_over():
        # Alternar entre jogadores
        # p1 sempre começa primeiro
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1

        # Desenha o tabuleiro antes que o usuário faça um movimento
        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()

        # Jogador atual faz um movimento
        current_player.take_action(env)

        # Atualiza estados
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

    if draw:
        env.draw_board()

    # Atualiza a função valor
    p1.update(env)
    p2.update(env)



if __name__ == '__main__':
    # Treinamento do Agente
    p1 = Agent()
    p2 = Agent()

    # Configura o valor inicial (V) para p1 e p2
    env = Environment()
    state_winner_triples = get_state_hash_and_winner(env)

    Vx = initialV_x(env, state_winner_triples)
    p1.setV(Vx)
    Vo = initialV_o(env, state_winner_triples)
    p2.setV(Vo)

    # Define o símbolo de cada jogador
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    # Número de iterações de treinamento
    T = 10000
    for t in range(T):
        if t % 200 == 0:
            print(t)
        play_game(p1, p2, Environment())

    # Jogando: Humano x Agente
    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
        play_game(p1, human, Environment(), draw=2)
        answer = input("Jogar novamente? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break
