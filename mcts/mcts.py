import math
import numpy as np
from copy import deepcopy
import itertools
from tablut.rules.ashton import Board, Player
from tablut.game import Game, WinException, LoseException, DrawException
from mcts.heuristics import Heuristic
import time

BOARD_SIDE = 9
BOARD_SIZE = BOARD_SIDE * BOARD_SIDE
ACTION_SPACE_SIZE = BOARD_SIZE**2


def flatten_move(start: tuple, end: tuple) -> int:
    flattened_start = (start[0] * BOARD_SIDE) + start[1]
    flattened_end = (end[0] * BOARD_SIDE) + end[1]
    return ((flattened_start * BOARD_SIZE) + flattened_end)


def deflatten_move(move: int) -> tuple:
    """
    Received moves in in [0, 64**2].
    To get the starting flattened we divide by 64, for the ending flattened we 
    take the modulo of 64.

    To get the single coords of a move we take division and modulo by 8
    """
    start, end = divmod(move, BOARD_SIZE)
    start = divmod(start, BOARD_SIDE)
    end = divmod(end, BOARD_SIDE)
    return start, end


class Node(object):
    """
    Based on https://www.moderndescartes.com/essays/deep_dive_mcts/
    """

    def __init__(self, game, parent=None, move=None, remaining_moves=50, C=np.sqrt(2), heuristic=None):
        self.heuristic = heuristic
        self.game = game
        self.move = move

        self.remaining_moves = remaining_moves
        self.C = C

        self.is_expanded = False
        self.parent = parent
        self.children = {}  # Dict[move, Node instance]

        # search legal moves starting from the current state
        self.legal_moves = [
            flatten_move(*x) for x in self.possible_moves()
            if game.board.is_legal(game.turn, *x)[0]]

        # if no moves can be performed the current player loses!
        if len(self.legal_moves) == 0:
            self.game.ended = True
            self.is_expanded = True
            self.total_value = -1 if game.turn is Player.WHITE else 1
        else:
            self.child_total_value = np.zeros(
                [len(self.legal_moves)], dtype=np.float32)
            self.child_number_visits = np.zeros(
                [len(self.legal_moves)], dtype=np.float32)

    def _possible_starting_positions(self):
        """
        Compute the possible starting positions for the current player
        """
        if self.game.turn is Player.WHITE:
            positions = (self.game.board.board == 2) | (
                self.game.board.board == 1) | (self.game.board.board == 1.7)
        else:
            positions = (self.game.board.board == -2.5) | (
                self.game.board.board == -2)
        return positions

    def _possible_ending_positions(self):
        """
        Compute the possible starting positions for the current player
        """
        position = self.game.board.board == 0
        if self.game.turn is Player.WHITE:
            position = position | (self.game.board.board == 0.7) | (
                self.game.board.board == 0.3)
        else:
            position = position | (self.game.board.board == -0.5)
        return position

    def _is_orthogonal(self, start, end):
        """
        Check if the move is orthogonal
        """
        return start[0] == end[0] or start[1] == end[1]

    def possible_moves(self):
        """
        Computes all the possible moves given the current game state
        """
        starting_positions = self._possible_starting_positions()
        sri = list(starting_positions.sum(0).nonzero()[0])
        sci = list(starting_positions.sum(1).nonzero()[0])
        starts = [x for x in itertools.product(sri, sci)]

        ending_positions = self._possible_ending_positions()
        eri = list(ending_positions.sum(0).nonzero()[0])
        eci = list(ending_positions.sum(1).nonzero()[0])
        ends = [x for x in itertools.product(eri, eci)]

        moves = (x for x in itertools.product(
            starts, ends) if x[0] != x[1] and self._is_orthogonal(*x))
        return moves

    @property
    def number_visits(self):
        """
        Number of times a child has been visited (reduce its U)
        """
        idx = self.parent.legal_moves.index(self.move)
        return self.parent.child_number_visits[idx]

    @number_visits.setter
    def number_visits(self, value):
        idx = self.parent.legal_moves.index(self.move)
        self.parent.child_number_visits[idx] = value

    @property
    def total_value(self):
        """
        How many win has been totalized passing through that node
        """
        idx = self.parent.legal_moves.index(self.move)
        return self.parent.child_total_value[idx]

    @total_value.setter
    def total_value(self, value):
        idx = self.parent.legal_moves.index(self.move)
        self.parent.child_total_value[idx] = value

    def child_Q(self):
        """
        Calculate Q of each child as the number of wins of that child divided by the number of plays of
        that child if we select him
        e.g. already know quality of the node, how good we know that child is 
        """
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        """
        Calculate U of each child as sqrt(2) * sqrt(ln(N) / n) where N is the number of visits of
        this node, and n is the number of visits of the child if we choose him 
        U is unexpectancy e.g. how little we have explored that child 
        """
        return self.C * np.sqrt(np.log(self.number_visits + 1) / (self.child_number_visits + 1))

    def child_Pb(self):
        """
        Compute the progressive bias for each child defined as h/n where h is the defined heuristics and n 
        the number of visits of the child if we choose him
        """
        child_heuristics = list()
        for move in self.legal_moves:
            next_game = self.game.what_if(*deflatten_move(move))
            packed_board = next_game.board.pack(next_game.board.board)
            child_heuristics.append(
                heuristics.evaluate(packed_board, self.game.turn))

        return child_heuristics / (self.child_number_visits + 1)

    def best_child(self):
        """
        Return the most promising move among childs
        """
        #idx = np.argmax(self.child_Q() + self.child_U() + self.child_Pb())
        idx = np.argmax(self.child_Q() + self.child_U())
        return self.legal_moves[idx]

    def select_leaf(self):
        """
        Traverse tree reaching a leaf node who hasn't been expanded
        yet, choose the best child based on the evaluations made on them
        """
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def expand(self):
        """
        Expand the game by taking random actions until a win condition is met
        """
        if not self.is_expanded:
            self.is_expanded = True
            current = self
            while not current.game.ended and current.remaining_moves > 0:
                # Get the best possible move
                if current.heuristic is not None:
                    def what_if_board(m):
                        return current.game.what_if(*deflatten_move(m)).board.board
                    move_priors = [current.heuristic.evaluate(what_if_board(m), current.game.turn)
                                   for m in current.legal_moves]
                    move = current.legal_moves[np.argmax(move_priors)]
                else:
                    move = np.random.choice(current.legal_moves)

                current = current.maybe_add_child(move)
        return current

    def add_child(self, move):
        """
        Add a child node
        """
        new_game = deepcopy(self.game)
        # skip legality check when making move as we only try legal moves
        if new_game.turn is Player.WHITE:
            new_game.white_move(*deflatten_move(move), known_legal=True)
        else:
            new_game.black_move(*deflatten_move(move), known_legal=True)

        rm = self.remaining_moves - 1
        self.children[move] = Node(
            new_game, parent=self, move=move, remaining_moves=rm, heuristic=self.heuristic)
        return self.children[move]

    def maybe_add_child(self, move):
        """
        If a move have already been performed in the past return its associated child,
        otherwise create it
        """
        if move not in self.children:
            self.add_child(move)
        return self.children[move]

    def backup(self):
        """
        Backpropagate results.
        Note that depending on the player who is making the move whe need to change the 
        value estimation sign so that each player can take the best possible actions.
        In this scenario as WHITE moves first we invert when BLACK is playing.
        """
        winner = self.game.winner
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.turn is winner:
                current.total_value += 1
            current = current.parent


class Root(Node):
    def __init__(self, game, **kwargs):
        super().__init__(game, parent=None, move=None, **kwargs)
        self._number_visits = 0
        self._total_value = 0

    @property
    def number_visits(self):
        """
        Number of times this node has been visited
        """
        return self._number_visits

    @number_visits.setter
    def number_visits(self, value):
        """
        Sets the times this node has been visited
        """
        self._number_visits = value

    @property
    def total_value(self):
        """
        How many wins passing through this node has been totalized
        """
        return self._total_value

    @total_value.setter
    def total_value(self, value):
        """
        Set how many wins passing through this node has been totalized
        """
        self._total_value = value


class MCTS(object):
    """
    Perform montecarlo tree search on the tablut game.
    # TODO: Implement adaptive max_depth? Detect how useless a reached state is?
    # TODO: Implement adaptive simulation? If we can do few legal moves it makes sense doing more simulations
    # TODO: Implement exploratory-quality-heuristics weights?
    # TODO: Implement self-adjusting heuristics? Later in the game being near the escape is more important than having less pieces
    """

    def __init__(self, game_state, max_depth=20, use_heuristics=True):
        self.heuristic = Heuristic() if use_heuristics else None
        self.game = deepcopy(game_state)
        self._needed_moves = list()
        self.max_depth = max_depth
        self._root = Root(game_state, remaining_moves=max_depth,
                          heuristic=self.heuristic)

    @property
    def root(self):
        return self._root

    def new_root(self, start, end):
        """
        Set as root the node obtained by applying the specified move in the current state
        """
        m = flatten_move(start, end)
        if m in self._root.children:
            self._root = self._root.children[m]
        else:
            # move is not available among root children so we obtain it manually
            if self.game.turn is Player.WHITE:
                self.game.white_move(start, end, known_legal=True)
            else:
                self.game.black_move(start, end, known_legal=True)

            self._root = Root(self.game, remaining_moves=self.max_depth,
                              heuristic=self.heuristic)

    def search(self, max_time):
        """
        Perform search using a specified amount of simulations
        Max time represents the number of secs before timeout
        # TODO: Detect if multiple CPUs and implement multithread search? (Node is not thread safe)
        """
        start = self._root
        start_t = time.time()
        self.simulations = 0

        s_per_simulation = list()
        def avg(l): return 0 if len(l) == 0 else sum(l) / len(l)

        while ((time.time() - start_t) + avg(s_per_simulation)) <= max_time:
            simulation_start_t = time.time()
            leaf = start.select_leaf()
            leaf = leaf.expand()
            self._needed_moves.append(self.max_depth - leaf.remaining_moves)
            leaf.backup()
            s_per_simulation.append(time.time() - simulation_start_t)
            self.simulations += 1
            if self.simulations % 1000 == 0:  # We print this at every k
                print("Currently at simulation: {}".format(self.simulations))

        # adapt new max_depth based on past needed moves
        avg = sum(self._needed_moves) / len(self._needed_moves)
        self.max_depth = int(avg)

        move, node = max(start.children.items(),
                         key=lambda item: item[1].number_visits)

        self._root = node
        self._root.remaining_moves = self.max_depth
        self.game = self._root.game
        return deflatten_move(move)


if __name__ == "__main__":
    simulations = 10
    game = Game(Board())
    mcts = MCTS(game, max_depth=50, use_heuristics=False)
    import time
    tick = time.time()
    start, end = mcts.search(simulations)
    tock = time.time()
    print("%s -> %s _  %d simulations in %.2fs" %
          (start, end, simulations, tock - tick))
    import resource
    print("Consumed %sB memory" %
          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
