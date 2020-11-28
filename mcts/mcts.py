import math
import numpy as np
from copy import deepcopy
import collections
from tablut.rules.ashton import Board, Player
from tablut.game import Game, WinException, LoseException, DrawException
import heuristics

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
    """
    Based on https://www.moderndescartes.com/essays/deep_dive_mcts/
    """
    def __init__(self, game, parent=None, move=None, remaining_moves=50, C=np.sqrt(2)):
        self.game = game
        self.move = move

        self.remaining_moves = remaining_moves
        self.C = C

        self.is_expanded = False
        self.parent = parent
        self.children = {}  # Dict[move, Node instance]

        # search legal moves starting from the current state
        def legal(m): return game.board.is_legal(
            game.turn, *deflatten_move(m))[0]
        self.legal_moves = list(filter(legal, range(ACTION_SPACE_SIZE)))
        self.child_total_value = np.zeros(
            [len(self.legal_moves)], dtype=np.float32)
        self.child_number_visits = np.zeros(
            [len(self.legal_moves)], dtype=np.float32)

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
        for move, node in self.children.items():
            player = node.game.turn
            next_board = node.game.what_if(*deflatten_move(move))
            child_heuristics.append(heuristics.evaluate(next_board, player))

        return child_heuristics / (self.child_number_visits + 1)

    def best_child(self):
        """
        Return the most promising move among childs
        """
        idx = np.argmax(self.child_Q() + self.child_U() + self.child_Pb())
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
        self.is_expanded = True
        current = self
        while not current.game.ended and current.remaining_moves > 0:
            # Get one random move
            move = np.random.choice(current.legal_moves)
            current = current.maybe_add_child(move)
        return current

    def add_child(self, move):
        """
        Add a child node
        """
        new_game = deepcopy(self.game)
        if new_game.turn is Player.WHITE:
            new_game.white_move(*deflatten_move(move))
        else:
            new_game.black_move(*deflatten_move(move))

        rm = self.remaining_moves - 1
        self.children[move] = Node(new_game, parent=self, move=move, remaining_moves=rm)
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
    def __init__(self, game, remaining_moves=50):
        super().__init__(game, parent=None, move=None, remaining_moves=remaining_moves)
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
        self._total_value = v


def move_search(game_state, simulations, start=None, max_depth=50):
    if start is None:
        start = Root(game_state, remaining_moves=2)



    for _ in range(simulations):
        leaf = start.select_leaf()
        leaf = leaf.expand()
        #print("Winner: %s" % leaf.game.winner)
        leaf.backup()

    move, node = max(start.children.items(),
                     key=lambda item: item[1].number_visits)
    start, end = deflatten_move(move)
    return start, end, node


if __name__ == "__main__":
    num_reads = 50
    game = Game(Board())
    import time
    tick = time.time()
    start, end, node = move_search(game, num_reads, max_depth=10, start=None)
    tock = time.time()
    print("%s -> %s _  %d simulations in %.2fs" %
          (start, end, num_reads, tock - tick))
    import resource
    print("Consumed %sB memory" %
          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
