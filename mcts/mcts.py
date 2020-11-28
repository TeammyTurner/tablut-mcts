import math
import numpy as np
from copy import deepcopy
import collections
from tablut.rules.ashton import Board, Player
from tablut.game import Game, WinException, LoseException, DrawException

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


class RootNode(object):
    def __init__(self):
        self.root = True
        self.legal_move_map = list(range(ACTION_SPACE_SIZE))
        self.child_total_value = np.zeros([ACTION_SPACE_SIZE], dtype=np.float32)
        self.child_number_visits = np.zeros([ACTION_SPACE_SIZE], dtype=np.float32)


class Node(object):
    """
    Based on https://www.moderndescartes.com/essays/deep_dive_mcts/
    """

    def __init__(self, game, parent=None, move=None, remaining_moves=50):
        """
        Initialize a node containing the board state, the move performed to reach this node and the
        parent of the node.

        Move is in flattened rappresentation: (1,0) = 10, (0,1) = 1, (2, 3) = 23
        
        child_priors -> how promising are the child
        child_total_value -> evaluation of childs (high if they bring to a win)
        child_number_visits -> how many times have this node been visited?
        """
        self.root = False
        self.game = game
        self.move = move

        self.remaining_moves = remaining_moves

        self.is_expanded = False
        self.parent = parent
        self.children = {} # Dict[move, Node instance]

        #self.child_priors = np.zeros([ACTION_SPACE_SIZE], dtype=np.float32)
        legal = lambda m: game.board.is_legal(game.turn, *deflatten_move(m))[0]
        self.legal_move_map = list(filter(legal, range(ACTION_SPACE_SIZE)))
        self.child_total_value = np.zeros([len(self.legal_move_map)], dtype=np.float32)
        self.child_number_visits = np.zeros([len(self.legal_move_map)], dtype=np.float32)

    @property
    def number_visits(self):
        """
        Number of times a child has been visited (reduce its U)
        """
        idx = self.parent.legal_move_map.index(self.move)
        return self.parent.child_number_visits[idx]

    @number_visits.setter
    def number_visits(self, value):
        idx = self.parent.legal_move_map.index(self.move)
        self.parent.child_number_visits[idx] = value

    @property
    def total_value(self):
        """
        How many win has been totalized passing through that node
        """
        idx = self.parent.legal_move_map.index(self.move)
        return self.parent.child_total_value[idx]

    @total_value.setter
    def total_value(self, value):
        idx = self.parent.legal_move_map.index(self.move)
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
        return np.sqrt(2) * np.sqrt(np.log(self.number_visits) / (self.child_number_visits + 1))

    def best_child(self):
        """
        Return the most promising move among childs
        """
        idx = np.argmax(self.child_Q() + self.child_U())
        return self.legal_move_map[idx]

    def select_leaf(self):
        """
        Traverse tree reaching a leaf node who hasn't been expanded
        yet, choose the best child based on the evaluations made on them
        """
        current = self
        while current.is_expanded:
            # Get most promising move
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)        
        return current

    def expand(self):
        """
        Expand the game by taking random actions until a win condition is met
        """
        current = self
        while not current.game.ended and current.remaining_moves > 0:
            # Get one random move
            move = np.random.choice(current.legal_move_map)
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

        self.children[move] = Node(new_game, parent=self, move=move, remaining_moves=(self.remaining_moves - 1))
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
        while not current.parent.root:
            current.number_visits += 1
            if current.game.turn is winner:
                current.total_value += 1
            current = current.parent


def move_search(game_state, simulations, start=None, max_depth=50):
    if start is None:
        start = Node(game_state, parent=RootNode(), remaining_moves=max_depth)
    
    for i in range(simulations):
        leaf = start.select_leaf()
        leaf = leaf.expand()
        print("Winner: %s" % leaf.game.winner)
        leaf.backup()

    # If start is Root than take the first children (which is indeed the first game state)
    # this is due to numpy usage where data about child is actually stored in parent
    if start.parent.root:
        end = list(start.children.values())[0]
    else:
        end = start

    move, node = max(end.children.items(), key=lambda item: item[1].number_visits)
    start, end = deflatten_move(move)
    return start, end, node

if __name__ == "__main__":
    num_reads = 50
    game = Game(Board())
    import time
    tick = time.time()
    start, end, node = move_search(game, num_reads, max_depth=10, start=None)
    tock = time.time()
    print("%s -> %s _  %d simulations in %.2fs" % (start, end, num_reads, tock - tick))
    import resource
    print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)