import math
import numpy as np
from copy import deepcopy
import collections
from tablut.rules.ashton import Board, Player
from tablut.game import WinException, LoseException, DrawException

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

    def __init__(self, board, player, parent=None, move=None, remaining_moves=50):
        """
        Initialize a node containing the board state, the move performed to reach this node and the
        parent of the node.

        Move is in flattened rappresentation: (1,0) = 10, (0,1) = 1, (2, 3) = 23
        
        child_priors -> how promising are the child
        child_total_value -> evaluation of childs (high if they bring to a win)
        child_number_visits -> how many times have this node been visited?
        """
        self.board = board
        self.player = player
        self.move = None

        self.remaining_moves = remaining_moves

        self.is_expanded = False
        self.parent = parent
        self.children = {} # Dict[move, Node instance]

        self.child_priors = np.random.random([ACTION_SPACE_SIZE])
        self.child_total_value = np.zeros([ACTION_SPACE_SIZE], dtype=np.float32)
        self.child_number_visits = np.zeros([ACTION_SPACE_SIZE], dtype=np.float32)

    @property
    def number_visits(self):
        """
        Number of times a child has been visited (reduce its U)
        """
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        """
        How good a Node is (high Q)
        """
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        """
        Calculate Q of each child
        Q is quality e.g. how good we know a child is 
        """
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        """
        Calculate U of each child
        U is unexpectancy e.g. how little we have explored a child 
        """
        return np.sqrt(self.number_visits) * (
            self.child_priors / (1 + self.child_number_visits))

    def best_child(self):
        """
        Return the most promising move among childs
        """
        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self):
        """
        Traverse tree reaching a leaf node who hasn't been expanded
        yet, choose the best child based on the evaluations made on them
        """
        current = self
        while current.is_expanded and current.remaining_moves != 0:
            # Get most promising move
            best_move = current.best_child()
            start, end = deflatten_move(best_move)
            # Check if move is legal
            legal, _ = current.board.is_legal(current.player, start, end)
            # Get che child
            if legal:
                try:
                    child = current.maybe_add_child(best_move) 
                    current = child
                except WinException:
                    # Reached leaf is a win condition for white
                    #print("Found win")
                    current.total_value = 1
                    print("WIN")
                    current.is_expanded = False
                except (DrawException, LoseException):
                    # Reached leaf is a lose condition
                    current.total_value = -1
                    current.is_expanded = False
            else:
                # Move is illegal so its probability of being chosen need to be 0
                current.child_priors[best_move] = 0
        return current

    def expand(self, child_priors):
        """
        Set this node as fully expanded and set the next nodes prior goodness
        """
        self.is_expanded = True
        self.child_priors = child_priors

    def maybe_add_child(self, move):
        """
        Check if a move from the current state has already been performed
        return the corresponding child node or create the node
        """
        if move not in self.children:
            # deflatten move for board execution
            start, end = deflatten_move(move)
            # get new state
            new_state = deepcopy(self.board)
            new_state.step(self.player, start, end)
            # create new node that will execute as other player
            self.children[move] = Node(new_state, self.player.next(), parent=self, remaining_moves=(self.remaining_moves - 1))
        return self.children[move]

    def backup(self, value_estimate):
        """
        Backpropagate results.
        Note that depending on the player who is making the move whe need to change the 
        value estimation sign so that each player can take the best possible actions.
        In this scenario as WHITE moves first we invert when BLACK is playing.
        """
        invert = 1 if self.player is Player.WHITE else -1
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += (value_estimate * invert)
            current = current.parent

class DummyNode(object):
  def __init__(self):
    self.parent = None
    self.child_priors = np.random.random([ACTION_SPACE_SIZE])
    self.child_total_value = np.zeros([ACTION_SPACE_SIZE], dtype=np.float32)
    self.child_number_visits = np.zeros([ACTION_SPACE_SIZE], dtype=np.float32)


def move_search(game_state, num_reads, root=None, max_depth=50):
    if root is None:
        root = Node(game_state, Player.WHITE, parent=DummyNode(), remaining_moves=max_depth)
    for i in range(num_reads):
        leaf = root.select_leaf()
        child_priors = np.random.random([ACTION_SPACE_SIZE])
        value_estimate = 0
        leaf.expand(child_priors)
        leaf.backup(value_estimate)

    move, node = max(root.children.items(), key=lambda item: item[1].number_visits)
    start, end = deflatten_move(move)
    return start, end, node

if __name__ == "__main__":
    num_reads = 1300
    game = Board()
    import time
    tick = time.time()
    start, end, node = move_search(game, num_reads, max_depth=50, root=None)
    tock = time.time()
    print("%s -> %s _  %d simulations in %.2fs" % (start, end, num_reads, tock - tick))
    import resource
    print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)