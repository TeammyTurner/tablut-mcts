from mcts import MCTS
from tablut.game import Game, Player
from tablut.rules.ashton import Board
from tablut.player import RandomPlayer
import time
from copy import deepcopy
import pprint


board = Board()
game = Game(board)
black_player = RandomPlayer(game, Player.BLACK)

# mcts parameters
num_reads = 10000
max_depth = 50

node = None
while not game.ended:
  tick = time.time()
  mcts = MCTS(deepcopy(game), max_depth=120)
  start, end = mcts.search(3)
  # TODO: Reuse the tree
  tock = time.time()
  print("%s -> %s _  %.2fs" % (start, end, tock - tick))

  game.white_move(start, end) 
  black_player.play()

  pprint.pprint(game.board.pack(game.board.board))

print(game.winner)