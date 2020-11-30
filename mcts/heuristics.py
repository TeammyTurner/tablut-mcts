from tablut.game import Player
import numpy as np

class Heuristic(object):
    def __init__(self):
        self.ESCAPE_DISTANCE_WEIGHT = 1
        self.PLAYER_RATIO_WIGHT = 1

    def evaluate(self, board, player):
        """
        Return the weighted sum of the parameters
        """
        score = 0

        # Player to enemies player ratio
        enemy = player.next()
        score += self.PLAYER_RATIO_WIGHT * (player_pieces(board, player) / player_pieces(board, enemy))
        
        # Low distance of king from escape is good for whites, bad for blacks
        how_good = 1 if player is Player.WHITE else -1
        score += how_good * self.ESCAPE_DISTANCE_WEIGHT * king_escape_distance(board)

        return score

def player_pieces(board, player):
    """
    Get the number of pieces of a player (escape occupied by king is escluded as its a winning condition)
    Players count are normalized between 0 and 1 because initially black have more pieces than white, we
    want a fair evaluation in comparison to initial state
    """
    count = 0
    if player is Player.WHITE:
        count = np.sum(board > 1.3) / 9
    else:
        count = np.sum(board < -2) / 12 

    return count

def king_escape_distance(board):
    """
    Distance between king ad escape
    king is considered near an escape if its outside a square of radius 4 with center on the castle:
    let king coords be (x, y), 
        if |(x - 4) + (y - 4)| + |(x - 4) - (y - 4)| > 4 => king is near the border
    """
    # king position
    p = np.transpose(((board == 1.7) | (board == 1) | (board == 1.3)).nonzero())
    if p.sum() > 0:
        p = p[0]
        distance = np.abs((p[0] - 4) + (p[1] + 4)) + np.abs((p[0] - 4) - (p[1] + 4))
    else:
        # If no king is found then has been captured
        distance = 5

    return distance


KING_IN_TROUBLE_WEIGHT = 1
KING_IN_TROUBLE_EXP_BASE = 1.2

PIECE_IN_TROUBLE_WEIGHT = 0.5
EXPONENT = 1.2

# Gets the board in the string format (output of the pack() function) and the player 
# that have to make the move and evaluates the given board and returns a number based on some indeces
def evaluate(board, player):
    reward = 0
    if player == Player.WHITE:
        return reward - king_in_trouble_rating(board) + pieces_in_trouble_rating(board,player)\
            - pieces_in_trouble_rating(board,Player.BLACK) - same_axis_as_king_rating(board)
    else:
        return reward + king_in_trouble_rating(board) + pieces_in_trouble_rating(board,player)\
            - pieces_in_trouble_rating(board,Player.WHITE) + same_axis_as_king_rating(board)


# Calculates a value for the king's threat of capture as an exponential
# function in the number of the black pieces that are around the king
# At 0, the function returns 0
# At more pieces, it will return exponentially more
def king_in_trouble_rating(board):
    black_pieces_around_king = 0
    black = "B"
    king = "K"
    #Find the king coordinates in the board
    for x in range(0,9):
        for y in range(0,9):
            if board[x][y][1] == king:
                king_x, king_y = x, y
                break

    black_pieces_around_king = 0
    #Check if the king has black pieces around
    if king_x - 1 >= 0:
        if board[king_x - 1][king_y][1] == black:
            black_pieces_around_king += 1
    if king_x + 1 <= 8:
        if board[king_x + 1][king_y][1] == black:
            black_pieces_around_king += 1
    if king_y - 1 >= 0:
        if board[king_x][king_y - 1][1] == black:
            black_pieces_around_king += 1
    if king_y + 1 <= 8:
        if board[king_x][king_y + 1][1] == black:
            black_pieces_around_king += 1

    return KING_IN_TROUBLE_WEIGHT*(KING_IN_TROUBLE_EXP_BASE**black_pieces_around_king - 1)



def pieces_in_trouble_rating(board, player):
    """
    Calculate a value based on how many pieces of a specific color can be captured by 
    the opponent with a single move
    At 0, the function returns 0
    At more pieces, it will return exponentially more
    """
    if player == "Player.WHITE":
        attacker = "W"
        defender = "B"
    else:
        attacker = "B"
        defender = "W"

    empty = "e"
    pieces_in_threat_rating = 0
    for x in range(0,9):
        for y in range(0,9):
            piece_danger = 0
            if board[x][y][1] == defender:
                # Check on all the ortogonal directions if there is a piece of the opponent
                # If there is, check if there is another pieces of the opponent that can make the capture
                if (x - 1 >= 0) and (x + 1 <= 8):
                    if board[x - 1][y][1] == attacker and board[x + 1][y][1] == empty:
                        for step in range(x+1,9):
                            if board[step][y][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[step][y][1] == empty: continue
                            else: break
                        for step in range(y,9):
                            if board[x+1][step][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[x+1][step][1] == empty: continue
                            else: break
                        for step in reversed(range(y)):
                            if board[x+1][step][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[x+1][step][1] == empty: continue
                            else: break
                    #Change direction            
                    if board[x + 1][y][1] == attacker and board[x - 1][y][1] == empty:
                        for step in reversed(range(x)):    
                            if board[step][y][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[step][y][1] == empty: continue
                            else: break
                        for step in range(y,9):
                            if board[x-1][step][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[x-1][step][1] == empty: continue
                            else: break
                        for step in reversed(range(y)):
                            if board[x-1][step][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[x-1][step][1] == empty: continue
                            else: break

                if (y - 1 >= 0) and (y + 1 <= 8):
                    #Change direction
                    if board[x][y - 1][1] == attacker and board[x][y + 1][1] == empty:
                        for step in range(y+1,9):
                            if board[x][step][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[x][step][1] == empty: continue
                            else: break
                        for step in range(x,9):
                            if board[step][y+1][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[step][y+1][1] == empty: continue
                            else: break
                        for step in reversed(range(x)):
                            if board[step][y+1][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[step][y+1][1] == empty: continue
                            else: break
                    #Change direction
                    if board[x][y + 1][1] == attacker and board[x][y - 1][1] == empty:
                        for step in reversed(range(y)):
                            if board[x][step][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[x][step][1] == empty: continue
                            else: break
                        for step in range(x,9):
                            if board[step][y-1][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[step][y-1][1] == empty: continue
                            else: break
                        for step in reversed(range(x)):
                            if board[step][y-1][1] == attacker:
                                piece_danger += 1
                                break
                            elif board[step][y-1][1] == empty: continue
                            else: break
            piece_rate = EXPONENT**piece_danger - 1
            pieces_in_threat_rating = pieces_in_threat_rating + piece_rate

    return pieces_in_threat_rating*PIECE_IN_TROUBLE_WEIGHT


# Calculates a value for the king's threat of capture based
# on the number of black pieces that are on the same ortogonal
# axies of the king.
# At 0, the function returns 0
# at more pieces, it will return exponentially more
def same_axis_as_king_rating(board):
    empty = "e"
    black = "B"
    king = "K"
    #find the king coordinates in the board
    for x in range(0,9):
        for y in range(0,9):
            if board[x][y][1] == king:
                king_x, king_y = x, y
                break

    same_axis_pieces = 0
    #Check if there are black pieces on the ortogonal axis that can attack the king
    if (king_x - 1 >= 0) and (king_x + 1 <= 8):
        for step in range(king_x+1,9):
            if board[step][king_y][1] == black:
                same_axis_pieces += 1
                break
            elif board[step][king_y][1] == empty: continue
            else: break

        for step in reversed(range(king_x)):    
            if board[step][king_y][1] == black:
                same_axis_pieces += 1
                break
            elif board[step][king_y][1] == empty: continue
            else: break

    if (king_y - 1 >= 0) and (king_y + 1 <= 8):
        for step in range(king_y+1,9):
            if board[king_x][step][1] == black:
                same_axis_pieces += 1
                break
            elif board[king_x][step][1] == empty: continue
            else: break

        for step in reversed(range(king_y)):
            if board[king_x][step][1] == black:
                same_axis_pieces += 1
                break
            elif board[king_x][step][1] == empty: continue
            else: break

    return KING_IN_TROUBLE_WEIGHT*(KING_IN_TROUBLE_EXP_BASE**same_axis_pieces - 1)