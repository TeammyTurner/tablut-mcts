from tablut.game import Player

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


#Calculate a value based on how many pieces of a specific color 
#can be captured by the opponent with a single move
# At 0, the function returns 0
# At more pieces, it will return exponentially more
def pieces_in_trouble_rating(board,player):
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