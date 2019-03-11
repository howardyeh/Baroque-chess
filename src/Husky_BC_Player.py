'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

'''

from BC_state_etc import *
from copy import deepcopy
import numpy as np

mySide = None
PIECE_PRIORITY = {'k':10000, 'f':80, 'c':60, 'i':40, 'w':30, 'l':20, 'p':10}

def nickname():
    return "Husky"

def introduce():
    return "I'm a husky. Wolf wolf!"

def prepare(player2Nickname):
    pass

def makeMove(currentState, currentRemark, timelimit):

    # Compute the new state for a move.
    # This is a placeholder that just copies the current state.
    global mySide
    if mySide is None:
        mySide = currentState.whose_move

    max_depth = 1
    move = iterative_deepening(currentState, max_depth, timelimit)
    newState = stateTransform(currentState, move)

    # Fix up whose turn it will be.
    newState.whose_move = 1 - currentState.whose_move
    
    # Construct a representation of the move that goes from the
    # currentState to the newState.
    # Here is a placeholder in the right format but with made-up
    # numbers:
    # move = ((6, 4), (3, 4))

    # Make up a new remark
    newRemark = "I'll think harder in some future game. Here's my move"

    return [[((move[1], move[2]),(move[4], move[5])), newState], newRemark]


def iterative_deepening(currentState, max_depth, timelimit):
    timesup = False
    best_move = ""
    for depth in range(max_depth):
        value, move = min_max(currentState, depth+1, [1000000, -1000000])
        if move is not None:
            best_move = move
        # if timesup:
        #     break;
        # else:
        #     value, best_move, timesup = min_max(currentState, depth, (1000000, -1000000))

    return best_move

    # TODO: add in time factor, break when time is up, maybe add a time stamp in min_max()
    #       and compare to the time from this part, if times up, break.


def min_max(state, depth, min_max_value, alphaBeta = False):

    all_move_list = []
    new_state_list = []
    best_move = ""

    if mySide == 1:
        if state.whose_move == 1: # my turn (white)
            if depth == 0:
                min_max_value[1] = staticEval(state)
                return min_max_value, ""

            all_move_list = possibleMoves(state)
            print(all_move_list)
            for m in all_move_list:
                new_state_list.append((stateTransform(state, m), m))

            for s_, m_ in new_state_list:
                nextLayer_min, _ = min_max(s_, depth - 1, min_max_value)
                if nextLayer_min[0] > min_max_value[1]:
                    min_max_value[1] = nextLayer_min[0]
                    best_move = m_
                    if alphaBeta and min_max_value[1] >= min_max_value[0]:
                        return min_max_value[1], best_move
            return min_max_value[1], best_move

        if state.whose_move == 0:
            if depth == 0:
                min_max_value[0] = staticEval(state)
                return min_max_value, ""

            all_move_list = possibleMoves(state)
            for m in all_move_list:
                new_state_list.append((stateTransform(state, m), m))

            for s_, m_ in possibleStates(state):
                nextLayer_max, _ = min_max(s_, depth - 1)
                if nextLayer_max[1] < min_max_value[0]:
                    min_max_value[0] = nextLayer_max[1]
                    best_move = m_
                    if alphaBeta and min_max_value[1] >= min_max_value[0]:
                        return min_max_value[0], best_move
            return min_max_value[0], best_move

    else:
        if state.whose_move == 0: # my turn (white)

            if depth == 0:
                min_max_value[1] = staticEval(state)
                return min_max_value, ""

            all_move_list = possibleMoves(state)
            for m in all_move_list:
                new_state_list.append((stateTransform(state, m), m))

            for s_, m_ in new_state_list:
                nextLayer_min, _ = min_max(s_, depth - 1, min_max_value)
                if nextLayer_min[0] > min_max_value[1]:
                    min_max_value[1] = nextLayer_min[0]
                    best_move = m_
                    if alphaBeta and min_max_value[1] >= min_max_value[0]:
                        return min_max_value[1], best_move
            return min_max_value[1], best_move

        if state.whose_move == 1:
            if depth == 0:
                min_max_value[0] = staticEval(state)
                return min_max_value, ""

            all_move_list = possibleMoves(state)
            for m in all_move_list:
                new_state_list.append((stateTransform(state, m), m))

            for s_, m_ in possibleStates(state):
                nextLayer_max, _ = min_max(s_, depth - 1)
                if nextLayer_max[1] < min_max_value[0]:
                    min_max_value[0] = nextLayer_max[1]
                    best_move = m_
                    if alphaBeta and min_max_value[1] >= min_max_value[0]:
                        return min_max_value[0], best_move
            return min_max_value[0], best_move


def possibleMoves(state):
    # for each chess move = [(role, move), (role, move)...]
    move = []
    board = state.board
    for i in range(0, 8):
        for j in range(0, 8):
            piece = CODE_TO_INIT[state.board[i][j]]
            if piece != '-':
                whitePlayer = piece.isupper()
                if (state.whose_move and whitePlayer) or (not state.whose_move and not whitePlayer):
                    frozen = isFreeze(board, i, j, state.whose_move)
                    if not frozen:
                        if piece.lower() == 'p':
                            print('pawn moves')
                            horizontal_vertical(piece, board, i, j, move)
                        # Freezer cannot capture
                        elif piece.lower() == 'f':
                            eightDirMove(piece, board, i, j, move)
                        elif piece.lower() == 'l':
                            print("leaper moves")
                            eightDirMove(piece, board, i, j, move)
                        elif piece.lower() == 'w':
                            print('width')
                            eightDirMove(piece, board, i, j, move)
                        # capture moves included
                        elif piece.lower() == 'k':
                            print('killer king')
                            kingMove(board, i, j, move)
                        # capture moves included
                        elif piece.lower() == 'c':
                            print('coordinator')
                            eightDirMove(piece, board, i, j, move)
                        elif piece.lower() == 'i':
                            eightDirMove(piece, board, i, j, move)
                            # Can capture adjacent enemy King
                            for p in range(i-1, i+2):
                                for q in range(j-1, j+2):
                                    if 0 <= p < 8 and 0 <= q < 8:
                                        if board[p][q] == 12 or board[p][q] == 13:
                                            if isOppositePiece(board, i, j, p, q):
                                                move.append(piece + str(i) + str(j) + '-' + str(p) + str(q))
    return move


def isFreeze(board, i, j, whiteMove):
    freezer = -1
    imitator = -1
    if whiteMove: 
        freezer = 14 # black freezer
        imitator = 9 # black imitator
    else:
        freezer = 15 # white freezer
        imitator = 8 # white imitator

    IamFreezer = (whiteMove and board[i][j] == 14) or (not whiteMove and board[i][j] == 15)
    for p in range(i-1, i+2):
        for q in range(j-1, j+2):
            if 0 <= p < 8 and 0 <= q < 8 and (p is not i and q is not j):
                if board[p][q]== freezer or (IamFreezer and board[p][q] is imitator):
                    return True
    return False


def stateTransform(state, move):
    newState = deepcopy(state)
    piece = move[0]
    fromi = int(move[1])
    fromj = int(move[2])
    toi = int(move[4])
    toj = int(move[5])
    newState.board[fromi][fromj] = 0
    newState.board[toi][toj] = INIT_TO_CODE[piece]
    handleCapture(newState, piece, fromi, fromj, toi, toj)
    newState.whose_move ^= 1

    return newState


def staticEval(state):
    board = state.board
    white_score = 0
    black_score = 0

    for row in range(8):
        for column in range(8):
            piece = board[row][column]
            if (piece != 0):
                if (piece%2==1):
                    white_score += PIECE_PRIORITY[CODE_TO_INIT[piece].lower()]
                else:
                    black_score += PIECE_PRIORITY[CODE_TO_INIT[piece].lower()]

    if mySide == 1:
        return white_score - black_score
    else :
        return black_score - white_score




### moving function ###
def horizontal_vertical(piece, board, i, j, move):
    # count down
    iptr = i+1
    while iptr < 8:
        if board[iptr][j]==0:
            move.append(piece + str(i) + str(j) + '-' + str(iptr) + str(j))
        elif iptr <= 6:
            if piece.lower() == 'l' and board[iptr+1][j] == 0 \
                        or piece.lower() == 'i' and CODE_TO_INIT[board[iptr][j]].lower() == 'l' and board[iptr+1][j] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(iptr + 1) + str(j))
        else:
            break
        iptr += 1

    # count up
    iptr = i-1
    while iptr >= 0:
        if board[iptr][j]==0:
            move.append(piece + str(i) + str(j) + '-' + str(iptr) + str(j))
        elif iptr >= 1:
            if piece.lower() == 'l' and board[iptr-1][j] == 0 \
                        or piece.lower() == 'i' and CODE_TO_INIT[board[iptr][j]].lower() == 'l' and board[iptr-1][j] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(iptr - 1) + str(j))
        else:
            break
        iptr -= 1

    # count right
    jptr = j+1
    while jptr < 8:
        if board[i][jptr]==0:
            move.append(piece + str(i) + str(j) + '-' + str(i) + str(jptr))
        elif jptr <= 6:
            if piece.lower() == 'l' and board[i][jptr+1] == 0 \
                        or piece.lower() == 'i' and CODE_TO_INIT[board[i][jptr]].lower() == 'l' and board[i][jptr+1] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(i) + str(jptr+1))
        else:
            break
        jptr += 1

    # count left
    jptr = j-1
    while jptr >= 0:
        if board[i][jptr]==0:
            move.append(piece + str(i) + str(j) + '-' + str(i) + str(jptr))
        elif jptr >= 1:
            if piece.lower() == 'l' and board[i][jptr-1] == 0 \
                        or piece.lower() == 'i' and CODE_TO_INIT[board[i][jptr]].lower() == 'l' and board[i][jptr-1] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(i) + str(jptr-1))
        else:
            break
        jptr -= 1

def diagonal(piece, board, i, j, move):
    # count right down
    iptr = i+1
    jptr = j+1
    while iptr < 8 and jptr < 8:
        if board[iptr][jptr]==0:
            move.append(piece + str(i) + str(j) + '-' + str(iptr) + str(jptr))
        elif iptr <= 6 and jptr <= 6:
            if piece.lower() == 'l' and board[iptr+1][jptr+1] == 0 \
                        or piece.lower() == 'i' and CODE_TO_INIT[board[iptr][jptr]].lower() == 'l' and board[iptr+1][jptr+1] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(iptr+1) + str(jptr+1))
        else:
            break
        iptr += 1
        jptr += 1

    # count left up
    iptr = i-1
    jptr = j-1
    while iptr >= 0 and jptr >= 0:
        if board[iptr][jptr]==0:
            move.append(piece + str(i) + str(j) + '-' + str(iptr) + str(jptr))
        elif iptr >= 1  and jptr >= 1:
            if piece.lower() == 'l' and board[iptr-1][jptr-1] == 0 \
                        or piece.lower() == 'i' and CODE_TO_INIT[board[iptr][jptr]].lower() == 'l' and board[iptr-1][jptr-1] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(iptr-1) + str(jptr-1))
        else:
            break
        iptr -= 1
        jptr -= 1

    # count left down
    iptr = i+1
    jptr = j-1
    while iptr < 8 and jptr >= 0:
        if board[iptr][jptr]==0:
            move.append(piece + str(i) + str(j) + '-' + str(iptr) + str(jptr))
        elif iptr <= 6  and jptr >= 1:
            if piece.lower() == 'l' and board[iptr+1][jptr-1] == 0 \
                        or piece.lower() == 'i' and CODE_TO_INIT[board[iptr][jptr]].lower() == 'l' and board[iptr+1][jptr-1] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(iptr+1) + str(jptr-1))
        else:
            break
        iptr += 1
        jptr -= 1

    # count right up
    iptr = i-1
    jptr = j+1
    while iptr >= 0 and jptr < 8:
        if board[iptr][jptr]==0:
            move.append(piece + str(i) + str(j) + '-' + str(iptr) + str(jptr))
        elif iptr >= 1 and jptr <= 6:
            if piece.lower() == 'l' and board[iptr-1][jptr+1] == 0 \
                        or piece.lower() == 'i' and CODE_TO_INIT[board[iptr][jptr]].lower() == 'l' and board[iptr-1][jptr+1] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(iptr-1) + str(jptr+1))
        else:
            break
        iptr -= 1
        jptr += 1

def eightDirMove(piece, board, i, j, move):
    horizontal_vertical(piece, board, i, j, move)
    diagonal(piece, board, i, j, move)

def kingMove(board, i, j, move):
    for p in range(i-1, i+2):
        for q in range(j-1, j+2):
            if 0 <= p < 8 and 0 <= q < 8 and board[i][j]%2 != board[p][q]%2: # only consider eating opponent
                move.append(CODE_TO_INIT[board[i][j]] + str(i) + str(j) + '-' + str(p) + str(q))


### capture function ###
def handleCapture(newState, piece, fromi, fromj, toi, toj):
    board = newState.board
    if piece.lower() == 'p':
        return pincherCapture(board, toi, toj)
    elif piece.lower() == 'i':
        return pincherCapture(board, toi, toj, True) or \
                withdrawerCapture(board, fromi, fromj, toi, toj, True) or \
                coordinatorCapture(board, toi, toj, newState.whose_move, True)
    elif piece.lower() == 'w':
        return withdrawerCapture(board, fromi, fromj, toi, toj)
    elif piece.lower() == 'c':
        return coordinatorCapture(board, toi, toj, newState.whose_move)
    else:
        return False

def pincherCapture(board, r, c, is_imitator=False): ##
    for i in range(-1, 2, 2):   #return -1,1
        adj_r = r + i
        adj_c = c + i
        if 0 <= adj_r <= 7 and 0 <= adj_c <= 7:
            if isOppositePiece(board, r, c, r, adj_c):
                if board[r][adj_c + i] % 2 == board[r][c] % 2: #check two stpes from the pincher
                    if is_imitator:
                        if CODE_TO_INIT[board[r][adj_c]].lower()!='p':
                            return False
                        #imitator = CODE_TO_INIT[board[r][c]]
                        #return adjacentCapture(board, r, adj_c, r, c, imitator)

                    #return adjacentCapture(board, r, adj_c, r, c)
                    board[r][adj_c] = 0
                    return True
            if isOppositePiece(board, r, c, adj_r, c):
                if board[adj_r+i][c] % 2 == board[r][c] % 2: ##
                    if is_imitator:
                        if CODE_TO_INIT[board[adj_r][c]].lower()!='p':
                            return False
                        #imitator = CODE_TO_INIT[board[r][c]]
                        #return adjacentCapture(board, adj_r, c, r, c, imitator)

                    #return adjacentCapture(board, adj_r, c, r, c)
                    board[adj_r][c]=0
                    return True
    return False

def withdrawerCapture(board, old_i, old_j, new_i, new_j, is_imitator=False):
    capture_i = old_i - np.sign(new_i - old_i)
    capture_j = old_j - np.sign(new_j - old_j)

    if 0 <= capture_i < 8 and 0 <= capture_j < 8:
        if isOppositePiece(board, old_i, old_j, capture_i, capture_j):
            if is_imitator:
                if CODE_TO_INIT[board[capture_i][capture_j]].lower() != 'w':
                    return False
            board[capture_i][capture_j] = 0
            return True
    return False


def coordinatorCapture(board, new_i, new_j, whose_move, is_imitator=False):
    king_i = -1
    king_j = -1
    i = 0
    j = 0
    while king_i == -1:
        while king_j == -1 and j < 8:
            if board[i][j] == (12 + whose_move):
                king_i = i
                king_j = j
            else:
                j += 1
        i += 1
        j = 0

    capture1 = isOppositePiece(board, new_i, new_j, king_i, new_j)
    capture2 = isOppositePiece(board, new_i, new_j, new_i, king_j)

    if is_imitator:
        capture1 = capture1 and (board[king_i][new_j] == 4 or board[king_i][new_j] == 5)
        capture2 = capture2 and (board[new_i][king_j] == 4 or board[new_i][king_j] == 5)
    if capture1:
        board[king_i][new_j] = 0
    if capture2:
        board[new_i][king_j] = 0
    return capture1 or capture2


def isOppositePiece(board, r1, c1, r2, c2):
    return board[r1][c1] != 0 and board[r2][c2] != 0 and board[r1][c1] % 2 != board[r2][c2] % 2