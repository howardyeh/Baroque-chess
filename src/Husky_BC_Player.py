'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

'''

from BC_state_etc import *
from copy import deepcopy
import numpy as np
from random import shuffle
import time
import uuid

mySide = None
overNot = False
startTime = 0
COUNT = 0
PIECE_PRIORITY = {'k':10000, 'f':80, 'c':60, 'i':40, 'w':30, 'l':20, 'p':10}
ZobristTable = np.random.randint(0, 9223372036854775806, size=(8,8,15), dtype=np.int64)
evaluationTable = dict()


def computeHash(board):
    h = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] != 0:
                h ^= ZobristTable[i][j][board[i][j]-1]
    return h


def nickname():
    return "Husky"

def introduce():
    return "I'm a husky. Wolf wolf!"

def prepare(player2Nickname):
    print(ZobristTable)
    

def makeMove(currentState, currentRemark, timelimit):

    # Compute the new state for a move.
    # This is a placeholder that just copies the current state.
    global mySide, startTime, COUNT
    if mySide is None:
        mySide = currentState.whose_move

    max_depth = 4
    startTime = time.time()

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
    # newRemark = "I'll think harder in some future game. Here's my move"
    REMARK=["I'll think harder in some future game. Here's my move",
            "Ha Ha, That's my move, Woof",
            "Do you need more time, Woof?",
            "Woof Woof!",
            "Please quickly Woof!",
            "Are you a cat Woof?",
            "Woof, I need some food!",
            "Woof, great move",
            "Woof, I'll hunt you down",
            "Woof, Can you think of a better move?"]
    COUNT+=1    
    newRemark = REMARK[COUNT%10]

    return [[((int(move[1]), int(move[2])),(int(move[4]), int(move[5]))), newState], newRemark]


def iterative_deepening(currentState, max_depth, timelimit):
    timesup = False
    best_move = ""
    hashValue = computeHash(currentState.board)
    for depth in range(max_depth):
        # print('max depth = ', depth+1)
        value, move = min_max(currentState, depth+1, [1000000, -1000000], timelimit, hashValue)
        if move is not None:
            best_move = move
        else:
            print('timesup! the deepest level = ', depth)
            return best_move # is move is None, time is up

    return best_move


def min_max(state, depth, min_max_value, timelimit, hashValue, alphaBeta = True):

    global evaluationTable

    all_move_list = []
    new_state_list = []
    best_move = ""
    # print('depth = ', depth)

    if(time.time() - startTime) >= (timelimit - 0.01):
        return (None, None), None

    if mySide == 1:
        if state.whose_move == 1: # my turn (white)
            # print('my turn')
            if depth == 0:
                if hashValue in evaluationTable:
                    # print('find hash')
                    min_max_value[1] = evaluationTable[hashValue]
                else:
                    min_max_value[1] = staticEval(state, timelimit)
                    if min_max_value[1] != None:
                        evaluationTable[hashValue] = min_max_value[1]
                # print(min_max_value[1])
                return min_max_value, ""

            all_move_list = possibleMoves(state)

            for m in all_move_list:
                new_state_list.append((stateTransform(state, m), m))
            shuffle(new_state_list)

            for s_, m_ in new_state_list:
                if over_or_not(s_.board, state.whose_move):
                    # print(1-mySide, 'its over~~')
                    min_max_value[1] = 20000
                    return min_max_value, m_

                ###### Zobrist Hashing ########################################
                hashValue ^= ZobristTable[int(m_[1])][int(m_[2])][INIT_TO_CODE[m_[0]]-1]
                hashValue ^= ZobristTable[int(m_[4])][int(m_[5])][INIT_TO_CODE[m_[0]]-1]
                ###############################################################
                nextLayer_min, _ = min_max(s_, depth - 1, deepcopy(min_max_value), timelimit, hashValue)
                ###### Zobrist Hashing inverse ################################
                hashValue ^= ZobristTable[int(m_[4])][int(m_[5])][INIT_TO_CODE[m_[0]]-1]
                hashValue ^= ZobristTable[int(m_[1])][int(m_[2])][INIT_TO_CODE[m_[0]]-1]
                ###############################################################
                
                if nextLayer_min[0] == None: # time's up
                    return (None, None), None

                if nextLayer_min[0] > min_max_value[1]:
                    min_max_value[1] = nextLayer_min[0]
                    best_move = m_
                    # print('best move = ', best_move)
                    if alphaBeta and min_max_value[1] >= min_max_value[0]:
                        # print('cut')
                        return min_max_value, best_move
            return min_max_value, best_move

        else:
            # print('black turn')
            if depth == 0:
                if hashValue in evaluationTable:
                    # print('find hash')
                    min_max_value[0] = evaluationTable[hashValue]
                else:
                    min_max_value[0] = staticEval(state, timelimit)
                    if min_max_value[0] != None:
                        evaluationTable[hashValue] = min_max_value[0]
                # print(min_max_value[0])
                return min_max_value, ""

            all_move_list = possibleMoves(state)
            for m in all_move_list:
                new_state_list.append((stateTransform(state, m), m))
            shuffle(new_state_list)

            for s_, m_ in new_state_list:
                if over_or_not(s_.board, state.whose_move):
                    # print(mySide,'its over~~')
                    min_max_value[0] = -20000
                    return min_max_value, m_

                ###### Zobrist Hashing ########################################
                hashValue ^= ZobristTable[int(m_[1])][int(m_[2])][INIT_TO_CODE[m_[0]]-1]
                hashValue ^= ZobristTable[int(m_[4])][int(m_[5])][INIT_TO_CODE[m_[0]]-1]
                ###############################################################
                nextLayer_max, _ = min_max(s_, depth - 1, deepcopy(min_max_value), timelimit, hashValue)
                ###### Zobrist Hashing inverse ################################
                hashValue ^= ZobristTable[int(m_[4])][int(m_[5])][INIT_TO_CODE[m_[0]]-1]
                hashValue ^= ZobristTable[int(m_[1])][int(m_[2])][INIT_TO_CODE[m_[0]]-1]
                ###############################################################

                if nextLayer_max[1] == None: # time's up
                    return (None, None), None
                
                if nextLayer_max[1] < min_max_value[0]:
                    min_max_value[0] = nextLayer_max[1]
                    best_move = m_
                    if alphaBeta and min_max_value[1] >= min_max_value[0]:
                        # print('cut')
                        return min_max_value, best_move
            return min_max_value, best_move

    # else:
    #     if state.whose_move == 0: # my turn (white)

    #         if depth == 0:
    #             min_max_value[1] = staticEval(state, timelimit)
    #             return min_max_value, ""

    #         all_move_list = possibleMoves(state)
    #         for m in all_move_list:
    #             new_state_list.append((stateTransform(state, m), m))

    #         for s_, m_ in new_state_list:
    #             nextLayer_min, _ = min_max(s_, depth - 1, deepcopy(min_max_value), timelimit, hashValue)
                
    #             if nextLayer_min == None: # time's up
    #                 return None, None

    #             if nextLayer_min[0] > min_max_value[1]:
    #                 min_max_value[1] = nextLayer_min[0]
    #                 best_move = m_
    #                 if alphaBeta and min_max_value[1] >= min_max_value[0]:
    #                     return min_max_value, best_move
    #         return min_max_value, best_move

    #     else:
    #         if depth == 0:
    #             min_max_value[0] = staticEval(state, timelimit)
    #             return min_max_value, ""

    #         all_move_list = possibleMoves(state)
    #         for m in all_move_list:
    #             new_state_list.append((stateTransform(state, m), m))

    #         for s_, m_ in new_state_list:
    #             nextLayer_max, _ = min_max(s_, depth - 1, deepcopy(min_max_value), timelimit, hashValue)

    #             if nextLayer_max == None: # time's up
    #                 return None, None

    #             if nextLayer_max[1] < min_max_value[0]:
    #                 min_max_value[0] = nextLayer_max[1]
    #                 best_move = m_
    #                 if alphaBeta and min_max_value[1] >= min_max_value[0]:
    #                     return min_max_value, best_move
    #         return min_max_value, best_move

def over_or_not(board, whose_move):
    for i in range(8):
        for j in range(8):
            if board[i][j] == (13 - whose_move):
                return False
    return True

def possibleMoves(state):
    # for each chess move = [(role, move), (role, move)...]
    move = []
    board = state.board
    for i in range(0, 8):
        for j in range(0, 8):
            if state.board[i][j] != 0 and state.board[i][j] % 2 == state.whose_move: 
                piece = CODE_TO_INIT[state.board[i][j]]
                frozen = isFreeze(board, i, j, state.whose_move)
                if not frozen:
                    if piece.lower() == 'p':
                        # print('pawn moves')
                        horizontal_vertical(piece, board, i, j, move)
                    # Freezer cannot capture
                    elif piece.lower() == 'f':
                        eightDirMove(piece, board, i, j, move)
                    elif piece.lower() == 'l':
                        # print("leaper moves")
                        eightDirMove(piece, board, i, j, move)
                    elif piece.lower() == 'w':
                        # print('width')
                        eightDirMove(piece, board, i, j, move)
                    # capture moves included
                    elif piece.lower() == 'k':
                        # print('killer king')
                        kingMove(board, i, j, move)
                    # capture moves included
                    elif piece.lower() == 'c':
                        # print('coordinator')
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
    if whiteMove:  # 1
        freezer = 14 # black freezer
        imitator = 9 # black imitator
    else:
        freezer = 15 # white freezer
        imitator = 8 # white imitator

    IamFreezer = (whiteMove and board[i][j] == 14) or (not whiteMove and board[i][j] == 15)
    for p in range(i-1, i+2):
        for q in range(j-1, j+2):
            if 0 <= p < 8 and 0 <= q < 8:
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


def staticEval(state, timelimit):
    global startTime
    board = state.board
    white_score = 0
    black_score = 0

    for row in range(8):
        for column in range(8):
            # if(time.time() - startTime) >= (timelimit - 0.005):
            #     return None
            piece = board[row][column]
            if (piece != 0):
                if (piece%2==1):
                    white_score += (PIECE_PRIORITY[CODE_TO_INIT[piece].lower()])
                else:
                    black_score += (PIECE_PRIORITY[CODE_TO_INIT[piece].lower()]+10)

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

        elif (piece.lower() == 'l' and iptr <= 6) \
                or (piece.lower() == 'i' and iptr <= 6 and CODE_TO_INIT[board[iptr][j]].lower() == 'l'):
            if isOppositePiece(board, i, j, iptr, j) and board[iptr+1][j] == 0:
                # print('leaper eat', iptr, j)
                move.append(piece + str(i) + str(j) + '-' + str(iptr + 1) + str(j))
            break
        else:
            break
        iptr += 1

    # count up
    iptr = i-1
    while iptr >= 0:
        if board[iptr][j]==0:
            move.append(piece + str(i) + str(j) + '-' + str(iptr) + str(j))

        elif (piece.lower() == 'l' and iptr > 0) \
                or (piece.lower() == 'i' and iptr > 0 and CODE_TO_INIT[board[iptr][j]].lower() == 'l'):
            if isOppositePiece(board, i, j, iptr, j) and board[iptr-1][j] == 0:
                # print('leaper eat', iptr, j)
                move.append(piece + str(i) + str(j) + '-' + str(iptr - 1) + str(j))
            break
        else:
            break
        iptr -= 1

    # count right
    jptr = j+1
    while jptr < 8:
        if board[i][jptr]==0:
            move.append(piece + str(i) + str(j) + '-' + str(i) + str(jptr))

        elif (piece.lower() == 'l' and jptr <= 6) \
                or (piece.lower() == 'i' and jptr <= 6 and CODE_TO_INIT[board[i][jptr]].lower() == 'l'):
            if isOppositePiece(board, i, j, i, jptr) and board[i][jptr+1] == 0:
                # print('leaper eat', i, jptr)
                move.append(piece + str(i) + str(j) + '-' + str(i) + str(jptr+1))
            break
        else:
            break
        jptr += 1

    # count left
    jptr = j-1
    while jptr >= 0:
        if board[i][jptr]==0:
            move.append(piece + str(i) + str(j) + '-' + str(i) + str(jptr))

        elif (piece.lower() == 'l' and jptr > 0) \
                or (piece.lower() == 'i' and jptr > 0 and CODE_TO_INIT[board[i][jptr]].lower() == 'l'):
            if isOppositePiece(board, i, j, i, jptr) and board[i][jptr-1] == 0:
                # print('leaper eat', i, jptr)
                move.append(piece + str(i) + str(j) + '-' + str(i) + str(jptr-1))
            break
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

        elif (piece.lower() == 'l' and  iptr <= 6 and jptr <= 6) \
                or (piece.lower() == 'i' and iptr <= 6 and jptr <= 6 and CODE_TO_INIT[board[i][jptr]].lower() == 'l'):
            if isOppositePiece(board, i, j, iptr, jptr) and board[iptr+1][jptr+1] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(iptr+1) + str(jptr+1))
            break
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

        elif (piece.lower() == 'l' and  iptr > 0 and jptr > 0) \
                or (piece.lower() == 'i' and iptr > 0 and jptr > 0 and CODE_TO_INIT[board[i][jptr]].lower() == 'l'):
            if isOppositePiece(board, i, j, iptr, jptr) and board[iptr-1][jptr-1] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(iptr-1) + str(jptr-1))
            break
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

        elif (piece.lower() == 'l' and  iptr <= 6 and jptr > 0) \
                or (piece.lower() == 'i' and iptr <= 6 and jptr > 0 and CODE_TO_INIT[board[i][jptr]].lower() == 'l'):
            if isOppositePiece(board, i, j, iptr, jptr) and board[iptr+1][jptr-1] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(iptr+1) + str(jptr-1))
            break
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

        elif (piece.lower() == 'l' and  iptr > 0 and jptr <= 6) \
                or (piece.lower() == 'i' and iptr > 0 and jptr <= 6 and CODE_TO_INIT[board[i][jptr]].lower() == 'l'):
            if isOppositePiece(board, i, j, iptr, jptr) and board[iptr-1][jptr+1] == 0:
                move.append(piece + str(i) + str(j) + '-' + str(iptr-1) + str(jptr+1))
            break
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
    capture_or_not = False
    if piece.lower() == 'p':
        capture_or_not = pincherCapture(board, toi, toj)
        # if capture_or_not:
            # print('Pincher capture')
        return capture_or_not
    elif piece.lower() == 'i':
        capture_or_not = pincherCapture(board, toi, toj, True) or \
                withdrawerCapture(board, fromi, fromj, toi, toj, True) or \
                coordinatorCapture(board, toi, toj, newState.whose_move, True) or \
                leaperCapture(board, fromi, fromj, toi, toj, True)
        # if capture_or_not:
            # print('imitator capture')
        return capture_or_not
    elif piece.lower() == 'w':
        capture_or_not = withdrawerCapture(board, fromi, fromj, toi, toj)
        # if capture_or_not:
            # print('withdrawer capture')
        return capture_or_not
    elif piece.lower() == 'c':
        capture_or_not = coordinatorCapture(board, toi, toj, newState.whose_move)
        # if capture_or_not:
            # print('coordinator capture')
        return capture_or_not
    elif piece.lower() == 'l':
        capture_or_not = leaperCapture(board, fromi, fromj, toi, toj)
        # if capture_or_not:
            # print('leaper capture')
        return capture_or_not
    else:
        return False

def pincherCapture(board, r, c, is_imitator=False): ##
    global overNot

    isCapture = False
    isImiPinc = True

    for i in range(-1, 2, 2):   #return -1,1
        adj_r = r + i
        adj_c = c + i
        if 0 < adj_r < 7 and 0 < adj_c < 7:
            if isOppositePiece(board, r, c, r, adj_c):
                if board[r][adj_c+i] != 0 and board[r][adj_c+i] % 2 == board[r][c] % 2:
                    if is_imitator:
                        if CODE_TO_INIT[board[r][adj_c]].lower()!='p':
                            isImiPinc = False
                            isCapture = isCapture or isImiPinc
                        else:
                            isCapture = isCapture or isImiPinc
                    else:
                        isCapture = isCapture or True

                    if isImiPinc:    
                        if board[r][adj_c] == 12 or board[r][adj_c] == 13:
                            overNot = True
                        board[r][adj_c] = 0
            
            isImiPinc = True
            if isOppositePiece(board, r, c, adj_r, c):
                if board[adj_r+i][c] != 0 and board[adj_r+i][c] % 2 == board[r][c] % 2: ##
                    if is_imitator:
                        if CODE_TO_INIT[board[adj_r][c]].lower()!='p':
                            isImiPinc = False
                            isCapture = isCapture or isImiPinc
                        else:
                            isCapture = isCapture or isImiPinc
                    else: 
                        isCapture = isCapture or True
                        
                    if isImiPinc:
                        if board[r][adj_c] == 12 or board[r][adj_c] == 13:
                            overNot = True
                        board[adj_r][c]=0
                    
    return isCapture

def withdrawerCapture(board, old_i, old_j, new_i, new_j, is_imitator=False):
    global overNot
    capture_i = old_i - np.sign(new_i - old_i)
    capture_j = old_j - np.sign(new_j - old_j)

    if 0 <= capture_i < 8 and 0 <= capture_j < 8:
        if isOppositePiece(board, new_i, new_j, capture_i, capture_j):
            if is_imitator:
                if CODE_TO_INIT[board[capture_i][capture_j]].lower() != 'w':
                    return False
            if board[capture_i][capture_j] == 12 or board[capture_i][capture_j] == 13:
                overNot = True
            board[capture_i][capture_j] = 0
            return True
    return False

def coordinatorCapture(board, new_i, new_j, whose_move, is_imitator=False):
    global overNot
    king_i = -1
    king_j = -1
    i = 0
    j = 0
    while king_i == -1 and i < 8:
        while king_j == -1 and j < 8:
            if board[i][j] == (12 + whose_move):
                king_i = i
                king_j = j
            else:
                j += 1
        i += 1
        j = 0

    if king_i==-1 and king_j==-1:
        return False

    capture1 = isOppositePiece(board, new_i, new_j, king_i, new_j)
    capture2 = isOppositePiece(board, new_i, new_j, new_i, king_j)

    if is_imitator:
        capture1 = capture1 and (board[king_i][new_j] == 4 or board[king_i][new_j] == 5)
        capture2 = capture2 and (board[new_i][king_j] == 4 or board[new_i][king_j] == 5)
    if capture1:
        if board[king_i][new_j] == 12 or board[king_i][new_j] == 13:
            overNot = True
        board[king_i][new_j] = 0
    if capture2:
        if board[new_i][king_j] == 12 or board[new_i][king_j] == 13:
            overNot = True
        board[new_i][king_j] = 0
    return capture1 or capture2

def leaperCapture(board, old_i, old_j, new_i, new_j, is_imitator=False):
    global overNot
    capture_i = new_i - np.sign(new_i - old_i)
    capture_j = new_j - np.sign(new_j - old_j)
    if isOppositePiece(board, new_i, new_j, capture_i, capture_j):
        if is_imitator:
            if CODE_TO_INIT[board[capture_i][capture_j]].lower() != 'l':
                return False

        if board[capture_i][capture_j] == 12 or board[capture_i][capture_j] == 13:
            overNot = True
        board[capture_i][capture_j] = 0
        return True
    return False

def isOppositePiece(board, r1, c1, r2, c2):
    return board[r1][c1] != 0 and board[r2][c2] != 0 and board[r1][c1] % 2 != board[r2][c2] % 2

# def isOppositePiece(board, i1, j1, i2, j2):
#     return board[i1][j1] != 0 and board[i2][j2] != 0 and board[i1][j1] % 2 != board[i2][j2] % 2



##################################################################

# PIECE_VALUES = {'p': 1000, 'P': 1000, 'c': 2900, 'C': 2900, 'l': 4300, 'L': 4300, 'i': 5300, 'I': 5300,
#                 'w': 3100, 'W': 3100, 'k': 300000, 'K': 300000, 'f': 8200, 'F': 8200, '-': 0}

# def staticEval(state, timelimit):
#     # even numbers map to black pieces
#     global startTime
#     material_diff = 0
#     king_eval = 0
#     f_penalty = 0
#     queen_bonus = 0
#     l_bonus = 0
#     pa = 0
#     pincher_bonus = 0
#     for r in range(8):
#         for c in range(8):
#             if(time.time() - startTime) >= (timelimit - 0.005):
#                 return None
#             piece = state.board[r][c]
#             if piece == 12:
#                 king_eval -= king_safety(state, r, c)
#             if piece == 13:
#                 king_eval += king_safety(state, r, c)
#             if piece != 0:  # if it's not an empty square
#                 # pincher bonus
#                 pincher_bonus += pinch(state, r, c)

#                 # checking number of squares each piece can move to, l is index
#                 # max of 2000
#                 copy_state = deepcopy(state)
#                 num_moves = len(possibleMoves(copy_state))
#                 copy_state.whose_move ^= 1
#                 num_moves2 = len(possibleMoves(copy_state))
#                 if state.whose_move:
#                     pa = 40 * (num_moves - num_moves2)
#                 else:
#                     pa = 40 * (num_moves2 - num_moves)
#                 material_diff += PIECE_VALUES[CODE_TO_INIT[piece]]
#                 if piece == 14 or piece == 15:
#                     f_penalty += freeze_penalty(state, piece, r, c)
#                     # check freezer bonus
#                 elif piece == 10 or piece == 11:
#                     queen_bonus += w_bonus(state, piece, r, c)
#                     # check with-drawer bonus
#                 elif piece == 6 or piece == 7:
#                     l_bonus += leaper_bonus(state, piece, r, c)
#                     # check leaper bonus
#     return material_diff
#             #+ king_eval + f_penalty + queen_bonus + l_bonus + pa + pincher_bonus

# def freeze_penalty(state, freezer_color, freezer_locationx, freezer_locationy):
#     # accounted in piece activity
#     # Freezer: Immobilized pieces lose fourth of their value, kings get penalty of -2500
#     # checks all adjacent squares
#     bonus = 0
#     for i in range(freezer_locationx - 1, freezer_locationx + 2):  # horizontal
#         if i in range(0, 8) and i != freezer_locationx:  # checking if legal square and doesn't match freezer location
#             for j in range(freezer_locationy - 1, freezer_locationy + 2):  # vertical
#                 if j in range(0, 8) and j != freezer_locationy:  # checking if legal square
#                     if state.board[i][j] != 0:
#                         if state.board[i][j] % freezer_color != 0:
#                             if state.board[i][j] == 12 or state.board[i][j] == 13:
#                                 bonus -= 2500
#                             else:
#                                 bonus -= PIECE_VALUES[CODE_TO_INIT[state.board[i][j]]] / 4
#     if freezer_color == 0:  # if black piece
#         return -1 * bonus
#     else:
#         return bonus


# def coordinator_bonus(state, x, y):
#     # 1000 / 14 for every enemy piece on line with king
#     bonus = 0
#     for i in range(0, 8):
#         if y != i:
#             if isOppositePiece(state.board, x, y, x, i):
#                 bonus += 1000 / 14
#         if x != i:
#             if isOppositePiece(state.board, x, y, i, y):
#                 bonus += 1000 / 14
#     return bonus


# def w_bonus(state, w_color, w_locationx, w_locationy):
#     # Withdrawer: Gains bonus proportional to values of pieces next to it and if it can withdraw
#     # max bonus of 1200
#     bonus = 0
#     for i in range(w_locationx - 1, w_locationx + 2):  # horizontal
#         if i in range(0, 8) and i != w_locationx:  # checking if legal square and not square piece is on
#             for j in range(w_locationy - 1, w_locationy + 2):  # vertical
#                 if j in range(0, 8) and j != w_locationy:  # checking if legal square
#                     if state.board[i][j] != 0:  # if adjacent piece
#                         new_state = deepcopy(state)
#                         if withdrawerCapture(new_state.board, w_locationx, w_locationy, i, j):
#                             # print(state.board[i][j])
#                             bonus += PIECE_VALUES[CODE_TO_INIT[state.board[i][j]]] / 20
#     if w_color == 0:
#         return -1 * bonus
#     else:
#         return bonus


# # checks one piece at r, c
# def pinch(state, r, c):
#     bonus = 0
#     for i in range(-1, 2, 2):
#         for j in range(-1, 2, 2):
#             if 0 <= r + i < 8 and 0 <= r + j < 8:
#                 if state.board[r + i][r + j] != 0:  # if not empty square
#                     if isOppositePiece(state.board, r, c, r + i, r + j):
#                         bonus += (PIECE_VALUES[CODE_TO_INIT[state.board[r + i][r + j]]] -
#                                   PIECE_VALUES[CODE_TO_INIT[state.board[r][c]]]) / 70
#     return bonus


# def leaper_bonus(state, color, leaper_x, leaper_y):
#     # what are specific rules for leaper?
#     # Leaper: bonus if there is an empty square to leap to and capture them
#     # max is 1000
#     largest_piece_value = 0
#     for move in getMoves(state.board, 'l', leaper_x, leaper_y, ):
#         if move[1] == leaper_x:  # moving vertically
#             if move[2] > leaper_y:  # jumping up or right
#                 if PIECE_VALUES[state.board[move[1], move[2] - 1]] > largest_piece_value:
#                     largest_piece_value = PIECE_VALUES[state.board[move[1], move[2] - 1]]
#             else:  # jumping down or left
#                 if PIECE_VALUES[state.board[move[1], move[2] + 1]] > largest_piece_value:
#                     largest_piece_value = PIECE_VALUES[state.board[move[1], move[2] + 1]]
#     if color == 0:
#         return -1 * largest_piece_value / 8
#     else:
#         return largest_piece_value


# def king_safety(state, x_location, y_location):
#     # King: loses value when no pieces are around it, more pieces around it the better
#     # ranges from -1000 to 1000
#     ks = 0
#     # white king safety
#     for i in range(round(x_location / 3), round(x_location / 3) + 3):  # horizontal
#         for j in range(round(y_location / 3), round(y_location / 3) + 3):  # vertical
#             # if square is adjacent
#             if (y_location - j) + (x_location - i) == 1 or (y_location - j) + (x_location - i) == 2:
#                 if state.board[i][j] != 0:
#                     if state.board[i][j] % 2 == 0:  # black piece
#                         ks -= 500
#                     else:
#                         ks += 500
#         return ks

# def getMoves(board, piece, i, j, move=[]):

#     if piece.lower() == 'p':
#         # print('pawn moves')
#         horizontal_vertical(piece, board, i, j, move)
#     # Freezer cannot capture
#     elif piece.lower() == 'f':
#         eightDirMove(piece, board, i, j, move)
#     elif piece.lower() == 'l':
#         # print("leaper moves")
#         eightDirMove(piece, board, i, j, move)
#     elif piece.lower() == 'w':
#         # print('width')
#         eightDirMove(piece, board, i, j, move)
#     # capture moves included
#     elif piece.lower() == 'k':
#         # print('killer king')
#         kingMove(board, i, j, move)
#     # capture moves included
#     elif piece.lower() == 'c':
#         # print('coordinator')
#         eightDirMove(piece, board, i, j, move)
#     elif piece.lower() == 'i':
#         eightDirMove(piece, board, i, j, move)
#         # Can capture adjacent enemy King
#         for p in range(i-1, i+2):
#             for q in range(j-1, j+2):
#                 if 0 <= p < 8 and 0 <= q < 8:
#                     if board[p][q] == 12 or board[p][q] == 13:
#                         if isOppositePiece(board, i, j, p, q):
#                             move.append(piece + str(i) + str(j) + '-' + str(p) + str(q))
#     return move