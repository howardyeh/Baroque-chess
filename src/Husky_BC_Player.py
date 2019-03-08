'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

'''

import BC_state_etc as BC

def makeMove(currentState, currentRemark, timelimit):

    # Compute the new state for a move.
    # This is a placeholder that just copies the current state.
    max_depth = 10
    difjadsoi = iterative_deepening(currentState, max_depth, timelimit)
        

    # Fix up whose turn it will be.
    newState.whose_move = 1 - currentState.whose_move
    
    # Construct a representation of the move that goes from the
    # currentState to the newState.
    # Here is a placeholder in the right format but with made-up
    # numbers:
    move = ((6, 4), (3, 4))

    # Make up a new remark
    newRemark = "I'll think harder in some future game. Here's my move"

    return [[move, newState], newRemark]

def nickname():
    return "Husky"

def introduce():
    return "I'm a husky. Wolf wolf!"

def prepare(player2Nickname):
    pass

def iterative_deepening(currentState, max_depth, timelimit):
    timesup = False
    for depth in range(max_depth):
        if timesup:
            break;
        else:
            value, timesup = mini_max(currentState, depth, (1000000, -1000000))

    # TODO: add in time factor, break when time is up, maybe add a time stamp in min_max()
    #       and compare to the time from this part, if times up, break.


def min_max(state, depth, min_max_value):
    
    if state.whose_move == 1:
        if depth == 0:
            min_max_value[1] = staticEval(state)
            return min_max_value
        for s_ in possibleStates(state):
            nextLayer_min = mini_max(s_, depth - 1, min_max_value)
            if nextLayer_min > min_max_value[1]:
                min_max_value[1] = nextLayer_min
        return min_max_value

    if state.whose_move == 0:
        if depth == 0:
            min_max_value[0] = staticEval(state)
            return min_max_value
        for s_ in possibleStates(state):
            nextLayer_max = mini_max(s_, depth - 1)
            if nextLayer_max < min_max_value[0]:
                min_max_value[0] = nextLayer_max
        return min_max_value

    # TODO: alpha beta pruning, when min_max_value[1]>min_max_value[0], return


def possibleStates(state):
    new_state_list = []
    
    for m in possibleMoves(state):
        new_state_list.append(stateTransform(state, m))

    return new_state_list

    # TODO: check every chess with its possible moves to a new state

def possibleMoves(state):
    # for each chess move = [(role, move), (role, move)...]
    move = []
    return move

    # TODO: calculate the possible moves of a state

def stateTransform(state, move):
    return new_state

    # TODO: calculate the new state from the old state by taking move

def staticEval(state):
    return 0 

    # TODO: calculate the value of the board state


# we still need different move function for each chess
# we still need different capture function for each chess
