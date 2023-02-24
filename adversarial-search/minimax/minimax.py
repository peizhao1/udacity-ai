
from minimax_helper import min_value, max_value


def minimax_decision(gameState):
    """ Return the move along a branch of the game tree that
    has the best possible value.  A move is a pair of coordinates
    in (column, row) order corresponding to a legal move for
    the searching player.
    
    You can ignore the special case of calling this function
    from a terminal state.
    """
    return max(
        gameState.actions(),
        key=lambda m: min_value(gameState.result(m))
    )
