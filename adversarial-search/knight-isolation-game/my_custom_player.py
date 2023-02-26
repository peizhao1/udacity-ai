from collections import defaultdict, Counter
import pickle
import random
import tqdm

from isolation import DebugState
from isolation import Isolation
from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    @classmethod
    def build_opening_book(cls, num_rounds=300000):

        def build_tree(state, book, depth=4):
            if depth <= 0 or state.terminal_test():
                return -simulate(state)
            action = random.choice(state.actions())
            reward = build_tree(state.result(action), book, depth - 1)
            book[state][action] += reward
            return -reward

        def simulate(state):
            while not state.terminal_test():
                state = state.result(random.choice(state.actions()))
            return -1 if state.utility(state.player()) < 0 else 1

        raw_book = defaultdict(Counter)
        print('Start building opening book:')
        for _ in tqdm.tqdm(range(num_rounds)):
            state = Isolation()
            build_tree(state, raw_book)

        opening_book = {k: max(v, key=v.get) for k, v in raw_book.items()}
        with open("data.pickle", 'wb') as f:
            pickle.dump(opening_book, f)

    def display_opening_book(self):
        state_0 = Isolation()
        action1 = self.data[state_0]
        state_1 = state_0.result(action1)
        print('Opening move')
        print(DebugState.from_state(state_1))
        action2 = self.data[state_1]
        state_2 = state_1.result(action2)
        print('Counter move')
        print(DebugState.from_state(state_2))

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # print(DebugState.from_state(state))
        if state.ply_count <= 4:
            # Try to refer the opening book
            # if hash(state) in self.data:
            #     self.queue.put(self.data[state])
            # else:
            #     self.queue.put(random.choice(state.actions()))
            # Randomly choose opening moves
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self._alpha_beta_search(state))

    def _alpha_beta_search(self, state, depth=4):

        def min_value(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value,
                            max_value(state.result(action),
                                      alpha, beta, depth-1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        def max_value(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value,
                            min_value(state.result(action),
                                      alpha, beta, depth-1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        def score(state):
            own_loc = state.locs[self.player_id]
            opp_loc = state.locs[1 - self.player_id]
            own_liberties = state.liberties(own_loc)
            opp_liberties = state.liberties(opp_loc)
            return len(own_liberties) - len(opp_liberties)

        return max(state.actions(),
                   key=lambda x: min_value(state.result(x),
                                           float("-inf"),
                                           float("inf"),
                                           depth - 1))


if __name__ == "__main__":
    CustomPlayer.build_opening_book()
    CustomPlayer(0).display_opening_book()
