from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate


class ExpertSystemPlayer(BasePokerPlayer):
    # big blind
    BB = 30
    # possible player's actions
    FOLD, LIMP, BET, MIN_RAISE, MAX_RAISE = 0, 1, 2, 3, 4

    def declare_action(self, valid_actions, hole_card, round_state, bot_state=None):
        # find out number of active players
        _nb_player = 0
        for player_info in round_state['seats']:
            if player_info['uuid'] == self.uuid:
                self.stack = player_info['stack']
            if player_info['state'] != 'folded':
                _nb_player += 1
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=550, nb_player=_nb_player, hole_card=gen_cards(hole_card),
            community_card=gen_cards(round_state['community_card'])
        )
        if round_state['street'] == 'preflop':
            next_action = self.__action_of_win_rate(win_rate, 0.1, 0.15, 0.2, 0.25)
        else:
            next_action = self.__action_of_win_rate(win_rate, 0.1, 0.15, 0.23, 0.3)
        return self.__action(next_action, valid_actions)

    def __action_of_win_rate(self, win_rate, p1, p2, p3, p4):
        if win_rate < p1:
            return self.FOLD
        if win_rate < p2:
            return self.LIMP
        if win_rate < p3:
            return self.BET
        if win_rate < p4:
            return self.MIN_RAISE
        return self.MAX_RAISE

    def __action(self, next_action, valid_actions):
        to_call = valid_actions[1]['amount']
        if self.stack <= to_call:
            if self.stack <= self.BB * 10 and next_action >= self.MIN_RAISE:
                return to_tuple(valid_actions[1])
            else:
                return to_tuple(valid_actions[0])
        if next_action == self.FOLD and to_call == 0:
            return to_tuple(valid_actions[1])
        if next_action == self.LIMP and to_call <= self.BB:
            return to_tuple(valid_actions[1])
        if next_action == self.BET:
            if to_call <= self.BB:
                return 'raise', valid_actions[2]['amount']['min']
            if to_call <= self.BB * 3:
                return to_tuple(valid_actions[1])
        if self.stack <= self.BB * 10:
            if next_action >= self.MIN_RAISE:
                return 'raise', valid_actions[2]['amount']['max']
            else:
                return to_tuple(valid_actions[0])
        if next_action == self.MIN_RAISE:
            if to_call <= self.BB * 3:
                return 'raise', valid_actions[2]['amount']['min']
            if to_call <= self.BB * 5:
                return to_tuple(valid_actions[1])
        if next_action == self.MAX_RAISE:
            raise_amount = valid_actions[2]['amount']['min'] * 3
            if raise_amount > valid_actions[2]['amount']['max']:
                raise_amount = valid_actions[2]['amount']['max']
            return 'raise', raise_amount
        # otherwise FOLD
        return to_tuple(valid_actions[0])

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_start_message(self, game_info):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


# utilities
def to_tuple(action):
    return action['action'], action['amount']


def setup_ai():
    return ExpertSystemPlayer()
