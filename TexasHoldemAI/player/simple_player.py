from pypokerengine.players import BasePokerPlayer


class SimplePlayer(BasePokerPlayer):
    FOLD, CALL, MIN_RAISE, MAX_RAISE = 0, 1, 2, 3

    def __init__(self):
        super(SimplePlayer, self).__init__()
        self.action = 1  # set CALL as default action

    def set_action(self, action):
        self.action = action

    def declare_action(self, valid_actions, hole_card, round_state):
        if self.FOLD == self.action:
            return valid_actions[0]['action'], valid_actions[0]['amount']
        elif self.CALL == self.action:
            return valid_actions[1]['action'], valid_actions[1]['amount']
        elif self.MIN_RAISE == self.action:
            return valid_actions[2]['action'], valid_actions[2]['amount']['min']
        elif self.MAX_RAISE == self.action:
            return valid_actions[2]['action'], valid_actions[2]['amount']['max']
        else:
            raise Exception("Invalid action [ %s ] is set" % self.action)

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_game_start_message(self, game_info):
        pass

    def receive_street_start_message(self, street, round_state):
        pass


def setup_ai():
    return SimplePlayer()
