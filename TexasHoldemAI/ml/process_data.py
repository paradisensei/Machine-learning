import json
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import numpy as np

with open('game_0000.json') as file:
    data = json.load(file)

train = []

# big blind
BB = 30
# streets
PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3
# actions
FOLD, CALL, MIN_RAISE, RAISE, MAX_RAISE = 0, 1, 2, 3, 4


# utility functions
def hole_card():
    for seat in round_state["seats"]:
        if seat["uuid"] == w_uuid:
            return seat["hole_card"]


def get_players():
    for seat in round_state["seats"]:
        if seat["start_state"] != "folded":
            players.add(seat["uuid"])


def get_community_card(street):
    community_card = round_state["community_card"]
    return {
        "preflop": [],
        "flop": community_card[0:2],
        "turn": community_card[0:3],
        "river": community_card[0:4]
    }[street]


def get_street(s):
    return {
        "preflop": PREFLOP,
        "flop": FLOP,
        "turn": TURN,
        "river": RIVER
    }[s]


def get_move(move, move_amount):
    if move == "FOLD":
        return FOLD
    if move == "CALL":
        return CALL
    if move == "RAISE":
        if move_amount < 2*BB:
            return MIN_RAISE
        if move_amount < 4*BB:
            return RAISE
        else:
            return MAX_RAISE


def add(street, to_call, move, move_amount):
    train.append(get_street(street))
    win_rate = estimate_hole_card_win_rate(
        nb_simulation=550, nb_player=len(players), hole_card=gen_cards(w_hole_card),
        community_card=gen_cards(get_community_card(street))
    )
    train.append(win_rate)
    train.append(to_call // BB)
    train.append(get_move(move, move_amount))
#


def process_street(street):
    for action in streets[street]:
        uuid = action["uuid"]
        if uuid == w_uuid and "bot" in action:
            valid_actions = action["bot"]["valid_actions"]
            for valid_action in valid_actions:
                if valid_action["action"] == "call":
                    to_call = valid_action["amount"]
                    move = action["action"]
                    move_amount = action["amount"]
                    add(street, to_call, move, move_amount)
        if action["action"] == "FOLD":
            players.remove(uuid)


for r in data["rounds"]:
    round_state = r["round_state"]
    streets = round_state["action_histories"]
    players = set()
    # add all participating players
    get_players()
    # if pot is shared
    for w in r["winners"]:
        w_uuid = w["uuid"]
        w_hole_card = hole_card()
        street = ["preflop", "flop", "turn", "river"]
        for s in street:
            if s in streets:
                process_street(s)

train = np.array(train).reshape(len(train) // 4, 4)
# train = np.around(train, decimals=2)
print(train)
