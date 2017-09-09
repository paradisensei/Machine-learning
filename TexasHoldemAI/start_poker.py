from pypokerengine.api.game import setup_config, start_poker

from player import SimplePlayer, ExpertSystemPlayer, EmulatorPlayer

config = setup_config(max_round=3, initial_stack=1500, small_blind_amount=15)
expert_system_player = ExpertSystemPlayer()
simple_player = SimplePlayer()
simple_player_1 = SimplePlayer()
simple_player_1.set_action(0)
config.register_player(name="p1", algorithm=expert_system_player)
config.register_player(name="p2", algorithm=simple_player)
config.register_player(name="p3", algorithm=simple_player_1)
print(start_poker(config, verbose=1))
