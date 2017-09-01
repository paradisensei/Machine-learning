from pypokerengine.api.game import setup_config, start_poker

from player import SimplePlayer, TightPlayer, EmulatorPlayer

config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)
tight_player = TightPlayer()
simple_player = SimplePlayer()
config.register_player(name="p1", algorithm=tight_player)
config.register_player(name="p2", algorithm=EmulatorPlayer(tight_player, simple_player))
print start_poker(config, verbose=1)
