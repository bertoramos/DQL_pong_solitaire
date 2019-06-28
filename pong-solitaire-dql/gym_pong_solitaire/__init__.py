
from gym.envs.registration import register

register(
    id='PongSolitaire-v0',
    entry_point='gym_pong_solitaire.envs:PongSolitaireEnv',
)
