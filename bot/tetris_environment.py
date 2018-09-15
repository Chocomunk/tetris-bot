import numpy as np
from game.tetris import Tetris

default_action_mapping = [
    (-1, 0),  # 0: left
    (1,  0),  # 1: right
    (0,  1),  # 2: down
    'rotate'  # 3: rotate
]

class TetrisEnvironment(object):

    def __init__(self, tetris_instance=None, action_map=default_action_mapping, time_limit=-1, dt=(1000. / 120)):
        self.time_limit = time_limit
        self.dt = dt
        self.action_map = action_map
        if tetris_instance is None:
            self.tetris_instance = Tetris(time_limit=time_limit)
        else:
            self.tetris_instance = tetris_instance
            self.tetris_instance.time_limit = time_limit

    def step(self, action):
        real_action = self.action_map[action]
        if real_action is 'rotate':
            self.tetris_instance.rotate()
        else:
            self.tetris_instance.update_piece(real_action[0], real_action[1])

        pre_points = self.tetris_instance.points
        new_state, game_over = self.tetris_instance.update(self.dt)
        step_reward = (self.tetris_instance.points - pre_points)

        return new_state, step_reward, game_over

    def reset(self):
        self.tetris_instance = Tetris(self.time_limit)
        return self.tetris_instance.serve_image()
