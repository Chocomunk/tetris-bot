import os
import numpy as np
import tensorflow as tf
from collections import deque

from bot.tetris_environment import TetrisEnvironment
from bot.DQNModel import DQNet


class TetrisBot(object):

    def __init__(self, tetris_instance, model_path, name):
        """ Initialize a tetris AI object"""
        self.env = TetrisEnvironment(tetris_instance=tetris_instance)
        self.model_path = model_path
        self.model = DQNet((22,10,4), 5, name=name)
        self.state_queue = deque([np.zeros((22, 10)) for _ in range(3)] + [self.env.tetris_instance.serve_image()], 4)

    def init(self):
        print("Loading Saved Model")
        if os.path.exists(path=self.model_path):
            print("Loading Saved Model")
            latest = tf.train.latest_checkpoint(self.model_path)
            self.model.load_weights(latest)
        else:
            print("Error: no saved checkpoint")
            return
        print("Model loaded")

    def update(self):
        """ Called every drawn frame, lets the bot make decisions"""
        a = self.model.predict_action(np.stack(self.state_queue, axis=-1).reshape((1, 22, 10, 4)))[0]
        # a = self.model.predict_action(self.state_queue[0].reshape(1,22,10,1))[0]
        s1, r, d, _ = self.env.step(a)
        self.state_queue.appendleft(s1)

        return s1, d

