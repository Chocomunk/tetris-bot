import numpy as np
import tensorflow as tf
from collections import deque
from bot.tetris_environment import TetrisEnvironment
from bot.DDQNetwork import DDQNetwork


class TetrisBot(object):

    def __init__(self, tetris_instance, model_path, name):
        """ Initialize a tetris AI object"""
        self.env = TetrisEnvironment(tetris_instance=tetris_instance)
        self.model_path = model_path
        self.model = DDQNetwork(64, [22, 10], 4, name=name)
        self.state_queue = deque([np.zeros((22, 10)) for _ in range(3)] + [self.env.tetris_instance.serve_image()], 4)
        self.saver = tf.train.Saver()

    def init(self, sess):
        print("Loading Saved Model")
        checkpt = tf.train.get_checkpoint_state(self.model_path)
        if checkpt is None:
            print("Error: no saved checkpoint")
            return
        print("Found model at {}".format(checkpt))
        self.saver.restore(sess, checkpt.model_checkpoint_path)
        print("Model loaded")

    def update(self, sess):
        """ Called every drawn frame, lets the bot make decisions"""
        a, qs, con = sess.run([self.model.best_action, self.model.outputQ, self.model.conv_out],
                              feed_dict={self.model.state_image: [np.stack(self.state_queue, axis=-1)]})
        a = a[0]
        print("{}: {}, {}".format(a, qs, np.sum(con)))
        s1, r, d = self.env.step(a)
        self.state_queue.appendleft(s1)

        return s1, d

