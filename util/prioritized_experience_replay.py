import random
import os
import pickle
import numpy as np

from util.sum_tree_buffer import SumTreeBuffer


class PrioritizedExperienceReplay(object):

    def __init__(self, buffer_size=200000, dtype=object, epsilon=.001, alpha=0.6):
        self._dtype = dtype
        self._epsilon = epsilon
        self._alpha = alpha
        self._buffer = SumTreeBuffer(size=buffer_size, dtype=dtype)

    def _get_priority(self, error):
        return (error + self._epsilon) ** self._alpha

    def add(self, error, experience):
        priority = self._get_priority(error)
        self._buffer.add(priority, experience)

    def extend(self, other_PER):
        self._buffer.extend(other_PER._buffer)

    def update_errors(self, idx, error):
        priority = self._get_priority(error)
        self._buffer.update(idx, priority)

    def get_sample_generator(self, batch_size):
        def generator_func():
            while True:
                bucket_width = self._buffer.total() / batch_size

                for i in range(batch_size):
                    data, idx = 0, -1
                    while type(data) == int:
                        a = bucket_width * i
                        b = a + bucket_width
                        s = random.uniform(a, b)
                        idx, data = self._buffer.get_entry(s)
                    yield data + (idx,)

        return generator_func

    def save_file(self, file_path):
        dir_name = os.path.dirname(os.path.realpath(file_path))

        if not os.path.exists(dir_name):
            print("Creating new directory: {}". format(dir_name))
            os.makedirs(dir_name)
        else:
            print("Found existing directory: {}".format(dir_name))

        with open(file_path, 'wb') as file:
            print("Dumping data to file {}".format(file_path))
            pickle.dump(self, file)

    @staticmethod
    def from_file(file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError("Replay Buffer file not found: {}".format(file_path))

        with open(file_path, 'rb') as file:
            return pickle.load(file)