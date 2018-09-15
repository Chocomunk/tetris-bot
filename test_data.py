import numpy as np
import tensorflow as tf
import random

class DataBuffer(object):

    def __init__(self, buffer_size):
        self._buffer = [(e, np.array([e, e+1])) for e in range(buffer_size)]
        self._buffer_size = buffer_size

    def sample_generator(self, batch_size):
        def generator_func():
            while True:
                samples = random.sample(self._buffer, batch_size)
                for sample in samples:
                    yield sample
        return generator_func


if __name__ == '__main__':
    databuf = DataBuffer(20)
    dataset = tf.data.Dataset.from_generator(databuf.sample_generator(2),
                                             output_types=(tf.float32, tf.float32),
                                             output_shapes=(
                                                 tf.TensorShape([]),
                                                 tf.TensorShape([None])
                                             )).batch(2)
    itera = dataset.make_initializable_iterator()
    inp = itera.get_next()
    lel = 1 - inp[0]

    with tf.Session() as sess:
        sess.run(itera.initializer)
        try:
            val = sess.run(lel)
            print(val)
        except tf.errors.OutOfRangeError:
            print("done?")
            print(databuf._buffer)
