import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense


class Split(tf.keras.layers.Layer):

    def __init__(self, num_or_size_splits, axis=0, num=None, name='split'):
        super(Split, self).__init__()
        self._num_split = num_or_size_splits
        self._axis = axis
        self._num = num
        self._name = name

    def call(self, inputs, **kwargs):
        return tf.split(inputs, self._num_split, self._axis, self._num, self._name)


class Dueling(tf.keras.layers.Layer):

    def call(self, value, advantage):
        return value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))


class DQNet(tf.keras.Model):

    def __init__(self, num_actions, conv_out_dim):
        super(DQNet, self).__init__()
        with tf.name_scope(name=self.name):
            # Feature-detecting convolutions
            self.conv1 = Conv2D(64, [4, 4])
            self.conv2 = Conv2D(conv_out_dim, [3, 3])

            # Split advantage and value functions
            self.split = Split(2, axis=3)
            self.value = Dense(1, name="value")
            self.advantage = Dense(num_actions, name="advantage")

            # Output: full q output
            self.q = Dueling(name="q_value")

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        v, a = self.split(x)
        v = self.value(v)
        a = self.advantage(a)
        return self.q(v, a)

    def predict_action(self, x, batch_size=32, verbose=0):
        q_values = self.predict(x, batch_size=batch_size, verbose=verbose)
        if q_values.shape[-1] > 1:
            return q_values.argmax(axis=-1)
        else:
            return (q_values > 0.5).astype('int32')
