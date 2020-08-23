import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Dropout, InputLayer
from tensorflow.keras.models import Model


class Split(tf.keras.layers.Layer):

    def __init__(self, num_or_size_splits, axis=0, num=None, name='split', *args, **kwargs):
        super(Split, self).__init__(trainable=False, *args, **kwargs)
        self._num_split = num_or_size_splits
        self._axis = axis
        self._num = num
        self._name = name

    def call(self, inputs, **kwargs):
        return tf.split(inputs, self._num_split, self._axis, self._num, self._name)


class Dueling(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super(Dueling, self).__init__(trainable=False, *args, **kwargs)

    def call(self, value, advantage):
        return value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))


class QPrediction(tf.keras.layers.Layer):

    def __init__(self, num_actions, *args, **kwargs):
        super(QPrediction, self).__init__(trainable=False, *args, **kwargs)
        self.num_actions = num_actions

    def call(self, actions, q_values):
        actions_onehot = tf.one_hot(actions, self.num_actions, dtype=tf.float32)
        return tf.reduce_sum(tf.multiply(q_values, actions_onehot), axis=1)


class DQNet(tf.keras.Model):

    def __init__(self, input_shape, num_actions, trainable=True, *args, **kwargs):
        super(DQNet, self).__init__(*args, **kwargs)
        self.num_actions = num_actions
        self.input_layer = InputLayer(input_shape=input_shape)

        # Feature-detecting convolutions
        self.conv1 = Conv2D(64, [4, 4], trainable=trainable, activation='relu')
        self.conv2 = Conv2D(64, [3, 3], trainable=trainable, activation='relu')
        self.drop1 = Dropout(.2)

        # Fully Connected Layers
        self.flatc = Flatten()
        self.ffnn1 = Dense(3000, activation='relu')
        self.ffnn2 = Dense(1000, activation='relu')
        self.drop2 = Dropout(.2)

        # Split advantage and value functions
        # self.split = Split(2, axis=3)
        self.split = Split(2, axis=1)
        # self.value_flatten = Flatten()
        # self.advantage_flatten = Flatten()
        self.value = Dense(1, trainable=trainable, name="value")
        self.advantage = Dense(num_actions, trainable=trainable, name="advantage")

        # Output: full q output
        self.q = Dueling(name="q_output")

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        if not self.trainable:
            training = False
        x = self.input_layer(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop1(x, training=training)
        x = self.flatc(x)
        x = self.ffnn1(x)
        x = self.ffnn2(x)
        x = self.drop2(x, training=training)
        v, a = self.split(x)
        # v = self.value_flatten(v)
        # a = self.advantage_flatten(a)
        v = self.value(v)
        a = self.advantage(a)
        return self.q(v, a)

    @tf.function
    def double_q(self, inputs, MainModel, batch_size):
        best_actions = MainModel.predict_action(inputs, batch_size=batch_size)
        target_qs = self(inputs, training=False)
        row_indices = tf.range(batch_size)
        full_indices = tf.stack([row_indices, best_actions], axis=1)
        return tf.gather_nd(target_qs, full_indices)

    @tf.function
    def predict_action(self, x, batch_size=32, verbose=0):
        # q_values = self.predict(x, batch_size=batch_size, verbose=verbose)
        q_values = self(x, training=False)
        if q_values.shape[-1] > 1:
            return tf.argmax(q_values, axis=-1, output_type=tf.int32)
            # return q_values.argmax(axis=-1)
        else:
            return (q_values > 0.5).astype('int32')

    def trainable_model(self, image_shape):
        image = Input(shape=image_shape, name="input_image")
        action = Input(shape=[], dtype=tf.int32, name="action")
        qs = self(image, training=True)
        q_pred = QPrediction(num_actions=self.num_actions, name="q_value")(action, qs)
        return Model(inputs=[image, action], outputs=q_pred)
