import tensorflow as tf
import tensorflow.contrib.slim as slim


class DDQNetwork(object):

    def __init__(self, conv_out_dim, image_size, num_actions, name):
        self.num_actions = num_actions
        self.image_size = image_size

        with tf.name_scope(name=name):
            self.state_image = tf.placeholder(shape=[None, *image_size, 1], dtype=tf.float32)

            # Feature-detecting convolutions
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                self.conv_out = slim.stack(self.state_image, slim.conv2d, [
                    # num_outputs, kernel_size, stride
                    (16,           [2, 2], [1, 1]),
                    (64,           [2, 2], [1, 1]),
                    (64,           [3, 3], [1, 1]),
                    (conv_out_dim, [3, 3], [1, 1])
                ])

            xavier_init = slim.xavier_initializer()

            # Split advantage and value functions
            self.conv_flatten = slim.flatten(self.conv_out)

            self.value_fc = tf.layers.dense(inputs=self.conv_flatten, units=512, kernel_initializer=xavier_init)
            self.advantage_fc = tf.layers.dense(inputs=self.conv_flatten, units=512, kernel_initializer=xavier_init)

            self.value = tf.layers.dense(inputs=self.value_fc, units=1, activation=None,
                                         kernel_initializer=xavier_init)
            self.advantage = tf.layers.dense(inputs=self.value_fc, units=num_actions, activation=None,
                                             kernel_initializer=xavier_init)

            # Output: full q output and best action
            self.outputQ = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
            self.best_action = tf.argmax(self.outputQ, 1)

    def predict_best_action(self, state_image, sess):
        return sess.run(self.best_action, feed_dict={self.state_image: state_image})
