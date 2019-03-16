import numpy as np
import tensorflow as tf
import os
from bot.DDQNetwork import DDQNetwork
from bot.tetris_environment import TetrisEnvironment
from util.prioritized_experience_replay import PrioritizedExperienceReplay


default_training_args = {
    'batch_size': 32,
    'update_frequency': 8,
    'gamma': .99,
    'start_epsilon': 1,
    'end_epsilon': .1,
    'annealing_steps': 10000,
    'pre_train_steps': 500,
    'max_episode_length': -1,
    'conv_out_dim': 512,
    'tau': .001,
    'output_types': (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32),
    'output_shapes': (
        tf.TensorShape([22, 10, 1]),
        tf.TensorShape([]),
        tf.TensorShape([]),
        tf.TensorShape([22, 10, 1]),
        tf.TensorShape([]),
        tf.TensorShape([])
    )
}


def get_target_update_ops(main_name, target_name, tau=0.001):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, main_name)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_name)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign((from_var.value()*tau) + ((1-tau)*to_var.value())))
    return op_holder


class DQNTrainer(object):

    def __init__(self, image_size, num_actions, model_path, main_model_name, target_model_name,
                 env=None, buffer=None, training_args=None):
        if training_args is None:
            training_args = default_training_args
        self.training_args = training_args
        self.num_actions = num_actions
        self.model_path = model_path
        self.env = TetrisEnvironment() if env is None else env

        # Extract training args
        self.gamma = training_args['gamma']
        self.update_freq = training_args['update_frequency']
        self.episode_length = training_args['max_episode_length']
        self.pre_train_steps = training_args['pre_train_steps']
        self.epsilon = training_args['start_epsilon']
        self.end_epsilon = training_args['end_epsilon']
        self.epsilon_step_drop = float(self.epsilon - self.end_epsilon) / training_args['annealing_steps']

        self.move_count_list = []
        self.reward_list = []
        self.total_steps = 0
        self.total_episodes = 0

        # Dataset setup
        self.batch_size = training_args['batch_size']
        self.experience_buffer = PrioritizedExperienceReplay() if buffer is None else buffer
        self.next_experience = self.experience_buffer.get_sample_generator(self.batch_size)

        # DQN definitions: using a second target network and the Double Q approach
        self.mainDQN = DDQNetwork(training_args['conv_out_dim'], image_size, num_actions, main_model_name)
        self.targetDQN = DDQNetwork(training_args['conv_out_dim'], image_size, num_actions, target_model_name)

        # Slowly update target network to the main network
        self.update_target_model_ops = get_target_update_ops(main_model_name, target_model_name, training_args['tau'])

        self.saver = tf.train.Saver()

        # Training functions
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

        self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)
        self.modelQ = tf.reduce_sum(tf.multiply(self.mainDQN.outputQ, self.actions_onehot), axis=1)

        self.conv_total = tf.reduce_sum(self.mainDQN.conv_out)

        # Loss ops
        self.td_error = self.targetQ - self.modelQ  # Kept in separate variable for training evaluation
        self.abs_error = tf.abs(self.td_error)
        self.cost = tf.reduce_mean(tf.square(self.td_error))

        # Training/Optimization
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.train_step = self.trainer.minimize(self.cost)

    def _get_target_q(self, sess, batch, batch_size):
        b_s, b_a, b_r, b_s1, b_d, b_i = batch

        best_actions, next_qs = sess.run((self.mainDQN.best_action, self.mainDQN.outputQ),
                                         feed_dict={self.mainDQN.state_image: b_s1})
        target_qs = sess.run(self.targetDQN.outputQ, feed_dict={self.targetDQN.state_image: b_s1})

        end_multiplier = 1 - b_d
        double_q = target_qs[range(batch_size), best_actions]
        target_q = b_r + (self.gamma * double_q * end_multiplier)

        return target_q

    def init(self, sess, load_model):
        self.epsilon = self.training_args['start_epsilon']      # Reset epsilon

        self.move_count_list.clear()
        self.reward_list.clear()
        self.total_steps = 0
        self.total_episodes = 0

        if not os.path.exists(path=self.model_path):
            os.makedirs(self.model_path)

        if load_model:
            print("Loading Saved Model")
            checkpt = tf.train.get_checkpoint_state(self.model_path)
            if checkpt is None:
                print("No saved model found, skipping load")
                return
            self.saver.restore(sess, checkpt.model_checkpoint_path)

    def step_episode(self, sess):
        episode_buffer = PrioritizedExperienceReplay()
        s = self.env.reset()
        d = False
        r_total = 0
        iteration = 0
        while (self.episode_length < 0 or iteration < self.episode_length) and not d:
            iteration += 1

            if np.random.rand(1) < self.epsilon or self.total_steps < self.pre_train_steps:
                a = np.random.randint(0, self.num_actions)
                type_a = "RAND_SELECT"
            else:
                a = sess.run(self.mainDQN.best_action, feed_dict={self.mainDQN.state_image: [s]})
                a = a[0]
                type_a = "DQN_VALUE"
            s1, r, d = self.env.step(a)

            fake_batch = ([s], None, r, [s1], d, None)
            t_q = self._get_target_q(sess, fake_batch, 1)
            error = sess.run(self.td_error, feed_dict={self.mainDQN.state_image: [s],
                                                       self.targetQ: t_q,
                                                       self.actions: [a]})

            episode_buffer.add(np.abs(error), (s, a, r, s1, d))
            if self.total_steps % 1000 == 0:
                print(self.total_episodes, self.total_steps, a, r_total, error, type_a)
            self.total_steps += 1

            if self.total_steps > self.pre_train_steps:
                if self.epsilon > self.end_epsilon:
                    self.epsilon -= self.epsilon_step_drop

                if self.total_steps % self.update_freq == 0:
                    batch = self.next_experience()
                    indices = batch[5]
                    target_qs = self._get_target_q(sess, batch, self.batch_size)
                    td_error, _ = sess.run((self.td_error, self.train_step),
                                           feed_dict={self.mainDQN.state_image: batch[0],
                                                      self.targetQ: target_qs,
                                                      self.actions: batch[1]})
                    for op in self.update_target_model_ops:
                        sess.run(op)
                    for i in range(self.batch_size):
                        self.experience_buffer.update_errors(indices[i], td_error[i])
                        # self.experience_buffer.update_errors(batch[5][i], td_error[i])

            r_total += r
            s = s1

        self.total_episodes += 1
        self.experience_buffer.extend(episode_buffer)
        self.move_count_list.append(iteration)
        self.reward_list.append(r_total)

    def save_model(self, sess, label):
        self.saver.save(sess, self.model_path+'/model-'+str(label)+'.ckpt')
        print("Saved Model")

    def print_progress(self):
        print("Episode: {0}, Total Steps: {1}, Average Reward: {2}, Epsilon: {3}".format(
            self.total_episodes, self.total_steps, np.mean(self.reward_list[-20:]), self.epsilon
        ))
