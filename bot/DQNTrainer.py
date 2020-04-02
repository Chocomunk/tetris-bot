import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam

from bot.DQNModel import DQNet
from collections import deque
from bot.tetris_environment import TetrisEnvironment
from util.prioritized_experience_replay import PrioritizedExperienceReplay
from util.circular_array import CircularArray


default_training_args = {
    'batch_size': 32,
    'update_frequency': 8,
    'gamma': .99,
    'start_epsilon': 1,
    'end_epsilon': .2,
    'annealing_steps': 10000,
    'pre_train_steps': 500,
    'max_episode_length': -1,
    'conv_out_dim': 64,
    'tau': .01,
    'output_types': (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32),
    'output_shapes': (
        tf.TensorShape([22, 10, 4]),
        tf.TensorShape([]),
        tf.TensorShape([]),
        tf.TensorShape([22, 10, 4]),
        tf.TensorShape([]),
        tf.TensorShape([])
    )
}


def DELETE_get_target_update_ops(main_name, target_name, tau=0.001):
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

        # Training logs
        self.total_reward_list = deque()
        self.total_reward = 0
        self.reward_entry_count = 0
        self.recent_reward_list = CircularArray(100, dtype=np.uint16)

        self.total_steps = 0

        # Dataset setup
        self.batch_size = training_args['batch_size']
        self.experience_buffer = PrioritizedExperienceReplay() if buffer is None else buffer
        self.experience_generator = self.experience_buffer.get_sample_generator(batch_size=self.batch_size)
        self.experience_dataset = tf.data.Dataset.from_generator(
            self.experience_generator,
            output_types=training_args['output_types'],
            output_shapes=training_args['output_shapes']
        ).batch(batch_size=self.batch_size)
        self.batched_iter = self.experience_dataset.make_initializable_iterator()
        self.next_batch = self.batched_iter.get_next()

        # DQN definitions: using a second target network and the Double Q approach
        opt = Adam()
        self.mainDQN = DQNet(num_actions, training_args['conv_out_dim'], name=main_model_name)
        self.targetDQN = DQNet(num_actions, training_args['conv_out_dim'], name=target_model_name)
        self.mainDQN.compile(loss=self.mainDQN.loss, optimizer=opt)
        self.targetDQN.compile(loss=self.targetDQN.loss, optimizer=opt)

    def init(self, load_model):
        self.epsilon = self.training_args['start_epsilon']      # Reset epsilon
        self.total_reward_list.clear()
        self.total_steps = 0

        if not os.path.exists(path=self.model_path):
            os.makedirs(self.model_path)

        if load_model:
            print("Loading Saved Model")
            checkpt = tf.train.get_checkpoint_state(self.model_path)
            if checkpt is None:
                print("No saved model found, skipping load")
                return
            self.saver.restore(sess, checkpt.model_checkpoint_path)

    def generate_samples(self, batch):
        return batch[0], (self.target_q(batch, batch_size=self.batch_size), batch[1])

    def target_q(self, batch, batch_size):
        _, _, b_r, b_s1, b_d, _ = batch

        best_actions = self.mainDQN.predict_action(b_s1, batch_size=batch_size)
        target_qs = self.targetDQN.predict(b_s1, batch_size=batch_size)

        end_multiplier = 1 - b_d
        double_q = target_qs[range(batch_size), best_actions]
        target_q = b_r + (self.gamma * double_q * end_multiplier)

        return target_q

    def select_action(self, inputs):
        if np.random.rand(1) < self.epsilon or self.total_steps < self.pre_train_steps:
            return np.random.randint(0, self.num_actions)
        else:
            if self.epsilon > self.end_epsilon:     # Annealing step
                self.epsilon -= self.epsilon_step_drop
            return self.mainDQN.predict_action(inputs)

    def add_experience(self, old_state, new_state, action, reward, done):
        fake_batch = ([old_state], None, reward, [new_state], done, None)
        targetq = self.target_q(fake_batch, 1)
        error = sess.run(self.abs_error, feed_dict={self.mainDQN.state_image: [old_state],
                                                    self.targetQ: targetq,
                                                    self.actions: [action]})

        self.experience_buffer.add(error, (old_state, action, reward, new_state, done))

    def step_episode(self):
        # Reset environment and initialize state_queue
        state_queue = deque([np.zeros((22, 10)) for _ in range(3)] + [self.env.reset()], 4)
        done = False
        r_total = 0
        iteration = 0

        while (self.episode_length < 0 or iteration < self.episode_length) and not done:
            iteration += 1
            self.total_steps += 1

            # Select action and send it to the game
            old_stack = np.stack(state_queue, axis=-1)
            action = self.select_action(old_stack)
            s1, r, done = self.env.step(action)
            state_queue.appendleft(s1)
            new_stack = np.stack(state_queue, axis=-1)

            # Remember experience
            self.add_experience(old_stack, new_stack, action, r, done)

            r_total += r
            self.total_reward += r

            if self.total_steps > self.pre_train_steps:
                # if self.total_steps % self.update_freq == 0:
                if True:
                    batch = sess.run(self.next_batch)
                    indices = batch[5]
                    target_qs = self._get_target_q(sess, batch, self.batch_size)
                    td_error, _, ac_1 = sess.run((self.abs_error, self.train_step, self.actions_onehot),
                                           feed_dict={self.mainDQN.state_image: batch[0],
                                                      self.targetQ: target_qs,
                                                      self.actions: batch[1]})
                    for op in self.update_target_model_ops:
                        sess.run(op)
                    for i in range(self.batch_size):
                        self.experience_buffer.update_errors(indices[i], td_error[i])
                        # self.experience_buffer.update_errors(batch[5][i], td_error[i])

        if self.reward_entry_count >= 10:
            self.total_reward_list.append(self.total_reward / 10.)
            self.reward_entry_count = 0
            self.total_reward = 0
        self.reward_entry_count += 1

        self.recent_reward_list.add(r_total)

    def save_model(self, sess, label):
        self.saver.save(sess, self.model_path+'/model-'+str(label)+'.ckpt')
        print("Saved Model")

