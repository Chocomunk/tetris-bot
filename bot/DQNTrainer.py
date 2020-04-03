import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

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
    'pre_train_steps': 33,
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


class DQNTrainer(object):

    def __init__(self, num_actions, model_path, main_model_name, target_model_name,
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
        self.tau = training_args['tau']

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

        # DQN definitions: using a second target network and the Double Q approach
        opt = Adam()
        self.mainDQN = DQNet(num_actions, training_args['conv_out_dim'], name=main_model_name, trainable=True)
        self.targetDQN = DQNet(num_actions, training_args['conv_out_dim'], name=target_model_name, trainable=False)
        self.trainDQN = self.mainDQN.trainable_model(training_args['output_shapes'][0])
        self.trainDQN.compile(loss='mean_squared_error', optimizer=opt)

        checkpoint = ModelCheckpoint("{0}/checkpoints/chkpt-{epoch:02d}-{val_accuracy:.2f}.hdf5", monitor='loss',
                                     save_best_only=True, mode='max', verbose=1)
        self.training_callbacks = [checkpoint]

    def init(self, load_from_file):
        self.epsilon = self.training_args['start_epsilon']      # Reset epsilon
        self.total_reward_list.clear()
        self.total_steps = 0

        if load_from_file:
            if os.path.exists(path=self.model_path + "/model"):
                print("Loading Saved Model")
                self.mainDQN = load_model(self.model_path+"/model")
            else:
                print("No saved model found, skipping load")

    def generate_samples(self, batch, batch_size):
        return [batch[0], batch[1]], self.target_q(batch, batch_size=batch_size)

    def train_target(self, tau=None):
        tau = tau or self.tau
        main_weights = self.mainDQN.get_weights()
        targ_weights = self.targetDQN.get_weights()
        new_weights = []
        for i in range(len(main_weights)):
            new_weights.append(targ_weights[i] * (1 - tau) + main_weights[i] * tau)
        self.targetDQN.set_weights(new_weights)

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
        fake_batch = self.generate_samples((old_state[np.newaxis, :, :, :], np.array([action]), reward,
                                            new_state[np.newaxis, :, :, :], done, None), batch_size=1)
        error = self.trainDQN.evaluate(fake_batch[0], fake_batch[1], verbose=0)

        self.experience_buffer.add(error, (old_state, action, reward, new_state, done))

    def simulate_step(self, state):
        # Select action and send it to the game
        old_stack = np.stack(state, axis=-1)
        action = self.select_action(old_stack)
        next_state, reward, done = self.env.step(action)
        state.appendleft(next_state)
        new_stack = np.stack(state, axis=-1)

        # Remember experience
        self.add_experience(old_stack, new_stack, action, reward, done)

        return reward

    def step_episode(self):
        # Reset environment and initialize state_queue
        state_queue = deque([np.zeros((22, 10)) for _ in range(3)] + [self.env.reset()], 4)
        done = False
        r_total = 0
        iteration = 0

        while (self.episode_length < 0 or iteration < self.episode_length) and not done:
            iteration += 1
            self.total_steps += 1

            # Play the the game for one action and remember the result
            r = self.simulate_step(state_queue)
            r_total += r
            self.total_reward += r

            # Training
            if self.total_steps > self.pre_train_steps:
                if self.total_steps % self.update_freq == 0:    # Train main model
                    batch = next(iter(self.experience_dataset.take(1)))
                    indices = batch[5]
                    samples = self.generate_samples(batch, batch_size=self.batch_size)
                    hist = self.trainDQN.fit(samples[0], samples[1], batch_size=self.batch_size, epochs=1, verbose=0)
                                            # TODO: Causing index out of bound error?
                                             # callbacks=self.training_callbacks)

                    # Update target network and experience buffer
                    self.train_target()
                    for i in range(self.batch_size):
                        self.experience_buffer.update_errors(indices[i], hist.history['loss'][0])

        # Average total reward over 10 steps
        if self.reward_entry_count >= 10:
            self.total_reward_list.append(self.total_reward / 10.)
            self.reward_entry_count = 0
            self.total_reward = 0
        self.reward_entry_count += 1

        self.recent_reward_list.add(r_total)

    def save_model(self):
        if not os.path.exists(path=self.model_path + "/model"):
            os.mkdir(self.model_path + "/model")
        self.mainDQN.save("{0}/model".format(self.model_path))
        print("Saved Model")

