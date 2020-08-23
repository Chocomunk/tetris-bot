import numpy as np
import tensorflow as tf
import os
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from bot.DQNModel import DQNet
from collections import deque
from bot.tetris_environment import TetrisEnvironment
from util.prioritized_experience_replay import PrioritizedExperienceReplay


default_training_args = {
    'batch_size': 32,
    'update_frequency': 4,
    'gamma': .99,
    'start_epsilon': 1,
    'end_epsilon': .2,
    'annealing_steps': 1000,
    'pre_train_steps': 1000,
    'max_episode_length': 10000,
    'tau': .95,
    'output_types': (tf.float32, tf.int32, tf.float32, tf.float32, tf.int32, tf.int32),
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
        self.avg_loss_list = deque()

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
        self.opt = Adam()
        self.loss = MeanSquaredError()
        self.mainDQN = DQNet(training_args['output_shapes'][0], num_actions, name=main_model_name, trainable=True)
        self.targetDQN = DQNet(training_args['output_shapes'][0], num_actions, name=target_model_name, trainable=False)
        self.trainDQN = self.mainDQN.trainable_model(training_args['output_shapes'][0])
        self.trainDQN.compile(loss=self.loss, optimizer=self.opt)

        checkpoint = ModelCheckpoint("checkpoints/chkpt-{epoch:02d}-{val_accuracy:.2f}.hdf5", monitor='loss',
                                     save_best_only=True, mode='max', verbose=1)
        self.training_callbacks = [checkpoint]

        self.state_input = tf.keras.layers.InputLayer(input_shape=training_args['output_shapes'][0])
        self.q_input = tf.keras.layers.InputLayer(input_shape=[])

        self.sim_play = 0
        self.sim_remember = 0
        self.targ_pred_main = 0
        self.targ_pred_dual = 0
        self.targ_comp_rw = 0
        self.targ_comp_qs = 0
        self.exp_batch = 0
        self.exp_error = 0
        self.exp_add = 0

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

        start = time.time()

        # best_actions = self.mainDQN.predict_action(b_s1, batch_size=batch_size)
        double_q = self.targetDQN.double_q(b_s1, self.mainDQN, batch_size)

        mid1 = time.time()
        # target_qs = self.targetDQN.predict(b_s1, batch_size=batch_size)
        # target_qs = self.targetDQN(b_s1, training=False).numpy()

        mid2 = time.time()

        end_multiplier = 1. - b_d
        # double_q = target_qs[range(batch_size), best_actions]
        mid3 = time.time()
        target_q = b_r + (self.gamma * double_q * end_multiplier)

        self.targ_comp_rw += time.time() - mid3
        self.targ_comp_qs += mid3 - mid2
        self.targ_pred_dual += mid2 - mid1
        self.targ_pred_main += mid1 - start

        return target_q

    def select_action(self, inputs):
        if np.random.rand(1) < self.epsilon or self.total_steps < self.pre_train_steps:
            return np.random.randint(0, self.num_actions)
        else:
            if self.epsilon > self.end_epsilon:     # Annealing step
                self.epsilon -= self.epsilon_step_drop
            return self.mainDQN.predict_action(inputs)

    @tf.function
    def experience_error(self, state, true):
        s_inp = self.state_input(state)
        q_inp = self.q_input(true)
        pred = self.trainDQN(s_inp, training=False)
        return self.loss(q_inp, pred)

    def add_experience(self, old_state, new_state, action, reward, done):
        start = time.time()
        fake_batch = self.generate_samples((old_state[np.newaxis, :, :, :], np.array([action]), reward,
                                            new_state[np.newaxis, :, :, :], done, None), batch_size=1)
        mid1 = time.time()
        error = self.experience_error(fake_batch[0], fake_batch[1])
        mid2 = time.time()

        self.experience_buffer.add(error, (old_state, action, reward, new_state, done))

        self.exp_add += time.time() - mid2
        self.exp_error += mid2 - mid1
        self.exp_batch += mid1 - start

    def simulate_step(self, state):
        # Select action and send it to the game
        start = time.time()

        old_stack = np.stack(state, axis=-1)
        action = self.select_action(old_stack)
        next_state, reward, done, _ = self.env.step(action)
        state.appendleft(next_state)
        new_stack = np.stack(state, axis=-1)

        mid = time.time()

        # Remember experience
        self.add_experience(old_stack, new_stack, action, reward, done)

        self.sim_remember += time.time() - mid
        self.sim_play += mid - start

        return reward

    def step_episode(self):
        # Reset environment and initialize state_queue
        start_time = time.time()
        state_queue = deque([np.zeros((22, 10)) for _ in range(3)] + [self.env.reset()], 4)
        done = False
        r_total = 0
        l_total = 0
        iteration = 0

        train_time = 0
        sim_time = 0
        data_time = 0
        buff_time = 0

        data_batch = 0
        data_samples = 0

        while (self.episode_length < 0 or iteration < self.episode_length) and not done:
            iteration += 1
            self.total_steps += 1

            # Play the the game for one action and remember the result
            s_start = time.time()
            r = self.simulate_step(state_queue)
            sim_time += time.time() - s_start
            r_total += r

            # Training
            loss = 0
            if self.total_steps > self.pre_train_steps:
                if self.total_steps % self.update_freq == 0:    # Train main model
                    d_start = time.time()
                    # batch = next(iter(self.experience_dataset.take(1)))
                    batch = next(self.experience_generator())
                    d_mid = time.time()
                    indices = batch[5]
                    samples = self.generate_samples(batch, batch_size=self.batch_size)
                    data_samples += time.time() - d_mid
                    data_batch += d_mid - d_start
                    data_time += time.time() - d_start
                    t_start = time.time()
                    hist = self.trainDQN.fit(samples[0], samples[1], batch_size=self.batch_size, epochs=1, verbose=0)
                                            # TODO: Causing index out of bound error?
                                             # callbacks=self.training_callbacks)
                    train_time += time.time() - t_start

                    # Update target network and experience buffer
                    loss = hist.history['loss'][0]
                    l_total += loss
                    b_start = time.time()
                    for i in range(self.batch_size):
                        self.experience_buffer.update_errors(indices[i], loss)
                    buff_time += time.time() - b_start
            if iteration % 1000 == 0:
                print("Iteration: {0}/{1} - Total Time: {2:.2f}s - Total Train Time: {3:.2f}s - loss: {4:.4f}".format(
                    iteration, self.episode_length, time.time() - start_time, train_time, loss))
                print("Simulation Time: {0:.2f}s - Data Time: {1:.2f}s - Buffer Time: {2:.2f}s".format(
                    sim_time, data_time, buff_time))
                print("Simulation Play: {0:.2f}s - Simulation Remember: {1:.2f}s".format(self.sim_play, self.sim_remember))
                print("Data Batch: {0:.2f}s - Data Samples: {1:.2f}s".format(data_batch, data_samples))
                print("Target Predict Main: {0:.2f}s - Target Predict Dual: {1:.2f}s - Target Compute Reward: {2:.2f}s - Target Compute Qs: {3:.2f}s".format(
                    self.targ_pred_main, self.targ_pred_dual, self.targ_comp_rw, self.targ_comp_qs))
                print("Experience Batch: {0:.2f}s - Experience Error: {1:.2f}s - Experience Add: {2:.2f}s".format(
                    self.exp_batch, self.exp_error, self.exp_add))
                print("")
                self.train_target()

                train_time = 0
                sim_time = 0; data_time = 0; buff_time = 0
                data_batch = 0; data_samples = 0
                self.sim_play = 0; self.sim_remember = 0
                self.targ_pred_main = 0; self.targ_pred_dual = 0; self.targ_comp_rw = 0; self.targ_comp_qs = 0
                self.exp_batch = 0; self.exp_error = 0; self.exp_add = 0

        # Log statistics
        self.total_reward_list.append(r_total)
        self.avg_loss_list.append(l_total / iteration)

    def save_model(self, episode):
        if not os.path.exists(path=self.model_path):
            os.mkdir(self.model_path)
        self.mainDQN.save_weights("{0}/cp-{1:04d}.ckpt".format(self.model_path, episode))
        print("Saved Model")

