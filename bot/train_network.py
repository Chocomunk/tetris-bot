import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from bot.DQNTrainer import DQNTrainer
from bot.tetris_environment import TetrisEnvironment
from util.prioritized_experience_replay import PrioritizedExperienceReplay

NUM_EPISODES = 700

if __name__ == '__main__':
    env = TetrisEnvironment(time_limit=120000)
    buffer = PrioritizedExperienceReplay.from_file('./logs/replay')
    # buffer = ExperienceReplayBuffer()
    trainer = DQNTrainer([22, 10], 4, "./bot/ddqn", "TetrisNet", "TetrisTargetDQN", buffer=buffer)

    init = tf.global_variables_initializer()

    try:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            trainer.init(sess, True)

            for i in range(NUM_EPISODES):
                trainer.step_episode(sess)
                if i % 10 == 0:
                    trainer.print_progress()
                    trainer.save_model(sess, i)
            trainer.print_progress()
            trainer.save_model(sess, 'FINAL')
    finally:
        rMat = np.resize(np.array(trainer.reward_list), [len(trainer.reward_list) // 10, 10])
        rMean = np.average(rMat, 1)
        plt.plot(rMean)
        plt.show()
