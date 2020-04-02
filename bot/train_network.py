import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from bot.DQNTrainer import DQNTrainer
from bot.tetris_environment import TetrisEnvironment
from util.prioritized_experience_replay import PrioritizedExperienceReplay

NUM_EPISODES = 1

if __name__ == '__main__':
    env = TetrisEnvironment(time_limit=120000)
    buffer = PrioritizedExperienceReplay.from_file('./logs/replay')
    # buffer = ExperienceReplayBuffer()
    trainer = DQNTrainer(4, "./bot/ddqn", "TetrisNet", "TetrisTargetDQN", buffer=buffer)

    try:
        for i in range(NUM_EPISODES):
            trainer.step_episode()
        print("FINISHED TRAINING")
        trainer.save_model()
    finally:
        # rMat = np.resize(np.array(trainer.reward_list), [len(trainer.reward_list) // 1000, 1000])
        rMat = np.array(trainer.total_reward_list)
        # rMean = np.average(rMat, 1)
        # plt.plot(rMean)
        plt.plot(rMat)
        plt.show()
