import os
import numpy as np
import matplotlib.pyplot as plt
from bot.DQNTrainer import DQNTrainer
from bot.tetris_environment import TetrisEnvironment
from util.prioritized_experience_replay import PrioritizedExperienceReplay

import time

NUM_EPISODES = 3

if __name__ == '__main__':
    env = TetrisEnvironment(time_limit=10000)
    buffer = PrioritizedExperienceReplay.from_file('./logs/replay')
    # buffer = PrioritizedExperienceReplay()
    # buffer = ExperienceReplayBuffer()
    trainer = DQNTrainer(5, "bot/ddqn", "TetrisNet", "TetrisTargetDQN", buffer=buffer)

    try:
        start = time.time()
        for i in range(NUM_EPISODES):
            print("Training Episode {}".format(i+1))
            trainer.step_episode()
            trainer.save_model(i + 1)
        print("FINISHED TRAINING")
        print("Training took: {}".format(time.time() - start))
    finally:
        if not os.path.exists(path="plots"):
            os.mkdir("plots")

        plt.plot(np.array(trainer.total_reward_list))
        plt.title("Total Reward Per Episode")
        plt.savefig("plots/reward_plot.png")
        plt.clf()

        plt.plot(np.array(trainer.avg_loss_list))
        plt.title("Average Loss Per Episode")
        plt.savefig("plots/loss_plot.png")
