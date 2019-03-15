import tensorflow as tf
from game.tetris import Tetris
from game.tetrisGUI import TetrisGUI
from bot.tetris_bot import TetrisBot

if __name__ == '__main__':
    human = False

    t = Tetris()
    w = TetrisGUI(t, time_scalar=3, bind_controls=True, replay_path="./logs/replay", log_events=human)
    if not human:
        b = TetrisBot(tetris_instance=t, model_path="./bot/ddqn", name="TetrisNet")
    else:
        b = None

    if human:
        w.start()
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            b.init(sess)
            w.start(b, sess)
