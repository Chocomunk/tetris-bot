import tensorflow as tf
import argparse
from game.tetris import Tetris
from game.tetrisGUI import TetrisGUI
from bot.tetris_bot import TetrisBot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs an instance of a tetris game GUI")
    parser.add_argument('--replay-path', '-r', action="store", dest="replay_path", type=str,
                        help="Path from which to save and load experience replays")
    parser.add_argument('--model-path', '-m', action="store", dest="model_path", type=str,
                        help="Path from which to load the model")
    parser.add_argument('--log-events', '-l', action="store_true", dest="log_events", default=False,
                        help="Enables logging events in an experience replay")
    parser.add_argument('--bot', '-b', action="store_true", dest="is_bot", default=False,
                        help="Enables play by AI")
    args = parser.parse_args()

    if args.log_events and not args.replay_path:
        raise ValueError("Trying to log events but no reply path is provided!")
    if args.is_bot and not args.model_path:
        raise ValueError("Trying to play with AI, but no model path is provided!")

    t = Tetris()
    w = TetrisGUI(t, time_scalar=3, bind_controls=True, replay_path=args.replay_path, log_events=args.log_events)
    if args.is_bot:
        b = TetrisBot(tetris_instance=t, model_path=args.model_path, name="TetrisNet")
    else:
        b = None

    if not args.is_bot:
        w.start()
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            b.init(sess)
            w.start(b, sess)
