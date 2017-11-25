from tetris import Tetris
from tetrisGUI import TetrisGUI

human = False

t = Tetris(is_human=human)
w = TetrisGUI(t, time_scalar=1, bind_controls=human)
