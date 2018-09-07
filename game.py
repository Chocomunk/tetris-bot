from game.tetris import Tetris
from game.tetrisGUI import TetrisGUI

human = True

t = Tetris()
w = TetrisGUI(t, time_scalar=1, bind_controls=human)
