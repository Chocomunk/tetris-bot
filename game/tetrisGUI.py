import tkinter as tk
import numpy as np
from util.prioritized_experience_replay import PrioritizedExperienceReplay
from collections import deque


class TetrisGUI(object):

    def __init__(self, tetris_game, block_dim=28, fps=10, time_scalar=1, bind_controls=True, replay_path=None, log_events=False):
        """ Initializes a tetris GUI wrapper"""
        self.tetris_game = tetris_game

        self.block_dim = block_dim
        self.fps = fps
        self.tick_ms = int(1000 / (fps * time_scalar))
        self.dt = 1000. / fps

        self.root = tk.Tk()
        self.root.title = "Tetris"
        self.root.resizable(0, 0)
        self.canvas = tk.Canvas(self.root,
                                width=block_dim * (tetris_game.width + 6) + 1,
                                height=block_dim * tetris_game.height, bd=5,
                                highlightthickness=0, bg='#192317')

        self.log_events = log_events
        if log_events:
            self.replay_path = replay_path
            try:
                self.buffer = PrioritizedExperienceReplay.from_file(replay_path)
            except FileNotFoundError:
                self.buffer = PrioritizedExperienceReplay()
            self.state_queue = deque([np.zeros((22, 10)) for _ in range(3)] + [tetris_game.serve_image()], 4)
            self.action = -1

        self.bot = None

        if bind_controls:
            self.bind_canvas(tetris_game)
        self.canvas.bind("<Escape>", self.quit_game)
        self.canvas.focus_set()

        self.game_over = False

    def start(self, bot=None):
        if bot is not None:
            self.bot = bot
            self.bot.env.dt = self.dt
        self.update()
        self.root.mainloop()

    def update(self):
        """ Updates the game by {fps} times a second"""
        self.canvas.delete('all')

        pre_points = self.tetris_game.points
        if self.bot is None:
            s1, d = self.tetris_game.update(self.dt)
        else:
            s1, d = self.bot.update()

        if self.log_events and s1 is not None and not self.game_over:
            old_stack = np.stack(self.state_queue, axis=-1)
            self.state_queue.appendleft(s1)
            new_stack = np.stack(self.state_queue, axis=-1)

            step_reward = (self.tetris_game.points - pre_points)
            self.buffer.add(np.abs(step_reward), (old_stack, self.action, step_reward, new_stack, d))

        if d:
            self.game_over = True

        self.draw_board(self.tetris_game)
        self.draw_auxillary_elements(self.tetris_game)

        self.canvas.pack()
        self.root.after(self.tick_ms, self.update)

    def draw_board(self, tetris_instance):
        """Draws the board and piece by adding rectangles to canvas"""

        width = tetris_instance.width
        height = tetris_instance.height
        board = tetris_instance.board
        colors = tetris_instance.colors
        this_piece = tetris_instance.piece

        # Draw board blocks
        for i in range(width):
            for j in range(height):
                if board[j, i]:
                    self.draw_rectangle(i, j, fill=colors[j, i])

        # Draw dividing line
        self.canvas.create_line(10 + self.block_dim * width, 0,
                                10 + self.block_dim * width,
                                10 + self.block_dim * height,
                                fill="pink")

        # Draw piece
        self.draw_piece(this_piece, True)

    def draw_auxillary_elements(self, tetris_instance):
        """ Draws auxillary game elements on the canvas"""

        width = tetris_instance.width
        height = tetris_instance.height
        this_piece_future_position = tetris_instance.get_future_position()
        next_piece = tetris_instance.next_piece.clone(x_pos=width + 1, y_pos=height / 2)

        # Draw future piece
        self.draw_piece(this_piece_future_position, False, outline='yellow')

        # Draw next piece
        self.draw_piece(next_piece, True)

        # Draw Labels
        self.canvas.create_text(40, 15, text="Points: {}".format(
                                tetris_instance.points),
                                fill='#ffff99')
        self.canvas.create_text(self.block_dim * (width + 2) + 10,
                                self.block_dim * (height / 2 - 2) + 10,
                                text="Next Piece:", fill="pink")

        if tetris_instance.game_over:
            self.canvas.create_text(40, 30, text="GAME OVER", fill="#ffffff")

    def draw_piece(self, piece, do_fill, outline='black'):
        """ Draws a given piece"""
        stone = piece.get_piece()
        for i in range(len(stone)):
            for j in range(len(stone[0]) - 1, -1, -1):
                pos = stone[i][j][0]
                data = stone[i][j][1]

                if data[0] and pos[1] >= 0:
                    color = ''
                    if do_fill:
                        color = data[1]
                    self.draw_rectangle(pos[0], pos[1], fill=color,
                                        outline=outline)

    def draw_rectangle(self, x, y, fill='', outline='black'):
        """Draws a rectangle on the canvas"""
        self.canvas.create_rectangle(5 + x * self.block_dim, y * self.block_dim,
                                     5 + (x + 1) * self.block_dim,
                                     (y + 1) * self.block_dim, fill=fill, outline=outline)

    def quit_game(self, event=None):
        if self.log_events:
            self.buffer.save_file(self.replay_path)
        self.root.destroy()

    def _register_action(self, action):
        self.action = action

    def bind_canvas(self, tetris_instance):
        """Handles binding of canvas events"""
        if self.log_events:
            left_func = lambda _: (tetris_instance.move_left(), self._register_action(0))
            right_func = lambda _: (tetris_instance.move_right(), self._register_action(1))
            down_func = lambda _: (tetris_instance.move_down(), self._register_action(2))
            rotate = lambda _: (tetris_instance.rotate(), self._register_action(3))

            self.canvas.bind("<Left>", left_func)
            self.canvas.bind("<Right>", right_func)
            self.canvas.bind("<Down>", down_func)
            self.canvas.bind("<Up>", rotate)
        else:
            self.canvas.bind("<Left>", tetris_instance.move_left)
            self.canvas.bind("<Right>", tetris_instance.move_right)
            self.canvas.bind("<Down>", tetris_instance.move_down)
            self.canvas.bind("<Up>", tetris_instance.rotate)
