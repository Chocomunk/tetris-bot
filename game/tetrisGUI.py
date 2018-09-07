import tkinter as tk
import numpy as np


class TetrisGUI(object):
    def __init__(self, tetris_game, block_dim=28, fps=120, time_scalar=1, bind_controls=True):
        """ Initializes a tetris GUI wrapper"""
        self.tetris_game = tetris_game

        self.block_dim = block_dim
        self.fps = fps
        self.tick_ms = int(1000 / (fps * time_scalar))

        self.root = tk.Tk()
        self.root.title = "Tetris"
        self.root.resizable(0, 0)
        self.canvas = tk.Canvas(self.root,
                                width=block_dim * (tetris_game.width + 6) + 1,
                                height=block_dim * tetris_game.height, bd=5,
                                highlightthickness=0, bg='#192317')

        if bind_controls:
            self.bind_canvas(tetris_game)
        self.canvas.bind("<Escape>", tetris_game.quit_game)
        self.canvas.focus_set()

        self.update()
        self.root.mainloop()

    def update(self):
        """ Updates the game by {fps} times a second"""
        self.canvas.delete('all')

        self.tetris_game.update()

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

    def bind_canvas(self, tetris_instance):
        """Handles binding of canvas events"""
        self.canvas.bind("<Left>", tetris_instance.move_left)
        self.canvas.bind("<Right>", tetris_instance.move_right)
        self.canvas.bind("<Down>", tetris_instance.move_down)
        self.canvas.bind("<Up>", tetris_instance.rotate)
