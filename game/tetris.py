import tkinter as tk
import random as r
import numpy as np
import sys
import copy
from bot import Bot
import traceback

shapes = [
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 2, 2],
     [2, 2, 0]],

    [[3, 3, 0],
     [0, 3, 3]],

    [[4, 0, 0],
     [4, 4, 4]],

    [[0, 0, 5],
     [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
     [7, 7]]
]

colors = [
    '#000000',
    '#FF5555',
    '#64C873',
    '#786CF5',
    '#FF953B',
    '#327834',
    '#92CA49',
    '#96A1DA',
    '#232323'
]

HEIGHTS = 0; HOLES = 1; BLOCKADES = 2;

class Stone(object):
    def __init__(self, block_type, x_pos=0, y_pos=0, rotation=0):
        ''' Initializes a tetris stone'''
        self.block = shapes[block_type]
        self.type = block_type
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.rotation = 0

        for i in range(rotation):
            self.rotate()

    def rotate(self):
        ''' Rotates the block clockwise'''
        self.block = [[self.block[y][x] for y in range(len(self.block) - 1, -1, -1)]
                      for x in range(len(self.block[0]))]
        self.rotation = (self.rotation + 1) % 4

    def move(self, x_dist, y_dist):
        ''' Moves the piece by x_dist and y_dist'''
        self.x_pos += x_dist
        self.y_pos += y_dist

    def get_piece(self):
        ''' Returns the piece in a form readable to the board'''
        piece = []
        for x in range(len(self.block[0])):
            column = []
            for y in range(len(self.block)):
                hasBlock = self.block[y][x] > 0
                pos = (x + self.x_pos, y + self.y_pos)
                if (hasBlock):
                    column.append([pos, [True, colors[self.type + 1], False]])   # pos, (exists, color, blockade)
                else:
                    column.append([pos, [False]])
            piece.append(column)

        return piece

    def clone(self, block_type='None', x_pos='None', y_pos='None', rotation='None'):
        ''' Returns a copy of this stone'''
        if block_type is 'None':
            block_type = self.type
        if x_pos is 'None':
            x_pos = self.x_pos
        if y_pos is 'None':
            y_pos = self.y_pos
        if rotation is 'None':
            rotation = self.rotation
        return Stone(block_type, x_pos=x_pos, y_pos=y_pos, rotation=rotation)

    def width(self):
        ''' Returns the width of the piece'''
        return len(self.block[0])


class Tetris(object):
    def __init__(self, dim=28, fps=120, time_scalar=1, level=0, is_human=True, time_limit=-1):
        ''' Initializes a tetris game object'''
        self.dim = dim
        self.fps = fps
        self.level = level
        self.tick_ms = int(1000 / (fps * time_scalar))
        self.width = 10
        self.height = 22

        # Board dimensions: (x pos, y pos, (state of position, color of block))
        self.board = [[[False,] for n in range(self.height)]
                      for m in range(self.width)]
        self.piece_list = [];

        self.piece = None
        self.next_piece = None
        self.new_piece()
        self.points = 0

        self.root = tk.Tk()
        self.root.title = "Tetris"
        self.root.resizable(0, 0)

        self.canvas = tk.Canvas(self.root, width=dim * (self.width + 6) + 1,
                                height=dim * self.height, bd=5, highlightthickness=0, bg='#192317')

        self.bot = None
        self.serve_data = None
        self.bot_data = [0, 0, 0]
        if is_human:
            self.bind_canvas()
        else:
            self.bot = Bot(self)
            self.canvas.bind("<Escape>", self.quit_game)
            self.canvas.focus_set()
            self.serve_data = self.get_data()

        self.time_limit = time_limit
        self.time_elapsed = 0
        self.frames_elapsed = 0
        self.game_over = False

        self.update()
        self.root.mainloop()

    def update(self):
        '''Updates the game by {fps} times a second'''
        self.canvas.delete('all')

        if self.time_elapsed >= self.time_limit > -1:
            self.game_over = True

        if self.frames_elapsed >= self.fps / 2 and not self.game_over:
            self.frames_elapsed = 0

            if not self.check_move(0, 1):
                self.game_over, self.board = self.apply_piece()
                if not self.game_over:
                    self.new_piece()
                if self.bot:
                    self.serve_data = self.get_data()
            else:
                self.move_down()

            self.check_rows()

        if self.bot:
            self.bot.update(self.serve_data)

        self.draw_board()

        self.canvas.pack()
        self.frames_elapsed += 1
        if self.time_limit > -1: self.time_elapsed += 1.0 / self.fps
        self.root.after(self.tick_ms, self.update)

    def draw_board(self):
        '''Draws the board and piece by adding rectangles to canvas'''
        # Draw board blocks
        for i in range(self.width):
            for j in range(self.height):
                if self.board[i][j][0]:
                    self.draw_rectangle(i, j, fill=self.board[i][j][1])

        # Draw dividing line
        self.canvas.create_line(10 + self.dim * (self.width), 0,
                                10 + self.dim * (self.width), 10 + self.dim * (self.height),
                                fill="pink")

        # Draw piece
        self.draw_piece(self.piece, True)

        # Draw future piece
        self.draw_piece(self.get_future_position(), False)

        # Draw next piece
        n_piece = self.next_piece.clone(x_pos=self.width + 1, y_pos=self.height / 2)
        self.draw_piece(n_piece, True, outline='black')

        # Draw Labels
        self.canvas.create_text(40, 15, text="Points: {}".format(self.points),
                                fill='#ffff99')
        self.canvas.create_text(self.dim * (self.width + 2) + 10, self.dim * (self.height / 2 - 2) + 10,
                                text="Next Piece:", fill="pink")
        if self.game_over:
            self.canvas.create_text(40, 30, text="GAME OVER", fill="#ffffff")

    def draw_piece(self, piece, do_fill, outline='yellow'):
        ''' Draws a given piece'''
        stone = piece.get_piece()
        for i in range(len(stone)):
            for j in range(len(stone[0]) - 1, -1, -1):
                pos = stone[i][j][0]
                data = stone[i][j][1]

                if data[0] and pos[1] >= 0:
                    color = ''
                    if do_fill:
                        color = data[1]
                    self.draw_rectangle(pos[0], pos[1], fill=color, outline=outline)

    def draw_rectangle(self, x, y, fill='', outline='black'):
        '''Draws a rectangle on the canvas'''
        self.canvas.create_rectangle(5 + x * self.dim, y * self.dim,
                                     5 + (x + 1) * self.dim, (y + 1) * self.dim, fill=fill, outline=outline)

    def bind_canvas(self):
        '''Handles binding of canvas events'''
        self.canvas.bind("<Left>", self.move_left)
        self.canvas.bind("<Right>", self.move_right)
        self.canvas.bind("<Down>", self.move_down)
        self.canvas.bind("<Up>", self.rotate)
        self.canvas.bind("<Escape>", self.quit_game)
        self.canvas.focus_set()

    def quit_game(self, event=None):
        # print(self.get_piece_data(self.get_future_position(), self.board))
        # print(
        #     [self.get_column_data(self.apply_piece(piece=self.get_future_position(),o_board=self.board)[1], column=i)
        #      for i in range(self.width)])
        sys.exit()

    def move_left(self, event=None):
        # self.piece.move(-1, 0)
        self.update_piece(-1, 0)

    def move_right(self, event=None):
        # self.piece.move(1, 0)
        self.update_piece(1, 0)

    def move_down(self, event=None):
        # self.piece.move(0, 1)
        self.update_piece(0, 1)

    def rotate(self, event=None):
        if self.check_rotate():
            self.piece.rotate()

    def update_piece(self, x, y):
        '''Update piece on board'''
        if self.check_move(x, y):
            self.piece.move(x, y)

    def check_move(self, x, y, piece=None):
        ''' Determines whether a piece is able to move'''
        canMove = True

        if not piece:
            piece = self.piece
        stone = piece.get_piece()
        for i in range(len(stone)):
            for j in range(len(stone[0])):
                pos = stone[i][j][0]
                data = stone[i][j][1]

                if not pos[1] + y < 0:  # ignore this block
                    if ((pos[0] + x < 0) or
                            (pos[0] + x > self.width - 1) or
                        # (pos[1] + y < 0) or
                            (pos[1] + y > self.height - 1)):
                        canMove = False
                    elif data[0]:  # Block exists
                        canMove = (canMove and not
                        self.board[pos[0] + x][pos[1] + y][0])

        return canMove

    def check_rotate(self):
        ''' Determines whether a piece is able to rotate'''
        rot_piece = self.piece.clone()
        rot_piece.rotate()

        return self.check_move(0, 0, rot_piece)

    def new_piece(self):
        ''' Update with new pieces'''
        if not self.next_piece:
            self.next_piece = self.get_piece()

        self.piece = self.next_piece
        valid_piece = False
        while not valid_piece:
            valid_piece = self.check_move(0, 0)
            if not valid_piece:
                self.piece.move(0, -1)

        self.next_piece = self.get_piece()

    def get_piece(self):
        ''' Create a new piece'''
        block_type = int(r.random() * len(shapes))
        # block_type = 5
        x_pos = int(r.random() * (self.width - len(shapes[block_type][0])))
        piece = Stone(block_type, x_pos=x_pos, y_pos=0)

        return piece

    def get_future_position(self, piece=None):
        ''' Calculate position if piece moves straight down'''
        if not piece:
            piece = self.piece

        fut_piece = piece.clone()
        valid_pos = True
        # print("init_pos: {}, {}, {}".format(fut_piece.y_pos, piece.rotation, fut_piece.rotation))
        while valid_pos:
            valid_pos = self.check_move(0, 1, piece=fut_piece)
            if valid_pos:
                fut_piece.move(0, 1)
                # if len(fut_piece.block) + fut_piece.y_pos > self.height:
                # 	print("{}, {}, {}".format(len(fut_piece.block), fut_piece.y_pos, fut_piece.rotation))
                # 	print(fut_piece.block)
        return fut_piece

    def apply_piece(self, piece=None, o_board=None):
        ''' Places all blocks of the piece onto the board'''
        if not piece:
            piece = self.piece
        if not o_board:
            o_board = self.board
        board = copy.deepcopy(o_board)

        isGameOver = False

        stone = piece.get_piece()
        for i in range(len(stone)):
            for j in range(len(stone[0])):
                pos = stone[i][j][0]
                data = stone[i][j][1]

                if data[0]:
                    if pos[1] <= 0:
                        isGameOver = True
                    else:
                        try:
                            board[pos[0]][pos[1]] = data
                        except IndexError:
                            traceback.print_exc()
                            # print("{}, {}".format(pos[0],pos[1]))
        return (isGameOver, board)

    def check_rows(self):
        ''' Checks each row for completion, and calculates points'''
        rows_done = []
        j = self.height - 1
        while j >= 0:
            full_row = True  # checks the row for fullness
            for i in range(self.width):
                full_row = full_row and self.board[i][j][0]
            if full_row:
                rows_done.append(j)
            j -= 1

        if len(rows_done) > 0:
            self.update_info_clear(rows_done)

            for j in rows_done:
                for i in range(self.width):
                    del self.board[i][j]
                    self.board[i] = [(False,)] + self.board[i]

        base_points = (0, 40, 100, 300, 1200)
        self.points += base_points[len(rows_done)] * (self.level + 1)

    def isGameOver(self):
        ''' Returns whether the game has ended'''
        return self.game_over

    def update_info_clear(self, rows, board=None, b_data=None):
        if not board:
            board = copy.deepcopy(self.board)
        if not b_data:
            b_data = copy.copy(self.bot_data)

        rows = sorted(rows)

        for i in range(rows[0], self.height):
            for j in range(self.width):
                if i in rows:
                    b_data[HEIGHTS] -= self.height - i
                    if board[j][i][2]:
                        b_data[BLOCKADES] -= 1
                else:
                    b_data[HEIGHTS] -= len(rows)

        return b_data

    def get_piece_data(self, bot_data, board, old_board, piece):
        """
        Heights, Blockade, Holes, Clear, Piece Data
        """
        wall_count = 0
        floor_count = 0
        block_count = 0
        clear_count = 0

        b_data = copy.copy(bot_data)

        stone = piece.get_piece()
        for i in range(len(stone[0])):
            is_full_row = True
            for j in range(self.width):
                if not board[j][i + piece.y_pos][0]:
                    is_full_row = False
                if piece.x_pos <= j < piece.x_pos + piece.width():
                    pos = stone[j-piece.x_pos][i][0]
                    data = stone[j-piece.x_pos][i][1]

                    if data[0]:
                        b_data[HEIGHTS] += self.height - pos[1]   # Add heights

                        if not old_board[pos[0]][pos[1]][0]:
                            b_data[HOLES] -= 1

                        if pos[1] < self.height-1 and ((board[pos[0]][pos[1]+1][0] and board[pos[0]][pos[1]+1][2])
                                                       or not board[pos[0]][pos[1]+1][0]):
                            b_data[BLOCKADES] += 1
                            board[pos[0]][pos[1]][2] = True

                        y = pos[1]+1
                        while y < self.height and not board[pos[0]][y][0]:
                            b_data[HOLES] += 1
                            y += 1

                        if pos[0] == 0 or pos[0] == self.width - 1:
                            wall_count += 1
                        if pos[1] == self.height - 1:
                            floor_count += 1

                        if pos[0] < self.width - 1 and old_board[pos[0] + 1][pos[1]][0]:
                            block_count += 1
                        if pos[0] > 0 and old_board[pos[0] - 1][pos[1]][0]:
                            block_count += 1
                        if pos[1] < self.height - 1 and old_board[pos[0]][pos[1] + 1][0]:
                            block_count += 1
                        if pos[1] > 0 and old_board[pos[0]][pos[1] - 1][0]:
                            block_count += 1
            if is_full_row:
                clear_count += 1
        return np.array(b_data), np.array((clear_count, block_count, wall_count, floor_count))

    def get_data(self, piece=None, next_piece=None, o_board=None, b_data=None):
        ''' Returns statistics of the current game state for all possibilities'''
        if not piece:
            piece = self.piece
        if not next_piece:
            next_piece = self.next_piece
        if not o_board:
            o_board = self.board
        if not b_data:
            b_data = self.bot_data
        data1 = b_data
        board1 = copy.deepcopy(o_board)

        data = []
        for r1 in range(4):  # Rotations for this piece
            this_piece = piece.clone(rotation=r1, x_pos=0, y_pos=0)

            data_x1 = []
            for x1 in range(self.width - len(this_piece.block[0]) + 1):  # Positions for this piece
                this_piece.x_pos = x1
                t_piece = self.get_future_position(this_piece)
                _, board2 = self.apply_piece(piece=t_piece, o_board=board1)

                data_p1 = self.get_piece_data(data1, board2, board1, t_piece)

                data_r2 = []
                for r2 in range(4):  # Rotations for next piece
                    this_next_piece = next_piece.clone(rotation=r2, x_pos=0, y_pos=0)

                    data_x2 = []
                    for x2 in range(self.width - len(this_next_piece.block[0]) + 1):  # Positions for next piece
                        this_next_piece.x_pos = x2
                        n_piece = self.get_future_position(this_next_piece)
                        _, board = self.apply_piece(piece=n_piece, o_board=board2)

                        data_p2 = self.get_piece_data(data_p1[0], board, board2, n_piece)

                        data_x2.append(np.hstack((data_p1[0] + data_p2[0], data_p1[1] + data_p2[1])))

                    data_r2.append(data_x2)
                data_x1.append(data_r2)
            data.append(data_x1)

        return data

    # def get_column_data(self, board, column):
    #     """ Returns the following data:
    #          - Sum of block heights in the board
    #          - Number of clears
    #          - Number of holes
    #          - Number of blockades"""
    #     hole_count = 0
    #     blockade_count = 0
    #     height_sum = 0
    #     is_full_row = 1
    #
    #     found_block = False
    #     blockage_group_count = 0
    #     for j in range(self.height):
    #         if board[column][j][0]:
    #             found_block = True
    #             height_sum += self.height - j
    #             blockage_group_count += 1
    #         else:
    #             is_full_row = 0
    #             if found_block:
    #                 blockade_count += blockage_group_count
    #                 blockage_group_count = 0
    #                 hole_count += 1
    #     return np.array([height_sum, hole_count, blockade_count, is_full_row])
    #
    # def get_piece_data(self, piece, board):
    #     """ Returns the following data:
    #          - Number of edges touching the wall
    #          - Number of edges touching the floor
    #          - Number of edges touching other blocks"""
    #     wall_count = 0
    #     floor_count = 0
    #     block_count = 0
    #     stone = piece.get_piece()
    #     for i in range(len(stone)):
    #         for j in range(len(stone[0])):
    #             pos = stone[i][j][0]
    #             data = stone[i][j][1]
    #
    #             if data[0]:
    #                 if pos[0] == 0 or pos[0] == self.width - 1:
    #                     wall_count += 1
    #                 if pos[1] == self.height - 1:
    #                     floor_count += 1
    #                 try:
    #                     if pos[0] < self.width - 1 and board[pos[0] + 1][pos[1]][0]:
    #                         block_count += 1
    #                     if pos[0] > 0 and board[pos[0] - 1][pos[1]][0]:
    #                         block_count += 1
    #                     if pos[1] < self.height - 1 and board[pos[0]][pos[1] + 1][0]:
    #                         block_count += 1
    #                     if pos[1] > 0 and board[pos[0]][pos[1] - 1][0]:
    #                         block_count += 1
    #                 except IndexError as e:
    #                     traceback.print_exc()
    #                     print(
    #                         "{}, {}, {}, {}, {}".format(pos[0], pos[1], piece.y_pos, len(piece.block), piece.rotation))
    #                     # print(piece.block)
    #
    #     return np.array([block_count, wall_count, floor_count])
    #
    # def get_data(self, piece=None, next_piece=None, o_board=None):
    #     ''' Returns statistics of the current game state for all possibilities'''
    #     if not piece:
    #         piece = self.piece
    #     if not next_piece:
    #         next_piece = self.next_piece
    #     if not o_board:
    #         o_board = self.board
    #     board1 = copy.deepcopy(o_board)
    #
    #     board_data = [self.get_column_data(board=board1,column=i) for i in range(self.width)]
    #
    #     data = []
    #     for r1 in range(4):  # Rotations for this piece
    #         this_piece = piece.clone(rotation=r1, x_pos=0, y_pos=0)
    #
    #         data_x1 = []
    #         for x1 in range(self.width - len(this_piece.block[0]) + 1):  # Positions for this piece
    #             this_piece.x_pos = x1
    #             t_piece = self.get_future_position(this_piece)
    #             piece_data_1 = self.get_piece_data(t_piece, board1)
    #             _, board2 = self.apply_piece(piece=t_piece, o_board=board1)
    #
    #             for i in range(x1, t_piece.width()):
    #                 board_data[i] = self.get_column_data(board=board2, column=i)
    #
    #             data_r2 = []
    #             for r2 in range(4):  # Rotations for next piece
    #                 this_next_piece = next_piece.clone(rotation=r2, x_pos=0, y_pos=0)
    #
    #                 data_x2 = []
    #                 for x2 in range(self.width - len(this_next_piece.block[0]) + 1):  # Positions for next piece
    #                     this_next_piece.x_pos = x2
    #                     n_piece = self.get_future_position(this_next_piece)
    #                     piece_data_2 = self.get_piece_data(n_piece, board2)
    #                     _, board = self.apply_piece(piece=n_piece, o_board=board2)
    #
    #                     for i in range(x2, n_piece.width()):
    #                         board_data[i] = self.get_column_data(board=board, column=i)
    #
    #                     data_x2.append(np.hstack((sum(board_data), (piece_data_1 + piece_data_2))))
    #
    #                 data_r2.append(data_x2)
    #             data_x1.append(data_r2)
    #         data.append(data_x1)
    #
    #     return data
