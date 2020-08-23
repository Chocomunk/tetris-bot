import random as r
import numpy as np
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


class Stone(object):
    def __init__(self, block_type, x_pos=0, y_pos=0, rotation=0):
        """ Initializes a tetris stone"""
        self.block = shapes[block_type]
        self.type = block_type
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.rotation = 0

        for i in range(rotation):
            self.rotate()

    def rotate(self):
        """ Rotates the block clockwise"""
        self.block = [
            [self.block[y][x] for y in range(len(self.block) - 1, -1, -1)]
            for x in range(len(self.block[0]))]
        self.rotation = (self.rotation + 1) % 4

    def move(self, x_dist, y_dist):
        """ Moves the piece by x_dist and y_dist"""
        self.x_pos += x_dist
        self.y_pos += y_dist

    def get_piece(self):
        """ Returns the piece in a form readable to the board"""
        piece = []
        for x in range(len(self.block[0])):
            column = []
            for y in range(len(self.block)):
                has_block = self.block[y][x] > 0
                pos = (x + self.x_pos, y + self.y_pos)
                if has_block:
                    # pos, (exists, color)
                    column.append([pos, [True, colors[self.type + 1]]])
                else:
                    column.append([pos, [False]])
            piece.append(column)

        return piece

    def clone(self, block_type='None', x_pos='None', y_pos='None',
              rotation='None'):
        """ Returns a copy of this stone"""
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
        """ Returns the width of the piece"""
        return len(self.block[0])


class Tetris(object):

    def __init__(self, time_limit=-1):
        """ Initializes a tetris game object"""
        self.width = 10
        self.height = 22

        # Bord elements are 0 for empty and 1 for filled
        self.board = np.zeros((self.height, self.width), dtype=np.float32)
        self.colors = np.zeros((self.height, self.width), dtype=np.dtype("a7"))     # dtype: 7 character string
        self.piece_list = []

        self.piece = None
        self.next_piece = None
        self.piece_moved = False
        self.new_piece()
        self.points = 0

        self.time_elapsed = 0
        self.time_limit = time_limit
        self.total_time = 0
        self.game_over = False

    def update(self, dt):
        """Updates the game by {fps} times a second"""

        if self.total_time >= self.time_limit > -1:
            self.game_over = True

        # if self.frames_elapsed >= self.fps / 2 and not self.game_over:
        if self.time_elapsed >= 500 and not self.game_over:
            self.time_elapsed = 0
            self.piece_moved = True

            if not self.check_move(0, 1):
                self.game_over = self.apply_piece()
                if not self.game_over:
                    self.new_piece()
                self.check_rows()
            else:
                self.move_down()

        self.time_elapsed += dt
        # if self.time_limit > -1: self.time_elapsed += 1.0 / self.fps
        if self.time_limit > -1:
            self.total_time += dt

        image = self.serve_image() if self.piece_moved else None
        self.piece_moved = False

        return image, self.game_over

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
        self.piece_moved = True
        if self.check_rotate():
            self.piece.rotate()

    def update_piece(self, x, y):
        """Update piece on board"""
        self.piece_moved = True
        if self.check_move(x, y):
            self.piece.move(x, y)

    def check_move(self, x, y, piece=None):
        """ Determines whether a piece is able to move"""
        can_move = True

        if piece is None:
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
                        can_move = False
                    elif data[0]:  # Block exists
                        can_move = (can_move and not
                                    self.board[pos[1] + y, pos[0] + x])

        return can_move

    def check_rotate(self):
        """ Determines whether a piece is able to rotate"""
        rot_piece = self.piece.clone()
        rot_piece.rotate()

        return self.check_move(0, 0, rot_piece)

    def new_piece(self):
        """ Update with new pieces"""
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
        """ Create a new piece"""
        block_type = int(r.random() * len(shapes))
        # block_type = 5
        x_pos = int(r.random() * (self.width - len(shapes[block_type][0])))
        piece = Stone(block_type, x_pos=x_pos, y_pos=0)

        return piece

    def get_future_position(self, piece=None):
        """ Calculate position if piece moves straight down"""
        if piece is None:
            piece = self.piece

        future_piece = piece.clone()
        valid_pos = True
        while valid_pos:
            valid_pos = self.check_move(0, 1, piece=future_piece)
            if valid_pos:
                future_piece.move(0, 1)
        return future_piece

    def apply_piece(self, piece=None, board=None, board_colors=None, compute_colors=True):
        """ Places all blocks of the piece onto the board"""
        if piece is None:
            piece = self.piece
        if board is None:
            board = self.board
        if board_colors is None and compute_colors:
            board_colors = self.colors

        is_game_over = False

        stone = piece.get_piece()
        for i in range(len(stone)):
            for j in range(len(stone[0])):
                pos = stone[i][j][0]
                data = stone[i][j][1]

                if data[0]:
                    if pos[1] <= 0:
                        is_game_over = True
                    else:
                        try:
                            board[pos[1], pos[0]] = data[0]
                            if compute_colors:
                                board_colors[pos[1], pos[0]] = data[1]
                        except IndexError:
                            traceback.print_exc()
        return is_game_over

    def check_rows(self):
        """ Checks each row for completion, and calculates points"""
        rows_done = []
        has_block = True
        j = self.height - 1
        while j > -1 and has_block:
            full_row = True  # checks the row for fullness
            has_block = False
            i = 0
            while i < self.width and (full_row or not has_block):
                full_row = full_row and self.board[j, i]
                has_block = has_block or self.board[j, i]
                i += 1
            if full_row:
                rows_done.append(j)
            j -= 1

        if len(rows_done) > 0:
            self.board = np.vstack((np.zeros((len(rows_done), self.width)),
                                    np.delete(self.board, rows_done, axis=0)))

        base_points = (0, 40, 100, 300, 1200)
        self.points += base_points[len(rows_done)]

    def serve_image(self):
        new_board = np.copy(self.board)
        self.apply_piece(board=new_board, compute_colors=False)
        return new_board.reshape(22, 10)

    def is_game_over(self):
        """ Returns whether the game has ended"""
        return self.game_over
