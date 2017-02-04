import tkinter as tk
import random as r
import sys

class Tetris(object):

	def __init__(self, dim=28, fps=60, is_human=True, level=0):
		''' Initializes a tetris game object'''
		self.dim = dim
		self.fps = fps
		self.level = level
		self.tick_ms = int(1000/self.fps)
		self.width = 10
		self.height = 22
		self.board = [[(False,) for n in range(self.height)]
						for m in range(self.width)]
		self.colors = ('#8F4144',
		  			   '#2BFDFF',
		  			   '#61DCDB',
		  			   '#FFB6C1')
		self.piece = None
		self.new_piece()
		self.points = 0

		self.root = tk.Tk()
		self.root.title = "Tetris"
		self.root.resizable(0,0)

		self.canvas = tk.Canvas(self.root, width=dim*self.width+1, 
			height=dim*self.height, bd=5, highlightthickness=0, bg='#192317')
		if is_human: self.bind_canvas()

		self.frames_elapsed = 0

		self.update()
		self.root.mainloop()

	def update(self):
		'''Updates the game by {fps} times a second'''
		self.canvas.delete('all')
		
		if self.frames_elapsed >= self.fps:
			self.frames_elapsed = 0

			if self.piece[3] >= 21 or self.board[self.piece[2]][self.piece[3]+1][0]:
				self.new_piece()
			else:
				self.move_down()

			self.check_rows()

		self.draw_board()
		self.canvas.create_text(40,15,text="Points: {}".format(self.points),
						fill='#ffff99')
		self.canvas.pack()
		self.frames_elapsed += 1
		self.root.after(self.tick_ms, self.update)

	def draw_board(self):
		'''Draws the board by adding rectangles to canvas'''
		for i in range(self.width):
			for j in range(self.height):
				if self.board[i][j][0]:
					self.canvas.create_rectangle(5+i*self.dim, j*self.dim,
						5+(i+1)*self.dim, (j+1)*self.dim, fill=self.board[i][j][1])

	def bind_canvas(self):
		'''Handles binding of canvas events'''
		self.canvas.bind("<Left>", self.move_left)
		self.canvas.bind("<Right>", self.move_right)
		self.canvas.bind("<Down>", self.move_down)
		self.canvas.bind("<Up>", self.rotate)
		self.canvas.bind("<Escape>", sys.exit)
		self.canvas.focus_set()

	def move_left(self, event=None):
		self.update_piece(-1,0)

	def move_right(self, event=None):
		self.update_piece(1,0)

	def move_down(self, event=None):
		self.update_piece(0,1)

	def rotate(self, event=None):
		pass

	def update_piece(self, x, y):
		'''Update piece on board'''
		self.board[self.piece[2]][self.piece[3]] = (False,)
		if (self.piece[2]+x >= 0 and self.piece[2]+x < self.width
			and not self.board[self.piece[2]+x][self.piece[3]][0]):
			self.piece[2] += x
		if (self.piece[3]+y >= 0 and self.piece[3]+y < self.height 
			and not self.board[self.piece[2]][self.piece[3]+y][0]):
			self.piece[3] += y
		self.board[self.piece[2]][self.piece[3]] = self.piece

	def new_piece(self):
		'''Create a new piece then update the board'''
		self.piece = [True, self.colors[int(r.random()*4)],
						int(r.random()*self.width), 0]
		self.update_piece(0,0)

	def check_rows(self):
		'''Checks each row for completion, and calculates points'''
		rows_done = 0
		for j in range(self.height):
			full_row = True			#checks the row for fullness
			for i in range(self.width):
				full_row &= self.board[i][j][0]
			if full_row:
				rows_done+=1
				for i in range(self.width):
					self.board[i][j] = (False,)
		base_points = (0,40,100,300,1200)
		self.points += base_points[rows_done] * (self.level+1)