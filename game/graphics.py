import tkinter as tk
import random as r

root = tk.Tk()
root.title = "Tetris"
root.resizable(0,0)

colors = ('#8F4144',
		  '#2BFDFF',
		  '#61DCDB',
		  '#FFB6C1')

dim = 28

canvas = tk.Canvas(root, width=dim*10+1, height=dim*22, bd=5, highlightthickness=0, bg='#192317')

t = 1000/60
t_e = 0
board = None

def draw_grid():
	global t, t_e, board
	canvas.delete('all')

	if t_e>=1000 or not board:
		t_e = 0
		board = generate_board()
	create_board(board)

	canvas.pack()
	t_e+=t
	root.after(int(t),draw_grid)

def generate_board():
	return [[(not (r.random()*(j/22)<0.3), colors[int(r.random()*4)]) for i in range(10)] for j in range(22)]

def create_board(grid):
	for i in range(10):
		for j in range(22):
			if grid[j][i][0]:
				canvas.create_rectangle(5+i*dim, j*dim, 5+(i+1)*dim, (j+1)*dim, fill=grid[j][i][1])

draw_grid()
root.mainloop()
