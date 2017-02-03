import tkinter as tk
import random as r

root = tk.Tk()
root.title = "Tetris"
root.resizable(0,0)

colors = ('#8F4144',
		  '#2BFDFF',
		  '#61DCDB',
		  '#FFB6C1')

canvas = tk.Canvas(root, width=280, height=616, bd=5, highlightthickness=0)

def draw_grid():
	canvas.delete('all')

	create_board(generate_board())

	canvas.pack()
	root.after(1000,draw_grid)

def generate_board():
	return [[(not (r.random()*(j/22)<0.3), colors[int(r.random()*4)]) for i in range(10)] for j in range(22)]

def create_board(grid):
	for i in range(10):
		for j in range(22):
			if grid[j][i][0]:
				canvas.create_rectangle(i*28, j*28, (i+1)*28, (j+1)*28, fill=grid[j][i][1])

draw_grid()
root.mainloop()