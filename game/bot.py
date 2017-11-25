import json
import numpy as np
import traceback


class NoData(Exception):
    def __init__(self):
        self.message = "No data found in file"
        Exception.__init__(self, self.message)


def write_data(data, filename='weights.txt'):
    ''' Write data to a json file'''
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def read_data(filename='weights.txt'):
    ''' Read data from json file'''
    data = None
    with open(filename, 'r') as infile:
        data = json.load(infile)
    return data


class Bot(object):
    def __init__(self, game, training=True):
        ''' Initialize a tetris AI object'''
        self.weights = np.asmatrix([-0.03, -7.5, -3.5, 80, 3, 2.5, 5]).getT()
        self.game = game

    def update(self, data):
        ''' Called every drawn frame, lets the bot make decisions'''
        if not data or not self.game:
            return

        calc_vals = self.calc_value(data)
        best = self.get_best_state(calc_vals)

        # print("{} ||| {}".format(best, calc_vals))
        rot = best[0]
        x_pos = best[1]

        while self.game.piece.rotation is not rot:
            self.game.rotate()

        if self.game.piece.x_pos is not x_pos:
            self.game.piece.x_pos = x_pos
        else:
            self.game.move_down()

    def calc_value(self, data):
        ''' Calculates the value of every case'''
        values = []
        for r1 in range(len(data)):
            for x1 in range(len(data[r1])):
                for r2 in range(len(data[r1][x1])):
                    for x2 in range(len(data[r1][x1][r2])):
                        try:
                            input_data = np.asmatrix(data[r1][x1][r2][x2])
                        except IndexError as e:
                            traceback.print_exc()
                            # print("{}, {}, {}, {}, {}".format(r1,x1,r2,x2, len(self.game.piece.block)))
                            print("{}, {}, {}".format(len(data[r1]), len(data[0]), x1))
                        result = input_data * self.weights
                        values.append((r1, x1, r2, x2, result))
        return values

    def get_best_state(self, values):
        ''' Finds the best input state for this piece'''
        largest_val = 0
        val_index = 0

        for i in range(len(values)):
            if values[i][4] > largest_val:
                largest_val = values[i][4]
                val_index = i
        return values[val_index]
