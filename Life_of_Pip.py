# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.initializers import RandomNormal
import numpy as np

class Pip(object):
    def __init__(self, x, y):
        self.brain = Sequential()
        self.brain.add(layers.Dense(10, activation='relu',
                                    input_shape=(24,),
                                    bias_initializer='RandomNormal'))
        self.brain.add(layers.Dense(10, activation='relu',
                                    bias_initializer='RandomNormal'))
        self.brain.add(layers.Dense(10, activation='relu',
                                    bias_initializer='RandomNormal'))
        self.brain.add(layers.Dense(8, activation='sigmoid'))
        
        self.brain.compile(optimizer = "adam", loss = "binary_crossentropy",
                           metrics = ["accuracy"])
        
        self.Actions = {1: (-1, 1), 2: (0, 1), 3: (1, 1), 4: (1, 0), 5: (1, -1),
                        6: (0, -1), 7: (-1, -1), 8: (-1, 0)}
        
        self.Fitness = 0
        self.X = x
        self.Y = y
        self.Vision = None
    
    def set_Vision(self, board):
        v_list = [item for sublist in board[self.X-2:self.X+3,
                                            self.Y-2:self.Y+3].tolist() for item in sublist]
        v_list.remove(100)
        v_array = np.array(v_list).reshape(1,24)
        self.Vision = v_array
        
    def set_pos(self, x, y):
        self.X = x
        self.Y = y
        
    def move(self, board, change):
        cx, cy = change
        new_x, new_y = self.X + cx, self.Y + cy
        if new_x > 10 or new_y > 10 or new_x < 0 or new_y < 0:
            return False
        else:
            if board[new_x][new_y] == 1:
                self.Fitness += 1
            
            board[new_x][new_y] = 100
            board[self.X][self.Y] = 0
            self.X, self.Y = new_x, new_y
        
    def predict(self, board):
        pred = self.brain.predict(self.Vision)
        d = np.where(pred == pred.max())[1][0] + 1
        decision = self.Actions[d]
        r = self.move(board, decision)
        if r:
            self.set_Vision(board)
        else:
            self.predict(board)
        

def create_board(x, y, pip_pos, food_pos=False):
    board = np.zeros((x, y))
    for px, py in pip_pos:
        board[px][py] = 100
    if food_pos:
        for fx, fy in food_pos:
            board[fx][fy] = 1
    else:
        pass
    
    return board


"""
p = Pip(5,5)
board = create_board(10, 10, [(5,5)], [(4,4), (5,4)])
p.set_Vision(board)

pred = p.brain.predict(p.Vision)
decision = pred.max()
print(decision)
"""

food_pos = [(4,4), (5,4), (6,7), (7,9), (1,3),
            (1,9), (9,7), (5,9), (8,2), (2,1)]

def play(minutes):
    p = Pip(5,5)
    board = create_board(10, 10, [(5,5)], food_pos=food_pos)
    p.set_Vision(board)

    for i in range(minutes):
        p.predict(board)

    return p

Pips = [play(100) for _ in range(100)]    
        
        

