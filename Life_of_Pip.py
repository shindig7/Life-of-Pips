# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.initializers import RandomNormal
import numpy as np
from numpy import random
from scipy.stats.norm import pdf

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
        self.Lifetime = 0
    
    def set_Vision(self, board):
        try:
            v_list = [item for sublist in board[self.X-2:self.X+3,
                                                self.Y-2:self.Y+3].tolist() for item in sublist]
            v_list.remove(100)
            v_array = np.array(v_list).reshape(1,24)
            self.Vision = v_array
        except ValueError:
            pass
        
    def set_pos(self, x, y):
        self.X = x
        self.Y = y
        
    def move(self, board, change):
        cx, cy = change
        new_x, new_y = self.X + cx, self.Y + cy
        if new_x >= 10 or new_y >= 10 or new_x < 0 or new_y < 0:
            return False
        else:
            if board[new_x][new_y] == 1:
                self.Fitness += 1
            
            board[new_x][new_y] = 100
            board[self.X][self.Y] = 0
            self.set_pos(new_x, new_y)
            return True
        
    def predict(self, board):
        pred = self.brain.predict(self.Vision)
        d = np.where(pred == pred.max())[1][0] + 1
        decision = self.Actions[d]
        r = self.move(board, decision)
        if r:
            self.set_Vision(board)
        else:
            pass

        

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

def get_normal(xL, xU, step, mu, sig):
    x = np.arange(xL, xU, step)
    prob = pdf(x, loc=mu, scale=sig)
    prob = prob / prob.sum()
    v = random.choice(x, p=prob)
    return v

def play(minutes):
    p = Pip(5,5)
    board = create_board(10, 10, [(5,5)], food_pos=food_pos)
    p.set_Vision(board)

    for i in range(minutes):
        p.predict(board)
        p.Lifetime += 1

    return p

def Breed(PipXX, PipXY):
    def combine_or_mutate(rx, ry):
        c = [rx, ry, 'mutate']
        w = [4, 4, 2]
        choice = random.choices(c, w)
        if choice == 'mutate':
            cc = random.choice(rx, ry)
            new_gene = cc * get_normal(.5, 2, .01, 1.25, .25)
        else:
            new_gene = choice
        return new_gene
            
        
    xw = PipXX.brain.get_weights()[:7]
    yw = PipXY.brain.get_weights()[:7]
    
    new_weight = []
    for weight_index in range(len(xw)):
        new_row = []
        for row_index in range(len(xw)):
            rx = xw[weight_index][row_index]
            ry = yw[weight_index][row_index]

            #for i in range(rx.size):        
            new_row.append(combine_or_mutate(rx, ry))
        new_weight.append(new_row)
    
    return new_weight


X, Y = [play(50) for _ in range(2)]
B = Breed(X, Y)