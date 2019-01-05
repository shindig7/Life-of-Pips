# -*- coding: utf-8 -*-
#import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.initializers import RandomNormal
import numpy as np
from numpy import random
import scipy.stats as ss
from string import ascii_lowercase


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
        self.Name = ''.join([random.choice(list(ascii_lowercase)) for _ in range(8)])
        
    
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
        lx, ly = board.shape
        cx, cy = change
        new_x, new_y = self.X + cx, self.Y + cy
        if new_x >= lx-1 or new_y >= ly-1 or new_x < 0 or new_y < 0:
            return False
        else:
            if board[new_y][new_x] == 1:
                self.Fitness += 1
            
            board[new_x][new_y] = 100
            board[self.X][self.Y] = 0
            self.set_pos(new_x, new_y)
            return True
        
    def choose_move(self, board):
        if self.Vision is None:
            self.set_Vision(board)
        pred = self.brain.predict(self.Vision)
        d = np.where(pred == pred.max())[1][0] + 1
        decision = self.Actions[d]
        r = self.move(board, decision)
        if r:
            self.set_Vision(board)
        else:
            pass
        
    def __str__(self):
        return "Name: {} Fitness: {}".format(self.Name, self.Fitness)

        

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
    prob = ss.norm.pdf(x, loc=mu, scale=sig)
    prob = prob / prob.sum()
    v = random.choice(x, p=prob)
    return v

def play(minutes, pip=None):
    if not pip:
        pip = Pip(5,5)
    
    board = create_board(15, 14, [(5,5)], food_pos=food_pos)
    pip.set_Vision(board)

    for i in range(minutes):
        pip.choose_move(board)
        pip.Lifetime += 1
        #print(board)

    return pip


def Breed(PipXX, PipXY):
    def combine_or_mutate(rx, ry):
        o = [rx, ry, 'mutate']
        c = [0, 1, 2]
        w = [.4, .4, .2]
        choice = random.choice(c, p=w)
        """
        if type(choice) is np.str_:
            cc = random.choice([1,2])
            if cc == 1:
                m_gene = rx
            else:
                m_gene = ry
            new_gene = m_gene * get_normal(.5, 2, .01, 1.25, .25)
        else:
            new_gene = choice
            """
        if choice == 2:
            cc = random.choice([1, 2])
            if cc == 1:
                m_gene = rx
            else:
                m_gene = ry
            new_gene = m_gene * get_normal(.5,2,.01,1.25, .25)
        else:
            new_gene = o[choice]
        return new_gene
            
        
    xw = PipXX.brain.get_weights()
    yw = PipXY.brain.get_weights()
    
    """
    new_pip_weights = []
    for i, layer in enumerate(xw):
        if len(layer) == 10:
            new_layer = np.zeros((1,10))
            for k, value in enumerate(layer):
                new_layer[0, k] = combine_or_mutate(xw[i][0, k], yw[i][0, k])
        else:
            new_layer = np.zeros((len(layer), int(layer.size / len(layer))))
            for j, input_weight in enumerate(layer):
                for k, value in enumerate(input_weight):
                    new_layer[j, k] = combine_or_mutate(xw[i][j, k], yw[i][j, k])
        new_pip_weights.append(new_layer)
    """
    new_pip_weights = [np.zeros(l.shape) for l in xw]
    for i, (xlayer, ylayer) in enumerate(zip(xw, yw)):
        for j, (xweight, yweight) in enumerate(zip(xlayer, ylayer)):
            new_pip_weights[i][j] = combine_or_mutate(xweight, yweight)
            
    offspring = Pip(random.choice(range(15)), random.choice(range(14)))
    offspring.brain.set_weights(new_pip_weights)
    
    offspring.Name = PipXX.Name[:4] + PipXY.Name[4:]
    
    return offspring
    
#X, Y = [play(50) for _ in range(2)]
def run_generation(population):
    pip_fit = sorted([(p, p.Fitness) for p in population], key=lambda k: k[1], reverse=True)[:-5]
    lottery = pip_fit[10:]
    
    pool = pip_fit[:10]
    print("Most Fit:")
    for p, _ in pool:
        print(p)
    
    for _ in range(5):
        lottery.append((Pip(5, 5), 0))
    
    for _ in range(16):
        winner = random.randint(0, len(lottery)-1)
        pool.append(lottery.pop(winner))
    
    print("Pool")
    for p in pool:
        print(p)
    
    breeders = [t[0] for t in pool]
    parents = []
    while breeders:
        dad_loc = random.randint(0, len(breeders)-1)
        dad = breeders.pop(dad_loc)
        if len(breeders) == 1:
            mom = breeders.pop()
        else:
            mom_loc = random.randint(0, len(breeders)-1)
            mom = breeders.pop(mom_loc)
        parents.append((dad, mom))
        
    next_gen = []
    for dad, mom in parents:
        print("Breeding {} and {}...".format(dad.Name, mom.Name))
        next_gen.append(Breed(dad, mom))
        next_gen.append(Breed(mom, dad))
    
    for i, child in enumerate(next_gen):
        child = play(50, pip=child)
        next_gen[i] = child
        
    ngf = [p.Fitness for p in next_gen]
    print("Max: {}\nMin: {}\nMean: {}\nMedian: \n".format(max(ngf), min(ngf), np.mean(ngf), np.median(ngf)))
    return next_gen    
        
population = [play(50) for _ in range(50)]
for i in range(10):
    print("Generation {}...".format(i))
    population = run_generation(population)
