# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 13:25:16 2018

@author: eagle
"""
from random import randint, choice
import numpy as np
from math import sqrt, e, pi
import scipy.stats as ss


PIP_MASTER_LIST = []
POSITION_LIST = []

class Landscape():
    def __init__(self, width, length):
        self.Width = width
        self.Length = length
        self.grid = np.zeros((length, width))
        
    def __repl__(self):
        return self.grid

class Pip():
    def __init__(self, x, y, score):
        self.X = x
        self.Y = y
        self.overall_score = score
        self.vision = []
        self.location = (self.X, self.Y)
                
    def move(self, mode='random'):
        direction = {'left': (-1, 0), 'right': (1,0), 'up': (0,1), 'down': (0,-1),
                     'dul': (-1, 1), 'dur': (1,1), 'ddl': (-1,-1), 'ddr': (1,-1)}
        
        try:
            POSITION_LIST.remove((self.X, self.Y))
        except ValueError:
            pass
        
        if mode=='random':
            xc, yc = direction[choice(list(direction.keys()))]
        else:
            xc, yc = direction[mode]
        
        new_x = self.X + xc
        new_y = self.Y + yc

        if not check_empty((new_x, new_y)):
            if mode == 'random':
                while not check_empty((new_x, new_y)):
                    xc, yc = direction[choice(list(direction.keys()))]
                    new_x = self.X + xc
                    new_y = self.Y + yc
                    self.X = new_x
                    self.Y = new_y
                    
                POSITION_LIST.append((new_x, new_y))   
                self.location = (new_x, new_y) 
            else:
                print('ERROR: Space is not empty')
                    
                   

    def __str__(self):
        return ("X: {}, Y: {}, Score: {}".format(self.X, self.Y, self.overall_score))


def get_normal_value(xL, xU, step, mu, sig):
    x = np.arange(xL, xU, step)
    prob = ss.norm.pdf(x, loc=mu, scale=sig)
    prob = prob / prob.sum()
    n = np.random.choice(x, p=prob)
    return n

        
def check_empty(position):
    if position in POSITION_LIST:
        return False
    else:
        return True
    
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def setup(width, length, number_pips):    
            
    for i in range(number_pips):
        x = randint(1,width-1)
        y = randint(1,length-1)
        while not check_empty((x, y)):
            x = randint(1,width-1)
            y = randint(1,length-1)
        p = Pip(x, y, randint(1,10))
        PIP_MASTER_LIST.append(p)
        POSITION_LIST.append((p.X, p.Y))
        
    for pip in PIP_MASTER_LIST:
        for p in PIP_MASTER_LIST:
            if pip != p:
                if distance(pip.location, p.location) < 2.83:
                    pip.vision.append(p)
    
    
def step():
    for pip in PIP_MASTER_LIST:
        pass        
    
def eq(mu, sig, x):
    return (1/sqrt(2*pi*sig**2))*e**(-((x-mu)**2/(2*sig**2)))


def mate(pip1, pip2):
    mc = choice([1,2])
    if mc == 1:
        mother = pip1
        father = pip2
    else:
        mother = pip2
        father = pip1
    mother_weight = get_normal_value(0,1,.01,.5,.1666)
    father_weight = 1- mother_weight
    child_score = (mother.overall_score * mother_weight) + (father.overall_score * father_weight)
    child = Pip(mother.X, mother.Y, child_score)
    child.move(mode='random')
    print(child)


def test():
    setup(5,5,5)
    pip = PIP_MASTER_LIST[0]
    print(pip.X, pip.Y)
    pip.move()
    print(pip.X, pip.Y)