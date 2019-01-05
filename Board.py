# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 08:11:02 2019

@author: eagle
"""
import numpy as np

class Board(object):
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.Array = np.zeros((self.Y, self.X))
        self.Space_List = None

    def set_space_list(self, sl):
        self.Space_List = sl
        
    def set_array(self):
        if not self.Space_List:
            print('Error: Board has no Space_List, exiting process')
        else:
            for space in self.Space_List:
                self.Array[space.Y, space.X] = space.Contents
    
    def __repr__(self):
        return str(self.Array)
            
    
class Space(object):
    def __init__(self, x, y, contains=0):
        self.X = x
        self.Y = y
        self.Contents = contains
        
    def is_occupied(self):
        if self.Contents == 0:
            return False
        else:
            return True