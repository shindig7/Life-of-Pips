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
        self.Space_List = []
        

    def set_space_list(self, sl):
        self.Space_List = sl
        
        
    def set_array(self):
        if len(self.Space_List) == 0:
            print('Error: Board has no Space_List, exiting process')
        else:
            for space in self.Space_List:
                self.Array[space.Y, space.X] = space.Contents
    
    @property
    def as_dict(self):
        return {(s.X, s.Y): s.Contents for s in self.Space_List}
    
    
    def create_from_dict(self, input_dict):
        for i in range(self.X):
            for j in range(self.Y):
                self.Space_List.append(Space(i, j, input_dict[(i, j)]))
        self.set_array()
    
    
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
        
        
#if __name__ == "__main__":
