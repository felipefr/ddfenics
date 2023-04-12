#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:28:22 2023

@author: ffiguere

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors
import numpy as np

class DDSearch:
    def __init__(self, metric, ddmat, algorithm = 'ball_tree', norm ='euclidean', opInit = 'zero', seed = 0):
        self.metric = metric
        self.ddmat = ddmat
        
        self.modelTree = NearestNeighbors(n_neighbors=1, algorithm = algorithm, metric = norm) 
        self.fit(self.ddmat.DB)
        self.map = self.init_map(opInit, seed, self.metric.sh0.dim(), len(self.ddmat.DB))
    
    def fit(self, DB):
        self.modelTree.fit(self.metric.transformL(DB.reshape(DB.shape[0],-1)))        
        
    def find_neighbours(self, z, kneigh = 1):
        self.local_dist , self.map = self.modelTree.kneighbors(self.metric.transformL(z), kneigh) # dist and map
        self.global_dist = self.metric.distance_distTree(self.local_dist[:,0])
        return self.ddmat.DB[self.map[:,0],:,:] 
    
    @staticmethod
    def init_map(op, seed, ng, Nd):
        
        if(op == 'zero'):
            return np.zeros((ng,1),dtype=int)
            
        elif(op == 'same'): # same order
            return np.arange(ng, dtype = int).reshape((ng,1))
            
        elif(op == 'random'):
            np.random.seed(seed)
            indexes = np.random.randint(0, Nd, ng)
            return indexes.reshape((ng,1))
        
        
        
