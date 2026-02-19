#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:48:09 2023

@author: ffiguere
This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""
import numpy as np
import ddfenicsx as dd
import fetricksx as ft
import scipy.optimize as opt
    
class DDSearchNNLS(dd.DDSearch):
    def __init__(self, metric, ddmat, algorithm = 'ball_tree', norm ='euclidean', opInit = 'zero', seed = 0, 
                 L2reg = 0.0001, penfac = 10e4, kneigh = 5):
        self.L2reg = L2reg
        self.penfac = penfac
        self.kneigh = kneigh
        super().__init__(metric, ddmat, algorithm, norm, opInit, seed)     
        
    def find_neighbours(self, z): 
        if(self.ddmat.DB.shape[0]<self.kneigh):
            return self.find_neighbours_classic(z, kneigh = 1) 
        
        ng, strain_dim = z.shape
        zbar = self.metric.transformL(z) # ng x strain_dim
        zdbbar = self.metric.transformL(self.ddmat.DB.reshape((-1,strain_dim)))
        zdb = np.zeros(zbar.shape) # to be predicted
        dist_loc = np.zeros(ng)
        
        ones = np.ones(self.kneigh).reshape((1,self.kneigh))
        
        self.fit(self.ddmat.DB)
        self.local_dist, self.map = self.modelTree.kneighbors(zbar, self.kneigh)
    
        A = np.zeros((strain_dim + 1 + self.kneigh, self.kneigh)) 
        b = np.zeros(strain_dim + 1 + self.kneigh)
        for i in range(ng):
            A[:strain_dim,:] = zdbbar[self.map[i,:],:].T
            penfac = self.penfac*np.linalg.norm(A[:strain_dim,:], 'fro')/self.kneigh
            L2reg = self.L2reg*penfac/self.penfac
            
            A[strain_dim, :] = np.sqrt(penfac)*ones
            A[strain_dim + 1:, np.arange(self.kneigh)]= L2reg*ones
            
            b[:strain_dim] = zbar[i,:] 
            b[strain_dim] = np.sqrt(penfac)
            
            w = opt.nnls(A,b)[0]
            zdb[i,:] = self.ddmat.DB.reshape((-1,strain_dim))[self.map[i,:],:].T @ w
            dist_loc[i] = np.linalg.norm(zbar[i,:] - A[:strain_dim,:]@w)
            
            
        self.global_dist = self.metric.distance_distTree(dist_loc)
        
        
        return zdb.reshape((ng,2,-1)) 
    
    def find_neighbours_classic(self, z, kneigh = 1):
        # print(z.shape)
        self.local_dist , self.map = self.modelTree.kneighbors(self.metric.transformL(z), kneigh) # dist and map
        self.global_dist = self.metric.distance_distTree(self.local_dist[:,0])
        # print(np.max(self.map[:,0]), self.map[:,0].shape[0], self.ddmat.DB.shape[0], kneigh)
        # input()
        return self.ddmat.DB[self.map[:,0],:,:] 
    
        
