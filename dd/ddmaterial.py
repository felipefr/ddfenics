#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:42:45 2023

@author: ffiguere
"""


import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

class DDMaterial:
    
    def __init__(self, DB, addzero=True, shuffle=1, setStressFunctions = True):
        
        if(type(DB) == type('s')):
            self.DB = self.read_data(DB)
        else:
            self.DB = DB

        self.strain_dim = self.DB.shape[-1]
        
        if(shuffle>-1):
            self.shuffleData(shuffle)

        if(addzero):
            self.addZeroState()
        
    def read_data(self, filename):
        file = open(filename,'r')
        fmt = float(file.readline())
        if (fmt > 1.0):
            raise RuntimeError("Unknown material database format")
        words = file.readline().split()
        DBsize = int(words[0])
        nItems = int(words[1])
        itemSz = []
        for i in range(nItems):
            itemSz.append(int(words[i+2]))
        
        data = np.zeros((DBsize,nItems,max(itemSz)))
 
        for n in range(DBsize):
            words = file.readline().split()
            ij = 0
            for i in range(nItems):
                for j in range(itemSz[i]):
                    data[n,i,j] = float(words[ij])
                    ij += 1
        
        return data

    def write_data(self, datafile):
        np.savetxt(datafile, self.DB, comments = '', fmt='%.8e', 
                   header = '1.0 \n%d 2 %d %d'%(len(self.DB), self.strain_dim, self.strain_dim) )

    def addZeroState(self):
        # add zero to database
        Z = np.zeros((1,self.DB.shape[1], self.DB.shape[2]))
        self.DB = np.append(Z, self.DB,axis=0)
        
    def shuffleData(self, seed = 1):
        print("shuffling")
        np.random.seed(seed)
        np.random.shuffle(self.DB) # along the axis = 0    
        
    
    # only for 2d
    def plotDB(self, namefig = None):

        color = lambda d : np.linspace(1,d.shape[0],d.shape[0])

        data = self.DB.reshape((-1,self.DB.shape[1]*self.DB.shape[2]))
        
        fig,(ax1,ax2) = plt.subplots(1,2)
        
        
        ax1.set_xlabel(r'$\epsilon_{xx}+\epsilon_{yy}$')
        ax1.set_ylabel(r'$\sigma_{xx}+\sigma_{yy}$')
        ax1.scatter(data[:,0] + data[:,1], data[:,3] + data[:,4], s = 20,
                    c = color(data), marker = '.')
        
        ax1.grid()
        
        ax2.set_xlabel(r'$\epsilon_{xy}$')
        ax2.set_ylabel(r'$\sigma_{xy}$')
        ax2.scatter(data[:,2], data[:,5],  s = 20,
                    c = color(data), marker = '.')
    
        ax2.grid()
        
        if(type(namefig) is not type(None)):            
            plt.savefig(namefig)
