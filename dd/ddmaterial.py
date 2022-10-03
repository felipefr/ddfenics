#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:20:19 2022

@author: felipe
"""

import dolfin as df
import numpy as np
from fetricks import symgrad_mandel
import matplotlib.pyplot as plt

class DDMaterial:
    
    def __init__(self, DB, C = None, grad = symgrad_mandel, addzero=True, shuffle=1):
        
        if(type(DB) == type('s')):
            self.DB = self.read_data(DB)
        else:
            self.DB = DB
        
        if(shuffle>-1):
            self.shuffleData(shuffle)

        if(addzero):
            self.addZeroState()
        
        self.grad = grad
        
        if(type(C) == type(None)):
            self.C = self.estimateC()
        else:
            self.C = C
            
        self.setStressFunctions()
        
    def setStressFunctions(self):
        
        if(self.C.shape[0] == 1 and self.C.shape[1] == 1):
            self.Cfe = df.Constant(self.C[0,0])
        else:
            self.Cfe = df.as_tensor(self.C)
                
        self.sigC = lambda u : self.Cfe*self.grad(u)
        self.sigC_e = lambda e : self.Cfe*e
    
    def estimateC(self, ):
                
        # U, sig, VT = np.linalg.svd(self.DB.reshape((-1,6)) ,full_matrices = False)
        
        # A = VT[:6,:6]*sig[:6]
        # print(A.shape)
        # Cest = A[3:6,3:6]@np.linalg.inv(A[:3,:3])
        
        
        Corr = self.DB.reshape((-1,6)).T@self.DB.reshape((-1,6))
        sig, U = np.linalg.eigh(Corr)

        asort = np.argsort(sig)
        sig = sig[asort[::-1]]
        U = U[:,asort[::-1]]
        
        Cest = U[3:6,:3]@np.linalg.inv(U[:3,:3])    
        
        self.C = 0.5*(Cest + Cest.T) 
        
        # Id = np.eye(3) 
        # E11 = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 0.]])
        
        # b1 = np.dot(Cest.flatten(), Id.flatten())
        # b2 = np.dot(Cest.flatten(), E11.flatten())
        
        # lamb = (3*b1 - 2*b2)/8
        # mu = (2*b2 - b1)/8
        
        # self.C = lamb*E11 + 2*mu*Id
    
        return self.C
    
            

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
        np.savetxt(datafile, self.DB, header = '1.0 \n%d 2 3 3'%Nd, comments = '', fmt='%.8e', )


    def addZeroState(self):
        # add zero to database
        Z = np.zeros((1,self.DB.shape[1], self.DB.shape[2]))
        self.DB = np.append(Z, self.DB,axis=0)
        
    def shuffleData(self, seed = 1):
        print("shuffling")
        np.random.seed(seed)
        np.random.shuffle(self.DB) # along the axis = 0    
        
        
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
