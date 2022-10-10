#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:18:12 2022

@author: felipe
"""

import numpy as np
import dolfin as df
import math


class DDMetric:
    def __init__(self, C, omega = -0.5, alpha = 1.0, V = None, dx = None):
        self.V = V
        self.dx = dx
        self.omega = omega
        self.alpha = alpha

        self.reset(C)

        if(type(self.V) != type(None)):
            self.set_fenics_metric()

    def reset(self, C):
        self.C = C
        n = self.C.shape[0]
        self.CC = np.zeros((2*n,2*n))
        self.CC[0:n,0:n] = self.C
        self.CC[n:2*n,n:2*n] = self.alpha*np.linalg.inv(self.C)
        self.CC[0:n,n:2*n] = self.omega*np.eye(n)
        self.CC[n:2*n,0:n] = self.omega*np.eye(n)
        
        self.L = np.linalg.cholesky(self.CC) 
        
        self.set_fenics_metric()
        
    def dist(self,state1,state2):
        return self.norm(state1-state2)
    
    def dist_sqr(self,state1,state2):
        return math.sqrt(self.dist(state1,state2))
    
    def norm(self,state):
        return np.dot(state,np.dot(self.CC,state))
    
    def transformL(self, state):
        return state @ self.L
    
    def normL(self,state):
        return np.linalg.norm(self.tranformL(state))

    def set_fenics_metric(self):
        n = self.C.shape[0]
        C_fe = df.Constant(self.C)
        Cinv_fe = df.Constant(self.CC[n:2*n,n:2*n]) # takes into account alpha
        
        if(type(self.dx) == type(None) ):
            self.dx = df.Measure('dx', self.V.mesh())
    
        self.deps = df.Function(self.V)
        self.dsig = df.Function(self.V)
        
        self.a_metric = (df.inner(df.dot(C_fe, self.deps), self.deps)*self.dx + 
            df.inner(df.dot(Cinv_fe, self.dsig), self.dsig)*self.dx + 
            2*self.omega*df.inner(self.dsig, self.deps)*self.dx)

    def dist_fenics(self, z_mech, z_db = None):
        if(type(z_db) == type(None) ):
            self.deps.assign(z_mech[0])
            self.dsig.assign(z_mech[1])
        else:
            self.deps.assign(z_mech[0] - z_db[0])
            self.dsig.assign(z_mech[1] - z_db[1])
            
        return np.sqrt( df.assemble(self.a_metric) ) 
    

    def diagonal(self):
        return DDMetric(self.C, omega = 0.0, V = self.V, alpha = 1.0, dx = self.dx)
        