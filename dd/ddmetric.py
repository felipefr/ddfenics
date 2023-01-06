#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:44:55 2023

@author: ffiguere
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:18:12 2022

@author: felipe
"""

import numpy as np
import dolfin as df
import math

from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors

def decideDistanceSpace(V):
    if(V.representation == "Quadrature"):
        Qe = df.VectorElement("Quadrature", V.mesh().ufl_cell(), 
                              degree = V.sub(0).ufl_element().degree(), dim = 1, quad_scheme='default')
        sh0 = df.FunctionSpace(V.mesh(), Qe ) # for stress
    
    elif(V.representation == "DG"):
        sh0 = df.FunctionSpace(V.mesh() , 'DG', 0)    
        
    return sh0

class DDMetric:
    def __init__(self, C = None, omega = 0.0, alpha = 1.0, V = None, dx = None, ddmat = None):
        self.V = V
        self.omega = omega
        self.alpha = alpha
        
        if(type(self.V) is not type(None) and type(dx) is type(None)):
            self.dx = self.V.dxm
        else:
            self.dx = dx
        
        self.C = None
        if(not isinstance(C, type(None))): # if it's not None
            self.reset(C)
        else:
            self.reset(self.__estimateC(ddmat.DB))
            
        if(type(self.V) != type(None)):
            self.set_fenics_metric()
        
        self.modelTree = NearestNeighbors(n_neighbors=1, algorithm = 'ball_tree', metric='euclidean') 
        
        self.sh0 = decideDistanceSpace(self.V)
        self.dist_func = df.Function(self.sh0)

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
        return np.linalg.norm(self.transformL(state))

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
        
        self.a_energy = 2.0*df.inner(self.dsig, self.deps)*self.dx
        
        self.C_fe = C_fe
        self.Cinv_fe = Cinv_fe
        
    def dist_fenics(self, z_mech, z_db = None):
        if(type(z_db) == type(None) ):
            self.deps.assign(z_mech[0])
            self.dsig.assign(z_mech[1])
        else:
            self.deps.assign(z_mech[0] - z_db[0])
            self.dsig.assign(z_mech[1] - z_db[1])
            
        return np.sqrt( df.assemble(self.a_metric) ) 
    

    def dist_form_fenics(self, z_mech, z_db):
        n = self.C.shape[0]
        C_fe = df.Constant(self.C)
        Cinv_fe = df.Constant(self.CC[n:2*n,n:2*n]) # takes into account alpha
        
        deps = z_mech[0] - z_db[0]
        dsig = z_mech[1] - z_db[1]
        
        form = (df.inner(C_fe*deps, deps)*self.dx + 
                df.inner(Cinv_fe*dsig, dsig)*self.dx + 
                2*self.omega*df.inner(dsig, deps)*self.dx)
            
        return form  
    

    def dist_energy_fenics(self, z_mech, z_db = None):
        if(type(z_db) == type(None) ):
            self.deps.assign(z_mech[0])
            self.dsig.assign(z_mech[1])
        else:
            self.deps.assign(z_mech[0] - z_db[0])
            self.dsig.assign(z_mech[1] - z_db[1])
            
        return np.sqrt( df.assemble(self.a_energy) ) 

    def diagonal(self):
        return DDMetric(self.C, omega = 0.0, V = self.V, alpha = 1.0, dx = self.dx)
    
    def fitTree(self,  DB):
        self.modelTree.fit(self.transformL(DB))        
        return self.modelTree
    
    def findNeighbours(self, states):
        distTree , map = self.modelTree.kneighbors(self.transformL(states)) # dist and map
        
        return self.distance_distTree(distTree), distTree, map 

    def distance_distTree(self, distTree):
        self.dist_func.vector().set_local(distTree)
        return np.sqrt(df.assemble((self.dist_func**2)*self.dx)) # L2 norm


    def __estimateC(self, DB):
        strain_dim = DB.shape[-1]
    
        Corr = DB.reshape((-1,2*strain_dim)).T@DB.reshape((-1,2*strain_dim))
        sig, U = np.linalg.eigh(Corr)
    
        asort = np.argsort(sig)
        sig = sig[asort[::-1]]
        U = U[:,asort[::-1]]
        
        Cest = U[strain_dim:2*strain_dim,:strain_dim]@np.linalg.inv(U[:strain_dim,:strain_dim])    
        
        C = 0.5*(Cest + Cest.T) 
    
        return C


    # if(not isPD(self.C)):
    #     self.C = get_near_psd(self.C)
    
    # lmax = 600.0
    
    # self.C = (lmax/max(lmax, np.max(np.linalg.eigvals(self.C))))*self.C
    
    # self.C = nearestPD(self.C)
    
    # print(np.linalg.eigvals(self.C))
    # assert np.all(np.linalg.eigvals(self.C)) > 0, "C is not PD : {0}".format(self.C)
    # input()
    
    # print(self.C)
    
    # input()
    
    
    # Id = np.eye(3) 
    # E11 = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 0.]])
    
    # b1 = np.dot(Cest.flatten(), Id.flatten())
    # b2 = np.dot(Cest.flatten(), E11.flatten())
    
    # lamb = (3*b1 - 2*b2)/8
    # mu = (2*b2 - b1)/8
    
    # self.C = lamb*E11 + 2*mu*Id
    # U, sig, VT = np.linalg.svd(self.DB.reshape((-1,6)) ,full_matrices = False)
    
    # A = np.diag(sig[:3])@VT[:3,:6]
    
    # Cest = A[:3,3:6]@np.linalg.inv(A[:3,:3])
    
    # self.C = 0.5*(Cest + Cest.T) 
    
    # print(self.C)
    