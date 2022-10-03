#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:22:08 2022

@author: felipe
"""

import math
import numpy as np
import dolfin as df
from timeit import default_timer as timer 

from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors

from ddfenics.dd.ddmetric import DDMetric
from ddfenics.dd.ddla import DDla

class DDSolver:
    
    def __init__(self, problem, opInit = 'random', seed = 0):
                
        self.problem = problem
        self.DB = self.problem.ddmat.DB.view()
        
        # distance function
        self.metric = self.problem.metric
        
        if(type(self.problem.z_mech) == type([])):
            Sh0 = self.problem.z_mech[0].function_space()
        else:
            Sh0 = self.problem.z_mech.function_space()
        
        if(Sh0.ufl_element().family() == 'Quadrature'):
            Qe = df.VectorElement("Quadrature", Sh0.mesh().ufl_cell(), degree = Sh0.ufl_element().degree(), dim = 1, quad_scheme='default')
            self.sh0 = df.FunctionSpace(Sh0.mesh(), Qe ) # for stress
        
        elif(Sh0.ufl_element().family() == 'Mixed' and Sh0.sub(0).ufl_element().family() == 'Quadrature' ):
            Qe = df.VectorElement("Quadrature", Sh0.mesh().ufl_cell(), degree = Sh0.sub(0).ufl_element().degree(), dim = 1, quad_scheme='default')
            self.sh0 = df.FunctionSpace(Sh0.mesh(), Qe ) # for stress
        
        else:
            self.sh0 = df.FunctionSpace(self.problem.u.function_space().mesh() , 'DG', 0)    
        
        self.dx = self.problem.dx
        self.dist_func = df.Function(self.sh0)

        # build tree for database
        self.tree = self.buildTree()

        self.map = self.initialisation(opInit, seed)
        
        self.LS = DDla(self.problem)
        
        self.hist = {'distance' : [], 'relative_distance': [], 'relative_energy': [], 'sizeDB': [], 'classical_relative_energy': []}
        
        self.metric = self.problem.metric 
        self.metric_energy = self.metric.diagonal()
        
        self.calls_hist = {}
        self.calls_hist['distance'] = lambda m, m_ref, m0 : m
        self.calls_hist['relative_distance'] = lambda m, m_ref, m0 : np.abs(m-m0)/m_ref
        self.calls_hist['relative_energy'] = lambda m, m_ref, m0 : (m/m_ref)**2 
        self.calls_hist['sizeDB'] = lambda m, m_ref, m0 : len(self.problem.ddmat.DB)
        self.calls_hist['classical_relative_energy'] = lambda m, m_ref, m0 : (self.metric_energy.dist_fenics(self.problem.z_mech, self.problem.z_db)/m_ref)**2.0
        

    def initialisation(self, op, seed):
        # initialize mapping
        
        if(type(self.problem.z_mech) == type([])):
            ng = self.problem.z_mech[0].ng
        else:
            ng = self.problem.z_mech.ng
        
        if(op == 'zero'):
            return np.zeros((ng,1),dtype=int)
            
        elif(op == 'same'): # same order
            return np.arange(ng, dtype = int).reshape((ng,1))
            
        elif(op == 'random'):
            np.random.seed(seed)
            indexes = np.random.randint(0, len(self.DB), ng)
            
            return indexes.reshape((ng,1))
    
    def buildTree(self,  algorithm = 'ball_tree'):
    
        # metric = DistanceMetric.get_metric('mahalanobis', VI = distance.CC)
        # metric = DistanceMetric.get_metric('euclidean')
        #metric = DistanceMetric.get_metric('pyfunc', func = distance.dist_sqr)
        # tree = BallTree(data,leaf_size= 1000, metric=metric)
        
        model = NearestNeighbors(n_neighbors=1, algorithm = algorithm, metric='euclidean') 
        tree = model.fit(self.metric.transformL(self.DB.reshape(self.DB.shape[0],-1)))
    
        return tree
    
    def solve(self, tol = 0.001, maxit = 100):
    
        dist0 = self.distance()
        error = 999.9
        k = 0
        
        while (error > tol and k < maxit):
            self.LS.solve()
            distTree = self.findNeighbours()
            dist = self.distance_distTree(distTree)
            dist_ref = self.distance_ref()
            self.append_hist(dist, dist_ref, dist0)

            error = self.hist["relative_distance"][-1]
            dist0 = dist            
            k+=1
            print("\tInner Iteration #%3d ,- Relative Error = %13.6e"%(k, error))

        return distTree, k 
    
    
    def distance_distTree(self, distTree):
        self.dist_func.vector().set_local(distTree)
        return np.sqrt(df.assemble((self.dist_func**2)*self.dx)) # L2 norm
        
    def distance(self):
        return self.metric.dist_fenics(self.problem.z_mech, self.problem.z_db)

    def distance_ref(self):
        # energy_ref = 0.5*df.assemble(df.inner(self.problem.z_mech[0], self.problem.z_mech[1])*self.dx) # energy
        return self.metric.dist_fenics(self.problem.z_mech) # only one argument becomes norm
        
    def distance_relative(self):
        return self.distance()/self.distance_ref()
       
    
    def append_hist(self, m, m_ref, m0 = 0.0):
        for key in self.hist.keys():
            self.hist[key].append(self.calls_hist[key](m, m_ref, m0))

    def findNeighbours(self):
        distTree, self.map = self.tree.kneighbors(self.metric.transformL(self.problem.get_state_mech_data()))
        self.problem.update_state_db(self.map[:,0])
        
        return distTree
