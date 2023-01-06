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

from ddfenics.dd.ddmetric import DDMetric


class DDSolver:
    
    def __init__(self, problem, opInit = 'random', seed = 0):
                
        self.problem = problem
        self.DB = self.problem.ddmat.DB.view()
        
        # distance function
        self.metric = self.problem.metric
    
        self.dx = self.problem.dx

        # build tree for database
        self.metric = self.problem.metric 
        self.metric_energy = self.metric.diagonal()
    
        self.metric.fitTree(self.DB.reshape(self.DB.shape[0],-1)) 

        self.map = self.initialisation(opInit, seed)
        

        
        self.hist = {'distance' : [], 'relative_distance': [], 'relative_energy': [], 'sizeDB': [], 'classical_relative_energy': []}
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
    
    
    def solve(self, tol = 0.001, maxit = 100):
        
        total_time_PE = 0.0
        total_time_PD = 0.0
    
        dist0 = self.distance_db_mech()
        error = 999.9
        k = 0
        
        while (error > tol and k < maxit):

            total_time_PE += self.project_onto_equilibrium()
            dist, distTree, delta_t_PD = self.project_onto_data()
            total_time_PD += delta_t_PD 
        
            norm_ref = self.norm_ref()
        
            self.append_hist(dist, norm_ref, dist0)

            error = self.hist["relative_distance"][-1]
            dist0 = dist            
            k+=1
            print("\tInner Iteration #%3d ,- Relative Error = %13.6e"%(k, error))

        return distTree, k, total_time_PE, total_time_PD 
    

    def project_onto_equilibrium(self):
        start = timer()
        self.problem.solve()
        self.problem.update_state_mech()
        end = timer()
        
        return end - start
        
    def project_onto_data(self):
        start = timer()
        dist, distTree, self.map = self.metric.findNeighbours(self.problem.get_state_mech_data())
        end = timer()
        self.problem.update_state_db(self.map[:,0])

        return dist, distTree, end - start


    def distance_db_mech(self):
        return self.metric.dist_fenics(self.problem.z_mech, self.problem.z_db)

    def norm_ref(self):
        return self.metric.dist_fenics(self.problem.z_mech) # only one argument becomes norm
        # return self.metric.dist_energy_fenics(self.problem.z_mech) # only one argument becomes norm
        
    def distance_relative(self):
        return self.distance_db_mech()/self.norm_ref()
       
    
    def append_hist(self, m, m_ref, m0 = 0.0):
        for key in self.hist.keys():
            self.hist[key].append(self.calls_hist[key](m, m_ref, m0))