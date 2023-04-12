#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:22:08 2022

@author: ffiguere

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""

import numpy as np
from timeit import default_timer as timer 

class DDSolver:
    
    def __init__(self, problem, search):
                
        self.problem = problem
        self.search = search

        # shortcuts
        self.ddmat = self.search.ddmat
        self.DB = self.ddmat.DB.view()
        self.metric = self.problem.metric    
        self.dx = self.problem.dx
        self.sol = self.problem.get_sol()

        # error metrics        
        self.hist = {'distance' : [], 'relative_distance': [], 'relative_energy': [], 'sizeDB': []}
        self.calls_hist = {}
        self.calls_hist['distance'] = lambda m, m_ref, m0 : m
        self.calls_hist['relative_distance'] = lambda m, m_ref, m0 : np.abs(m-m0)/m_ref
        self.calls_hist['relative_energy'] = lambda m, m_ref, m0 : (m/m_ref)**2 
        self.calls_hist['sizeDB'] = lambda m, m_ref, m0 : len(self.DB)
    
        
    def solve(self, tol = 0.001, maxit = 100):
        
        total_time_PE = 0.0
        total_time_PD = 0.0
    
        dist0 = self.distance_db_mech()
        error = 999.9
        k = 0
        
        while (error > tol and k < maxit):

            total_time_PE += self.project_onto_equilibrium()
            total_time_PD += self.project_onto_data() 
        
            norm_ref = self.norm_ref()
        
            self.append_hist(self.search.global_dist, norm_ref, dist0)

            error = self.hist["relative_distance"][-1]
            dist0 = self.search.global_dist            
            k+=1
            print("\tInner Iteration #%3d ,- Relative Error = %13.6e"%(k, error))

        return self.search.local_dist, k, total_time_PE, total_time_PD 
    

    def project_onto_equilibrium(self):
        start = timer()
        self.problem.solve()
        self.problem.update_state_mech()
        end = timer()
        return end - start
        
    def project_onto_data(self):
        start = timer()
        self.problem.update_state_db(self.search.find_neighbours(self.problem.get_state_mech_data()))
        end = timer()
        return end - start


    def distance_db_mech(self):
        return self.metric.dist_fenics(self.problem.z_mech, self.problem.z_db)

    def norm_ref(self):
        # return self.metric.dist_fenics(self.problem.z_mech) # only one argument becomes norm
        return self.metric.norm_energy_fenics(self.problem.z_mech) # only one argument becomes norm
        
    def distance_relative(self):
        # return self.distance_db_mech()/self.norm_ref()
        return self.metric.dist_energy_fenics(self.problem.z_mech, self.problem.z_db)/self.norm_ref()
       
    
    def append_hist(self, m, m_ref, m0 = 0.0):
        for key in self.hist.keys():
            self.hist[key].append(self.calls_hist[key](m, m_ref, m0))

    def get_sol(self):
        return self.sol
    
    def get_state_mech_data(self):
        return np.concatenate(tuple([z_i.vector().get_local()[:].reshape((-1, self.problem.strain_dim)) 
                                     for z_i in self.sol["state_mech"]]) , axis = 1)
