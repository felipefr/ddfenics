#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:34:56 2023

@author: ffiguere
"""

import ddfenicsx as dd

class DDSolverNested(dd.DDSolver):
    
    def __init__(self, solver_inner, updater):
        
        self.solver_inner = solver_inner
        self.updater = updater
        
        super().__init__(self.solver_inner.problem, self.solver_inner.search)
        
        self.problem = self.solver_inner.problem
        self.DB = self.solver_inner.ddmat.DB
        self.map = self.updater.map
    
    # def schedule_maxupd(self, nit_inner):
    #     if(nit_inner<3):
    #         self.updater.reset_maxupdate(80)
    #     elif(nit_inner>18):
    #         self.updater.reset_maxupdate(10)
    #     else:
    #         self.updater.reset_maxupdate()
            
    # if inner solver is DDSolverDropout, it will call inner solver with default parameters
    
    def run_callbacks(self, callbacks, k):
        
        for foo in callbacks['inner']:
            foo(k, self.solver_inner)
        
        for foo in callbacks['outer']:
            foo(k, self)
            
            
    def solve(self, tolout = 0.01, tolin = 0.1, maxitin = 10, maxitout = 10, callbacks = []):
    
        self.problem.update_state_db(self.DB[self.map[:,0],:,:])
        
        error = 999.9
        k = 0 
        
        self.run_callbacks(callbacks, k)
        
            
        while (error > tolout and k < maxitout):
            # idea: some action que be based on nit_inner
            nit_inner = self.solver_inner.solve(tolin, maxitin)[1] 
            # self.schedule_maxupd(nit_inner)
            self.updater(self.solver_inner.get_state_mech_data(), k)
            self.solver_inner.project_onto_data() # original
            
            error = self.distance_relative()**2.0 # indeed relative energy 
            k+=1
            print("\tOuter Iteration #%3d ,- Relative Error Energy (energy) = %13.6e"%(k, error))
        

            self.run_callbacks(callbacks, k)
        
        # Optional, just to adjust to the last DB update
        self.solver_inner.solve(tolin, maxitin)
        self.project_onto_data()
        
        return self.hist, k
    
    # it allows to control inner dropout solver call with custom parameters 
    def solve_dropout(self, tolout = 0.01, tolin = 0.1, maxitin = 10, maxitout = 10, callbacks = [],                      
                      p_drop=0.8, n_restart = 9, decay = 1.0, seed = 5):
    
        self.problem.update_state_db(self.DB[self.map[:,0],:,:])
        
        error = 999.9
        k = 0 
        
        for foo in callbacks:
            foo(k, self.solver_inner)
            
        while (error > tolout and k < maxitout):
            
            self.solver_inner.solve(tolin, maxitin, p_drop, n_restart, decay, seed)[0]
            self.updater(self.solver_inner.get_state_mech_data(), k) 
            self.solver_inner.project_onto_data()
            
            
            seed = seed + 1 # arbitary, but to leads to reproduceable simulations
            error = self.distance_relative()**2.0 # indeed relative energy 
            k+=1
            print("\tOuter Iteration #%3d ,- Relative Error Energy (energy) = %13.6e"%(k, error))
        
            for foo in callbacks:
                foo(k, self.solver_inner)
        
        # Optional, just to adjust to the last DB update
        self.solver_inner.solve(tolin, maxitin, p_drop, n_restart, decay, seed)
        self.project_onto_data()
        
        return self.hist, k


    # it allows to control inner dropout solver call with custom parameters 
    # the only difference wrt to solve_dropout in the use of the modified solver_inner.project_onto_data_dropout
    def solve_dropout_modified_projection(self, tolout = 0.01, tolin = 0.1, maxitin = 10, maxitout = 10, callbacks = [],                      
                      p_drop=0.8, n_restart = 9, decay = 1.0, seed = 5):
    
        self.problem.update_state_db(self.DB[self.map[:,0],:,:])
        
        error = 999.9
        k = 0 
        
        for foo in callbacks:
            foo(k, self.solver_inner)
            
        while (error > tolout and k < maxitout):
            
            self.solver_inner.solve(tolin, maxitin, p_drop, n_restart, decay, seed)[0]
            self.updater(self.solver_inner.get_state_mech_data(), k) 
            self.solver_inner.project_onto_data_dropout(p_drop)
            
            seed = seed + 1 # arbitary, but to leads to reproduceable simulations
            error = self.distance_relative()**2.0 # indeed relative energy 
            k+=1
            print("\tOuter Iteration #%3d ,- Relative Error Energy (energy) = %13.6e"%(k, error))
        
            for foo in callbacks:
                foo(k, self.solver_inner)
        
        # Optional, just to adjust to the last DB update
        self.solver_inner.solve(tolin, maxitin, p_drop, n_restart, decay, seed)
        self.project_onto_data()
        
        return self.hist, k

    def get_sol(self):
        return self.solver_inner.problem.get_sol()
