#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:03:38 2023

@author: ffiguere
"""

import numpy as np
import dolfin as df
import ufl

from ddfenics.dd.ddfunction import DDFunction

class DDProblemBase:
    
    def __init__(self, spaces, grad, L, bcs, metric,  
                 form_compiler_parameters = {}, bcsPF = []):
    
        self.Uh, self.Sh = spaces 
        self.metric = metric
        
        self.z_mech = [DDFunction(self.Sh), DDFunction(self.Sh)] 
        self.z_db = [DDFunction(self.Sh), DDFunction(self.Sh)] 
        self.strain_dim = self.Sh.num_sub_spaces()
        
        self.L = L
        self.grad = grad
        self.dx = self.Sh.dxm
        self.ds = df.Measure('ds', domain = self.Uh.mesh())
        self.bcs = bcs
        self.bcsPF = bcsPF
        
        self.C = self.metric.C_fe
        self.Cinv = self.metric.Cinv_fe
    
        self.solver, self.z = self.create_problem()
     
    # Typically, you should instanciate self.u, return the solver, and the symbolic update for z    
    def create_problem(self):
        pass         
        # return solver, z
    

    def get_sol(self):
        return {"state_mech" : self.z_mech ,
                "state_db": self.z_db ,
                "u" : self.u }
    
    def solve(self):
        self.solver.solve() 
        
    def accelerated_update(self, z_mech, z_db):

        return [ z_mech[0] + 0.5*self.Cinv*(z_mech[1] - z_db[1]), 
                 z_mech[1] + 0.5*self.C*(z_mech[0] - z_db[0])]
        
    def update_state_mech(self):
        state_update = self.accelerated_update(self.z, self.z_db)
        # state_update = self.z

        for i, z_i in enumerate(self.z_mech):
            z_i.update(state_update[i])
            
    def update_state_db(self, state_db):
        for i, z_i in enumerate(self.z_db):
            z_i.update(state_db[:,i,:].reshape((-1,self.strain_dim)))
            

    def get_state_mech_data(self):
        return np.concatenate(tuple([z_i.data() for i, z_i in enumerate(self.z_mech)]) , axis = 1)
