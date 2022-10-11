#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:44:00 2022

@author: felipe
"""


# This uncoupled formulation keeps the same variables as the classical formulation

import numpy as np
import dolfin as df
import ufl

from ddfenics.dd.ddproblem_abstract import DDProblemAbstract

import copy

# This is based on the class DDbilinear
class DDProblem(DDProblemAbstract):
    
    def __init__(self, a, L, sol, bcs,  
                 form_compiler_parameters = {}, bcsPF = [], metric = None):
    
        self.metric = metric
        self.omega = -0.5 # relaxation value
        bcs_eta = self.__convertBCs_eta(bcs)
        
        super().__init__(a, L, sol, bcs, form_compiler_parameters, bcsPF, bcs_eta = bcs_eta)
        
            
    def __convertBCs_eta(self, bcs):
    
        bcs_eta = [df.DirichletBC(b) for b in bcs]
        for b in bcs_eta:
            c = df.Constant(tuple(self.omega*b.value().values()))
            b.set_value(c)
            
        return bcs_eta
    
    def createSubproblems(self):
        # it might be rewritten to exclude the dependence on eta for LDBcomp
        f_comp = self.a.action_strain(self.z_db[0])      
        f_bal = self.a.action_stress(self.z_db[1]) - self.L
        
        f_comp_gen = f_comp + self.omega*f_bal
        f_bal_gen = self.omega*f_comp + f_bal
        
        self.problem_comp = df.LinearVariationalProblem(self.a.bilinear, f_comp_gen, self.u, self.bcs)
        self.problem_bal = df.LinearVariationalProblem(self.a.bilinear, f_bal_gen, self.eta, self.bcs_eta)

        
    def update_state_mech(self):
        w = self.omega
        
        state_update = ( self.ddmat.grad(self.u), 
                        self.z_db[1] + self.ddmat.sigC_e(w*self.z_db[0] - self.a.ddmat.grad(self.eta)) ) 
        
    
        if(type(self.z_mech) == type([])):
            for i, z_i in enumerate(self.z_mech):
                z_i.update(state_update[i])
                
        else: # all state variables are in a same object
            self.z_mech.update(df.as_tensor(state_update))
