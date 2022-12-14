#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 23:26:49 2023

@author: ffiguere
"""

import numpy as np
import dolfin as df
import ufl

from fetricks.fenics.la.wrapper_solvers import BlockSolver
from ddfenics.dd.ddfunction import DDFunction
from ddfenics.dd.ddproblem_base import DDProblemBase
import fetricks as ft

import copy

class DDProblemGeneric(DDProblemBase):
    def __init__(self, spaces, grad, a, L, bcs, metric,  
                 form_compiler_parameters = {}, bcsPF = []):
    
        self.a = a 
        super().__init__(spaces, a.grad, L, bcs, metric,  
                     form_compiler_parameters, bcsPF)
        
        
    def create_problem(self):
        
        self.u = df.Function(self.Uh)
        self.eta = df.Function(self.Uh)

        bcs_eta = [df.DirichletBC(b) for b in self.bcs] # creates a copy
        [b.homogenize() for b in bcs_eta]
        
        f_comp = self.a(self.z_db[0] , 'strain')     
        f_bal = self.L - self.a(self.z_db[1] , 'stress')
        
        self.problem_comp = df.LinearVariationalProblem(self.a.a_uv, f_comp, self.u, self.bcs)
        self.problem_bal = df.LinearVariationalProblem(self.a.a_uv, f_bal, self.eta, bcs_eta)
        
        blocksolver = BlockSolver([self.problem_comp, self.problem_bal]) 
        z = [self.grad(self.u), self.z_db[1] + self.C*self.grad(self.eta)] # z_mech symbolic

        return blocksolver, z
    

    def get_sol(self):
        return {"state_mech" : self.z_mech ,
                "state_db": self.z_db ,
                "u" : self.u }
    