#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  11 00:20:39 2023

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

class DDProblemPoisson(DDProblemBase):
    def __init__(self, spaces, grad, L, bcs, metric,  
                 form_compiler_parameters = {}, bcsPF = []):
    
        super().__init__(spaces, grad, L, bcs, metric,  
                     form_compiler_parameters, bcsPF)
        
        
    def create_problem(self):
        
        self.u = df.Function(self.Uh)
        self.eta = df.Function(self.Uh)

        bcs_eta = [df.DirichletBC(b) for b in self.bcs] # creates a copy
        [b.homogenize() for b in bcs_eta]
        
        uh = df.TrialFunction(self.Uh)
        vh = df.TestFunction(self.Uh)
        
        a = df.inner(self.C*self.grad(uh), self.grad(vh))*self.dx
        
        f_comp = df.inner(self.C*self.z_db[0] , self.grad(vh))*self.dx     
        f_bal = self.L(vh)  - df.inner(self.z_db[1], self.grad(vh))*self.dx
        
        self.problem_comp = df.LinearVariationalProblem(a, f_comp, self.u, self.bcs)
        self.problem_bal = df.LinearVariationalProblem(a, f_bal, self.eta, bcs_eta)
        
        blocksolver = BlockSolver([self.problem_comp, self.problem_bal]) 
        z = [self.grad(self.u), self.z_db[1] + self.C*self.grad(self.eta)] # z_mech symbolic

        return blocksolver, z
    

    def get_sol(self):
        return {"state_mech" : self.z_mech ,
                "state_db": self.z_db ,
                "u" : self.u }
    