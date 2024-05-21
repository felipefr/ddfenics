#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 00:20:39 2023

@author: ffiguere

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""

import dolfin as df
from fetricks.fenics.la.wrapper_solvers import BlockSolver
import ddfenics as dd

class DDProblemInfinitesimalStrain(dd.DDProblemBase):
    def __init__(self, spaces, grad, L, bcs, metric,  
                 form_compiler_parameters = {}, bcsPF = [], is_accelerated = True):
    
        super().__init__(spaces, grad, L, bcs, metric,  
                     form_compiler_parameters, bcsPF, is_accelerated)
        
        
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
    