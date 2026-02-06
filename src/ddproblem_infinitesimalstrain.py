#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 2024

@author: ffiguere

This file is part of ddfenicsx, a FEniCsx-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2024, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or
<f.rocha.felipe@gmail.com>

"""

# known limitation: only working with homogeneous bcs

import ufl
from dolfinx import fem
import ddfenicsx as dd
import fetricksx as ft

class DDProblemInfinitesimalStrain(dd.DDProblemBase):
    def __init__(self, spaces, L, bcs, metric,  
                 form_compiler_parameters = {}, bcsPF = [], is_accelerated = True):
        
        self.grad = ft.symgrad_mandel
        super().__init__(spaces, L, bcs, metric,  
                     form_compiler_parameters, bcsPF, is_accelerated)
        
        
    def create_problem(self):
        
        self.u = fem.Function(self.Uh)
        self.eta = fem.Function(self.Uh)

        self.bcs_eta = self.bcs # this bcs should be always homogenous
        # bcs_eta = [fem.bcs.DirichletBC(b) for b in self.bcs] # creates a copy
        # [b.homogenize() for b in bcs_eta] # in the case tested its already homogeneous
        
        uh = ufl.TrialFunction(self.Uh)
        vh = ufl.TestFunction(self.Uh)
        
        a = ufl.inner(self.C*self.grad(uh), self.grad(vh))*self.dx
        
        f_comp = ufl.inner(self.C*self.z_db[0] , self.grad(vh))*self.dx     
        f_bal = self.L(vh)  - ufl.inner(self.z_db[1], self.grad(vh))*self.dx
                
        blocksolver = ft.BlockSolver(a, [f_comp, f_bal], [self.u, self.eta], [self.bcs, self.bcs_eta]) 
        z = [self.grad(self.u), self.z_db[1] + self.C*self.grad(self.eta)] # z_mech symbolic

        return blocksolver, z
    

    def get_sol(self):
        return {"state_mech" : self.z_mech ,
                "state_db": self.z_db ,
                "u" : self.u }
    