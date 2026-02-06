#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  11 00:20:39 2023

@author: ffiguere

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""

import ufl
from dolfinx import fem
import ddfenicsx as dd
import fetricksx as ft


class DDProblemPoisson(dd.DDProblemBase):
    def __init__(self, spaces, L, bcs, metric,  
                 form_compiler_parameters = {}, bcsPF = [], is_accelerated=True):
        
        self.grad = ufl.grad
        
        super().__init__(spaces, L, bcs, metric,  
                     form_compiler_parameters, bcsPF, is_accelerated=True)
        
        
    def create_problem(self):
        
        self.u = fem.Function(self.Uh)
        self.eta = fem.Function(self.Uh)

        bcs_u, bcs_eta = self.bcs 
        
        uh = ufl.TrialFunction(self.Uh)
        vh = ufl.TestFunction(self.Uh)
        
        a = ufl.inner(self.C*self.grad(uh), self.grad(vh))*self.dx
        
        f_comp = ufl.inner(self.C*self.z_db[0] , self.grad(vh))*self.dx     
        f_bal = self.L(vh)  - ufl.inner(self.z_db[1], self.grad(vh))*self.dx

        blocksolver = ft.block_solver(a, [f_comp, f_bal], 
                                      [self.u, self.eta], [bcs_u, bcs_eta]) 
        
        z = [self.grad(self.u), self.z_db[1] + self.C*self.grad(self.eta)] # z_mech symbolic

        return blocksolver, z
    
    
