#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 22:58:13 2024
@author: felipe

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2026, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or
<f.rocha.felipe@gmail.com>

For more info check : https://hal.science/GEM/hal-04611046v1
Felipe Rocha, Thiago Quinelato, Laurent Stainier. 
Some experiences in mixed finite element formulations for (model-free) data-driven computational mechanics. 
16ème Colloque National en Calcul de Structures (CSMA 2024), CNRS; CSMA; ENS Paris-Saclay; CentraleSupélec, May 2024, Hyères, France. 
⟨hal-04611046⟩
"""

import ufl
import numpy as np
from dolfinx import fem
import ddfenicsx as dd
import fetricksx as ft
import ufl
from dolfinx.fem.petsc import LinearProblem

class DDProblemPoissonHdiv(dd.DDProblemBase):
    def __init__(self, spaces, L, bcs, metric, Hdiv_space_param = {'family': 'BDM', 'order': 1},
                 form_compiler_parameters = {}, bcsPF = [], is_accelerated= True):
    
        self.Hdiv_space_param = Hdiv_space_param
        self.grad = ufl.grad
        
        super().__init__(spaces, L, bcs, metric,  
                     form_compiler_parameters, bcsPF, is_accelerated = is_accelerated)
        
        
    def create_problem(self):
        
        self.u = fem.Function(self.Uh)
        v = ufl.TestFunction(self.Uh)
        u_ = ufl.TrialFunction(self.Uh)

        k = self.Hdiv_space_param['order']
        self.Wh = ft.mixed_functionspace(self.Uh.mesh, [(self.Hdiv_space_param['family'], k), ('DG', k-1)])
        self.w = fem.Function(self.Wh)

        dw = ufl.TestFunction(self.Wh)
        w_ = ufl.TrialFunction(self.Wh)

        tau, xsi = ufl.split(dw)
        q, eta = ufl.split(w_)
        
        bcs_u, bcs_w = self.__create_bcs()

        a_comp = ufl.inner(self.C*self.grad(u_) , self.grad(v))*self.dx    
        f_comp = -ufl.inner(self.C*self.z_db[0] , self.grad(v))*self.dx     
        
        a_bal = ufl.inner(self.Cinv*q, tau)*self.dx - ufl.inner(ufl.div(tau), eta)*self.dx - ufl.inner(ufl.div(q), xsi)*self.dx  
        f_bal = self.L(xsi,tau) + ufl.inner(self.Cinv*self.z_db[1], tau)*self.dx
        
        blocksolver = ft.block_solver([a_comp, a_bal], [f_comp, f_bal], 
                                      [self.u, self.w], [bcs_u, bcs_w]) 
    
        qh = self.w.sub(0)

        z = [-self.grad(self.u), qh] # z_mech symbolic (to be projected on the quadradures)

        return blocksolver, z
    

    # [bcs_D_u, bcs_N_q] : Note that dirichlet bcs on eta are natural, but they are vanishing in this case
    def __create_bcs(self):
        bcs_u = [b(self.Uh) for b in self.bcs[0]]
        bcs_q = [b(self.Wh.sub(0)) for b in self.bcs[1]]
        return bcs_u, bcs_q
    