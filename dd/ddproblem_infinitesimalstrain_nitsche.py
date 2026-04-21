#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:59:42 2026

@author: frocha
"""

import ufl
from dolfinx import fem
import ddfenicsx as dd
from ddfenicsx.utils.fetricks import symgrad_mandel, BlockSolver, mandel2tensor
from .ddproblem_infinitesimalstrain import DDProblemInfinitesimalStrain

class DDProblemInfinitesimalStrainNitsche(DDProblemInfinitesimalStrain):
    def create_problem(self):
        gamma = 100*self.C.value[0,0]
        meshdata = self.bcs[0]
        bc_flag = self.bcs[1][0]
        uD = self.bcs[1][1]
        
        domain = meshdata.mesh
        facet_tags = meshdata.facet_tags
        physical = meshdata.physical_groups
        
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        dsD = ds(physical[bc_flag].tag)
        n = ufl.FacetNormal(domain)
        h = ufl.CellDiameter(domain)
        
        uD_ = fem.Constant(domain, uD)
        
        self.u = fem.Function(self.Uh, name = "u")
        self.eta = fem.Function(self.Uh, name = "eta")
        
        uh = ufl.TrialFunction(self.Uh)
        vh = ufl.TestFunction(self.Uh)
        
        sigma = lambda w : mandel2tensor(self.C*self.grad(w))
        
        a = ufl.inner(self.C*self.grad(uh), self.grad(vh))*self.dx

        f_comp = ufl.inner(self.C*self.z_db[0] , self.grad(vh))*self.dx     
        f_bal = self.L(vh)  - ufl.inner(self.z_db[1], self.grad(vh))*self.dx

        # Nitsche terms
        a_nit = gamma / h * ufl.inner(uh, vh) * dsD
        a_nit += - ufl.inner(ufl.dot(sigma(uh), n), vh) * dsD
        a_nit += - ufl.inner(ufl.dot(sigma(vh), n), uh) * dsD
        
        L_nit = gamma / h * ufl.inner(uD_, vh) * dsD
        L_nit += - ufl.inner(ufl.dot(sigma(vh), n), uD_) * dsD

        a += a_nit
        f_bal += L_nit
        f_comp += L_nit
        
        blocksolver = BlockSolver(a, [f_comp, f_bal], [self.u, self.eta], [[], []]) 
        z = [self.grad(self.u), self.z_db[1] + self.C*self.grad(self.eta)] # z_mech symbolic

        return blocksolver, z