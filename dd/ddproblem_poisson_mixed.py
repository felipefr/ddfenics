#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  11 00:20:39 2023

@author: ffiguere
"""

import numpy as np
import dolfin as df
import ufl

from fetricks.fenics.la.wrapper_solvers import BlockSolverIndependent
from ddfenics.dd.ddfunction import DDFunction
from ddfenics.dd.ddproblem_base import DDProblemBase
import fetricks as ft

import copy

class DDProblemPoissonMixed(DDProblemBase):
    def __init__(self, spaces, grad, L, bcs, metric,  
                 form_compiler_parameters = {}, bcsPF = []):
    
        super().__init__(spaces, grad, L, bcs, metric,  
                     form_compiler_parameters, bcsPF)
        
        
    def create_problem(self):
        
        self.u = df.Function(self.Uh)
        v = df.TestFunction(self.Uh)
        u_ = df.TrialFunction(self.Uh)

        self.Wh = self.__create_mixed_space(self.Uh)
        self.w = df.Function(self.Wh)

        dw = df.TestFunction(self.Wh)
        w_ = df.TrialFunction(self.Wh)

        tau, xsi = df.split(dw)
        q, eta = df.split(w_)
        
        bcs_u , bcs_w = self.bcs
        bcs_u_ = [b(self.Uh) for b in bcs_u]
        
        bcs_w = self.__create_bcs(bcs_w, bcs_u, self.Wh) # to do (should take into account neumann as essential)

        a_comp = df.inner(self.C*self.grad(u_) , self.grad(v))*self.dx    
        f_comp = df.inner(self.C*self.z_db[0] , self.grad(v))*self.dx     
        
        normal = df.FacetNormal(self.Uh.mesh())
        
        a_bal = df.inner(self.Cinv*q, tau)*self.dx + df.inner(df.div(tau), eta)*self.dx + df.inner(df.div(q), xsi)*self.dx  
        f_bal = (-self.L(xsi) + df.inner(self.Cinv*self.z_db[1], tau)*self.dx 
                 + df.inner(self.u, df.inner(tau, normal))*self.ds )# assuming eta vanishes on Gamma_D
        
        self.problem_comp = df.LinearVariationalProblem(a_comp, f_comp, self.u, bcs_u_)
        self.problem_bal = df.LinearVariationalProblem(a_bal, f_bal, self.w, bcs_w)
        
        blocksolver = BlockSolverIndependent([self.problem_comp, self.problem_bal]) 

        qh, dummy = df.split(self.w) # eta is dummy here

        z = [self.grad(self.u), qh] # z_mech symbolic (to be projected on the quadradures)

        return blocksolver, z
    

    def get_sol(self):
        return {"state_mech" : self.z_mech ,
                "state_db": self.z_db ,
                "u" : self.u }

    def __create_mixed_space(self,Uh):
        mesh = Uh.mesh()
        BDM_e = df.FiniteElement("BDM", mesh.ufl_cell(), 2)
        DG_e = df.FiniteElement("DG", mesh.ufl_cell(), 1)
        
        We = df.MixedElement(BDM_e, DG_e)
        Wh = df.FunctionSpace(mesh, We)
        
        # fa_W2UU = df.FunctionAssigner([Uh, Uh], Wh) # receiver, assigner
        # fa_UU2W = df.FunctionAssigner(Wh, [Uh, Uh]) # receiver, assigner
        
        return Wh
    

    def __create_bcs(self, bcs_w_constructor, bcs_u_constructor,  Wh):
        bcs_q = [b(Wh.sub(0)) for b in bcs_w_constructor]
    
        bcs_eta = [b(Wh.sub(1)) for b in bcs_u_constructor]
        [b.homogenize() for b in bcs_eta]
    
        return bcs_q + bcs_eta