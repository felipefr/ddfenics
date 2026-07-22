#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:53:08 2023

@author: ffiguere
"""

import numpy as np

import ddfenicsx as dd
import fetricksx as ft
import ufl
from dolfinx import fem
import basix
import dolfinx.fem.petsc

from inspect import signature # check the number of arguments of a function
from ddfenicsx.utils.fetricks import symgrad_mandel, tensor2mandel

# with line search
# petsc_options = {
#     "snes_type": "newtonls",
#     "snes_linesearch_type": "bt",    # backtracking
#     "snes_linesearch_damping": 1.0,
#     "snes_linesearch_minlambda": 1e-12,
#     "snes_linesearch_rtol": 1e-3,
#     "snes_linesearch_max_it": 30,
#     "snes_linesearch_norms": True,
#     #"snes_monitor": None,
#     #"snes_atol": 1e-10,
#     "snes_rtol": 1e-8,
#     "snes_stol": 0.0,
#     "ksp_type": "preonly",
#     "pc_type": "lu",
#     "pc_factor_mat_solver_type": "mumps"
# }

            
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_max_it": 10,
    "snes_monitor_cancel": None,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-10,
    "snes_stol": 1e-10,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
            


# delta E(u;v) : Directional derivative Green-Lagrange from displacements (Mandel)
def get_deltaGL_mandel(u, v):
    return symgrad_mandel(v) + tensor2mandel(ufl.grad(u).T*ufl.grad(v))

# Green-Lagrange from displacements (Mandel)
# Note that E(u) = delta E(0.5*u; u) = eps(u) + 0.5*(grad(u).T*grad(u))
def get_GL_mandel(u):
    return get_deltaGL_mandel(0.5*u, u)


# Function aiming to pass the boundary conditions into constant functions
def get_constant_function(uD_, U):
    u = fem.Function(U)
    if u.ufl_shape == ():  # scalar field
        u.x.array[:] = uD_
    else:
        gdim = u.ufl_shape[0]
        value = np.zeros_like(u.x.array).reshape((-1,gdim))
        for i in range(gdim):
            value[:,i] = uD_[i]        
        u.x.array[:] = value.flatten()
    
    return u


class DDProblemFiniteStrain(dd.DDProblemBase):

        
    def __init__(self, spaces, L, bcs, metric,  
                 form_compiler_parameters = {}, bcsPF = [], 
                 is_accelerated=True, maxits_newton = 10):
    
        self.maxits_newton = maxits_newton 
        
        super().__init__(spaces, L, bcs, metric,  
                     form_compiler_parameters, bcsPF, is_accelerated)
        
        
    def create_problem(self):
        
        self.Wh = self.__create_mixed_space(self.Uh) # with function assigners
        self.w = fem.Function(self.Wh)

        bcs_W = self.__create_bcs(self.bcs, self.Wh)
        
        (u, eta) = ufl.split(self.w)
        
        E = get_GL_mandel
        dE = get_deltaGL_mandel
        
        # E(u), S(u, eta)
        z = [E(u), self.z_db[1] + self.C*dE(u,eta)] # z_mech symbolic
        
        dist2 = self.metric.dist_form_fenics(z, self.z_db)
        
        print(len(signature(self.L).parameters))
        if(len(signature(self.L).parameters)==1):
            Ptot = ufl.inner(z[1] , dE(u,eta))*self.dx - self.L(eta)
        else:
            Ptot = ufl.inner(z[1] , dE(u,eta))*self.dx - self.L(u, eta)
            
        Lag = 0.5*dist2 - Ptot  
        
        res = ufl.derivative(Lag, self.w) # R_u(u,eta; delta u) , R_eta(u,eta; delta eta)
        Jac = ufl.derivative(res, self.w)
        
        nls_solver = fem.petsc.NonlinearProblem(res, self.w, bcs = bcs_W, J = Jac, 
                                                 petsc_options = petsc_options, 
                                                 petsc_options_prefix='dd_finite_strain')
        
    
        return nls_solver, z


    def get_sol(self):
        
        if(self.is_accelerated):        
            u = self.w.sub(0).collapse()
            eta = self.w.sub(1).collapse()
            u.x.array[:] = u.x.array[:] + 0.5*eta.x.array[:]
        else:
            u = self.w.sub(0).collapse()
            
        u.name = 'u'
        
        return {"state_mech" : self.z_mech ,
                "state_db": self.z_db ,
                "u" : u,
                "w" : self.w}
    
    
    def __create_mixed_space(self, Uh):
        e = Uh.element.basix_element 
        degree = e.degree
        tdim = Uh.mesh.topology.dim
        cell_type = e.cell_type
        family = e.family
        Ue = basix.ufl.element(family, cell_type, degree, shape=(tdim,))  
        We =  basix.ufl.mixed_element([Ue, Ue])
        Wh = fem.functionspace(Uh.mesh, We)  
        return Wh
    

    def __create_bcs(self, bcs_constructor, Wh):
        # bcs_u = [fem.dirichletbc(b(Wh.sub(0))) for b in bcs_constructor]
        # bcs_eta = [fem.dirichletbc(b(Wh.sub(1))) for b in bcs_constructor]
        # [b.homogenize() for b in bcs_eta]
        
        mesh = Wh.mesh
        Vu = Wh.sub(0)
        Veta = Wh.sub(1)
        
        Vu_, _ = Vu.collapse()
        Veta_, _ = Veta.collapse()

        # zero_vec_u = get_constant_function(np.zeros(3), Vu_)
        # zero_vec_eta = get_constant_function(np.zeros(3), Veta_)
        
        zero_vec_u = fem.Function(Vu_) 
        zero_vec_eta = fem.Function(Veta_)
        
        facets = mesh.facet_tags.find(mesh.physical["Gamma_bottom"].tag)
        dofs_u = fem.locate_dofs_topological((Wh, Vu_), mesh.physical["Gamma_bottom"].dim, facets)
        bcs_u = [ fem.dirichletbc(zero_vec_u, dofs_u, Vu) ]
        
        dofs_eta = fem.locate_dofs_topological((Wh, Veta_), mesh.physical["Gamma_bottom"].dim, facets)
        bcs_eta = [ fem.dirichletbc(zero_vec_eta, dofs_eta, Veta) ]
        
        return bcs_u + bcs_eta
    