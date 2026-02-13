#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:54:38 2024

@author: felipe
"""

import numpy as np
import scipy

import dolfinx, ufl
from dolfinx import fem
from petsc4py import PETSc
from ddfenicsx.utils.fetricks import L2norm_given_form
import ddfenicsx as dd

class DDMetric:
    def __init__(self, C = None, V = None, dx = None, ddmat = None, comm = None,
                 block_structure = "standard", C_estimator_method = "eigendecomp"):
        
        self.V = V
        self.mesh = self.V.space.mesh
        self.block_structure = self.__standard_block_structure if block_structure=="standard" else block_structure
        self.C_estimator_method = dd.get_estimate_C_method(C_estimator_method)
        self.comm = comm if comm else self.mesh.comm

        if(type(self.V.space) is not type(None) and type(dx) is type(None)):
            self.dx = self.V.dxm
        else:
            self.dx = dx
        
        self.C = None
        if(not isinstance(C, type(None))): # if it's not None
            self.reset(C)
        else:
            self.reset(self.estimateC(ddmat.DB))
            
        if(type(self.V.space) != type(None)):
            self.set_fenics_metric()
                
        self.sh0 = self.V.scalar_space
        self.dist_func = fem.Function(self.sh0)
        
        
    def reset(self, C):
        self.C = C
        self.CC = self.block_structure(self.C)
        self.L = np.linalg.cholesky(self.CC) 
        self.set_fenics_metric()
        
    # Distance functions related (numpy based)
    def dist(self,state1,state2):
        return self.norm(state1-state2)
    
    def dist_sqr(self,state1,state2):
        return np.sqrt(self.dist(state1,state2))
    
    def norm(self,state):
        return np.dot(state,np.dot(self.CC,state))
    
    def transformL(self, state):
        return state @ self.L
    
    def normL(self,state):
        return np.linalg.norm(self.transformL(state))
    
    # Fenics-related functions
    def set_fenics_metric(self):
        n = self.C.shape[0]
        self.CC_fe = fem.Constant(self.mesh, self.CC.astype(PETSc.ScalarType))
        
        if(type(self.dx) == type(None) ):
            self.dx = ufl.Measure('dx', self.mesh)
        
        self.form_inner_prod = lambda dz: ufl.inner(ufl.dot(self.CC_fe, dz), dz)*self.dx
        self.form_energy = lambda dz: 2.0*ufl.inner(dz[0], dz[1])*self.dx
        
        self.C_fe = fem.Constant(self.mesh, self.CC[:n,:n].astype(PETSc.ScalarType))
        self.Cinv_fe = fem.Constant(self.mesh, self.CC[n:2*n, n:2*n].astype(PETSc.ScalarType))
        
        
        # readded for retrocompatibility
        self.deps = fem.Function(self.V.space)
        self.dsig = fem.Function(self.V.space)
        self.a_metric =  (ufl.inner(ufl.dot(self.C_fe, self.deps), self.deps)*self.dx + 
                         ufl.inner(ufl.dot(self.Cinv_fe, self.dsig), self.dsig)*self.dx)
    
        self.a_metric = fem.form(self.a_metric)
                         # ufl.inner(self.dsig, self.deps)*self.dx)

    # re-added for retrocompatibility
    def dist_fenics(self, z_mech, z_db = None):
        if(type(z_db) == type(None) ):
            self.deps.x.array[:] = z_mech[0].x.array[:]
            self.dsig.x.array[:] = z_mech[1].x.array[:]
        else:
            self.deps.x.array[:] = z_mech[0].x.array[:] - z_db[0].x.array[:]
            self.dsig.x.array[:] = z_mech[1].x.array[:] - z_db[1].x.array[:]
            
        return L2norm_given_form(self.a_metric, self.comm)
    
    # re-added for retrocompatibility
    def dist_form_fenics(self, z_mech, z_db):
        n = self.C.shape[0]
        C_fe = fem.Constant(self.mesh, self.C.astype(PETSc.ScalarType))
        Cinv_fe = fem.Constant(self.mesh, self.CC[n:2*n,n:2*n].astype(PETSc.ScalarType)) # takes into account alpha
        
        deps = z_mech[0] - z_db[0]
        dsig = z_mech[1] - z_db[1]
        
        form = (ufl.inner(C_fe*deps, deps)*self.dx + 
                ufl.inner(Cinv_fe*dsig, dsig)*self.dx)
        
        form = fem.form(form)
        return form      
        

    def norm_fenics(self, z): # receives a list of DDFunction
        return L2norm_given_form(fem.form(self.form_inner_prod(z.as_vector())), self.comm)     

    # def dist_energy_fenics(self, z_mech, z_db):
    #     return np.sqrt(np.abs(self.form_energy(z_mech.diff(z_db))))  
    
    
    def norm_energy_fenics(self, z):
        return L2norm_given_form(fem.form(self.form_energy(z)), self.comm)  

    
    def distance_distTree(self, distTree):
        self.dist_func.x.array[:] = distTree.flatten()
        return L2norm_given_form(fem.form(self.dist_func**2*self.dx), self.comm) # L2 norm
        
    def estimateC(self, DB):
        C = self.C_estimator_method(DB)
        self.C = dd.check_positiveness(C, C_default = self.C)
        return self.C

    @staticmethod
    def __standard_block_structure(C):
        n = C.shape[0]
        CC = np.zeros((2*n,2*n))
        CC[0:n,0:n] = C
        CC[n:2*n,n:2*n] = np.linalg.inv(C)
        
        return CC