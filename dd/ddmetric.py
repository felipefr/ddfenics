#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:44:55 2023

@author: ffiguere

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import numpy as np
import scipy
import copy
from sklearn.decomposition import PCA

import dolfinx, ufl
from dolfinx import fem
from petsc4py import PETSc
from ddfenicsx.utils.fetricks import L2norm_given_form, tensor2mandel
import ddfenicsx as dd

class DDMetric:
    def __init__(self, C = None, Sh = None, dx = None, ddmat = None,  comm = None,
                 block_structure = "standard", C_estimator_method = "eigendecomp"):
        
        self.Sh = Sh
        self.mesh = self.Sh.space.mesh
        self.block_structure = self.__standard_block_structure if block_structure=="standard" else block_structure
        self.C_estimator_method = C_estimator_method
        self.comm = comm if comm else self.mesh.comm

        if(type(self.Sh) is not type(None) and type(dx) is type(None)):
            self.dx = self.Sh.dxm
        else:
            self.dx = dx
        
        self.C = None
        if(not isinstance(C, type(None))): # if it's not None
            self.reset(C)
        else:
            self.reset(self.estimateC(ddmat.DB))
            
        if(type(self.Sh) != type(None)):
            self.set_fenics_metric()
                
        self.sh0 = self.Sh.scalar_space
        self.dist_func = fem.Function(self.sh0)
        
        
    def reset(self, C):
        self.C = C
        self.CC = self.block_structure(self.C)
        self.L = np.linalg.cholesky(self.CC) 
        self.Linv = np.linalg.inv(self.L)
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
    
    def transformLinv(self, state):
        return state @ self.Linv
    
    def normL(self,state):
        return np.linalg.norm(self.transformL(state))
    
    # Fenics-related functions
    def set_fenics_metric(self):
        n = self.C.shape[0]
        self.CC_fe = fem.Constant(self.mesh, self.CC.astype(PETSc.ScalarType))
        
        if(type(self.dx) == type(None) ):
            self.dx = ufl.Measure('dx', self.mesh)
        
        self.form_inner_prod = lambda dz: fem.form(ufl.inner(ufl.dot(self.CC_fe, dz), dz)*self.dx)
        self.form_energy = lambda dz: fem.form(ufl.inner(dz[0], dz[1])*self.dx) # removing constant 2*
        
        self.C_fe = fem.Constant(self.mesh, self.CC[:n,:n].astype(PETSc.ScalarType))
        self.Cinv_fe = fem.Constant(self.mesh, self.CC[n:2*n, n:2*n].astype(PETSc.ScalarType))
        
        
        # readded for retrocompatibility
        self.deps = fem.Function(self.Sh.space)
        self.dsig = fem.Function(self.Sh.space)
        self.a_metric = (ufl.inner(ufl.dot(self.C_fe, self.deps), self.deps)*self.dx + 
                         ufl.inner(ufl.dot(self.Cinv_fe, self.dsig), self.dsig)*self.dx)
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
    
    
    # removed for retrocompatibilty
    # def dist_fenics(self, z_mech, z_db):
    #     return np.sqrt( self.form_inner_prod(z_mech - z_db)) 
    
    
    # readded for retrocompatibility
    def dist_form_fenics(self, z_mech, z_db):
        n = self.C.shape[0]
        C_fe = fem.Constant(self.mesh, self.C.astype(PETSc.ScalarType))
        Cinv_fe = fem.Constant(self.mesh, self.CC[n:2*n,n:2*n].astype(PETSc.ScalarType)) # takes into account alpha
        
        deps = z_mech[0] - z_db[0]
        dsig = z_mech[1] - z_db[1]
        
        form = (ufl.inner(C_fe*deps, deps)*self.dx + 
                ufl.inner(Cinv_fe*dsig, dsig)*self.dx)
        
        return form      
        
    def norm_fenics(self, z): # receives a list of DDFunction
        return L2norm_given_form(fem.form(self.form_inner_prod(z.as_vector())), self.comm)    
    
    def dist_energy_fenics(self, z_mech, z_db):
        return self.norm_energy_fenics(self.form_energy(z_mech.diff(z_db)))  

    # def dist_energy_fenics(self, z_mech, z_db):
    #     return np.sqrt(np.abs(self.form_energy(z_mech.diff(z_db))))  

    def norm_energy_fenics(self, z):
        return L2norm_given_form(self.form_energy(z), self.comm)  
    
    # def norm_energy_fenics(self, z):
    #     return np.sqrt(np.abs(self.form_energy(z)))  

    
    def distance_distTree(self, distTree):
        self.dist_func.x.array[:] = distTree.flatten()
        return L2norm_given_form(fem.form(self.dist_func**2*self.dx), self.comm) # L2 norm

   
    def estimateC(self, DB):
        C = self.estimateC_static(DB, self.C_estimator_method)
        self.C = self.check_positiveness(C, C_default = self.C)
        return self.C

    @staticmethod 
    def check_positiveness(C, C_default = None):
        import ddfenicsx.utils.nearestPD as nearestPD
        if(not nearestPD.isPD(C)):
            print("estimation is not PD --> using last default")
            print("eigenvalues:")
            print(np.linalg.eig(C)[0])
            if(type(C_default) != type(None)):
                return copy.deepcopy(C_default) # using the last one
            else:
                print("a default value for C should be provided")
                input()
        else:
            return C
        
    @staticmethod
    def estimateC_static(DB, method):

        dict_method = {'PCA': DDMetric.estimateC_PCA,
                       'eigendecomp': DDMetric.estimateC_eigen_decomposition,
                       'LSQ': DDMetric.estimateC_leastsquares,
                       'sylvester': DDMetric.estimateC_sylvester,
                       'sylvester_cov': DDMetric.estimateC_sylvester_cov,
                       'sylvester_cov_C': DDMetric.estimateC_sylvester_cov_C,
                       'sylvester_cov_Cinv': DDMetric.estimateC_sylvester_cov_Cinv,
                       'sylvester_C': DDMetric.estimateC_sylvester,
                       'sylvester_Cinv': DDMetric.estimateC_sylvester_Cinv,
                       'sylvester_C_isotropy': DDMetric.estimateC_sylvester_C_isotropy}

        return dict_method[method](DB)
        
    @staticmethod
    def estimateC_PCA(DB):
        strain_dim = DB.shape[-1]
        pca = PCA(n_components = strain_dim)
        pca.fit(DB.reshape((-1,2*strain_dim)))
        
        Cest = pca.components_[:,strain_dim:]@np.linalg.inv(pca.components_[:,:strain_dim])    
        
        C = 0.5*(Cest + Cest.T) 
    
        return C
    
    @staticmethod
    def estimateC_eigen_decomposition(DB):
        strain_dim = DB.shape[-1]
        
        Corr = DB.reshape((-1,2*strain_dim)).T@DB.reshape((-1,2*strain_dim))
        sig, U = np.linalg.eigh(Corr)
    
        asort = np.argsort(sig)
        sig = sig[asort[::-1]]
        U = U[:,asort[::-1]]
        
        Cest = U[strain_dim:2*strain_dim,:strain_dim]@np.linalg.inv(U[:strain_dim,:strain_dim])    
        
        C = 0.5*(Cest + Cest.T) 
        
        return C
    
    @staticmethod
    def estimateC_leastsquares(DB):
        Corr_EE = DB[:,0,:].T @ DB[:,0,:]
        Corr_SE = DB[:,1,:].T @ DB[:,0,:]
        Corr_EE = 0.5*(Corr_EE + Corr_EE.T)
        Corr_SE = 0.5*(Corr_SE + Corr_SE.T)
        C = Corr_SE@np.linalg.inv(Corr_EE)
        C = 0.5*(C + C.T) 
    
        return C
    
    # Sylvester equation is found when performing a leastsquares imposing symmetry for C
    @staticmethod
    def estimateC_sylvester_C(DB):
        Corr_EE = DB[:,0,:].T @ DB[:,0,:]
        Corr_SE = DB[:,1,:].T @ DB[:,0,:]
        
        # AX + BX = Q ==> Corr_EE*C + C*Corr_EE = (Corr_SE + Corr_SE.T) 
        C = scipy.linalg.solve_sylvester(Corr_EE, Corr_EE, Corr_SE + Corr_SE.T)

        return C


    @staticmethod
    def estimateC_sylvester_C_isotropy(DB):
        Corr_EE = DB[:,0,:].T @ DB[:,0,:]
        Corr_SE = DB[:,1,:].T @ DB[:,0,:]
        
        # AX + BX = Q ==> Corr_EE*C + C*Corr_EE = (Corr_SE + Corr_SE.T) 
        C = scipy.linalg.solve_sylvester(Corr_EE, Corr_EE, Corr_SE + Corr_SE.T)

        n = C.shape[0]
        Id_mandel = tensor2mandel(np.eye(int(n/2)))
        Basis = [ np.outer(Id_mandel, Id_mandel).flatten(), np.eye(n).flatten()]
        A = np.array([[np.dot(bi,bj) for bj in Basis] for bi in Basis])
        b = np.array([np.dot(C.flatten(), bi) for bi in Basis])
        x = np.linalg.solve(A,b)


        return (x[0]*Basis[0] +  x[1]*Basis[1]).reshape((n,n))

    # Sylvester equation is found when performing a leastsquares imposing symmetry for Cinv
    @staticmethod
    def estimateC_sylvester_Cinv(DB):
        Corr_SS = DB[:,1,:].T @ DB[:,1,:]
        Corr_ES = DB[:,0,:].T @ DB[:,1,:]
        
        # AX + BX = Q ==> Corr_SS*Cinv + Cinv*Corr_SS = (Corr_ES + Corr_ES.T) 
        Cinv = scipy.linalg.solve_sylvester(Corr_SS, Corr_SS, Corr_ES + Corr_ES.T)

        return np.linalg.inv(Cinv)

    # Average sylvester for C and Cinv
    @staticmethod
    def estimateC_sylvester(DB):
        return 0.5*( DDMetric.estimateC_sylvester_C(DB) + DDMetric.estimateC_sylvester_Cinv(DB))


    # Sylvester equation is found when performing a leastsquares imposing symmetry for C
    @staticmethod
    def estimateC_sylvester_cov(DB):
        return 0.5*( DDMetric.estimateC_sylvester_cov_C(DB) + DDMetric.estimateC_sylvester_cov_Cinv(DB))

    def estimateC_sylvester_cov_C(DB):
        n_strain = DB.shape[-1]
        cov = np.cov(DB[:,0,:].T, DB[:,1,:].T)
        Corr_EE = cov[0:n_strain, 0:n_strain]
        Corr_SE = cov[n_strain : 2*n_strain, 0:n_strain]
        
        # AX + BX = Q ==> Corr_EE*C + C*Corr_EE = (Corr_SE + Corr_SE.T) 
        C = scipy.linalg.solve_sylvester(Corr_EE, Corr_EE, Corr_SE + Corr_SE.T)

        return C
    
    def estimateC_sylvester_cov_Cinv(DB):
        n_strain = DB.shape[-1]
        cov = np.cov(DB[:,0,:].T, DB[:,1,:].T)
        Corr_SE = cov[n_strain : 2*n_strain, 0:n_strain]
        Corr_SS = cov[n_strain : 2*n_strain, n_strain : 2*n_strain]
        
        # AX + BX = Q ==> Corr_EE*C + C*Corr_EE = (Corr_SE + Corr_SE.T) 
        Cinv = scipy.linalg.solve_sylvester(Corr_SS, Corr_SS, Corr_SE + Corr_SE.T)

        return np.linalg.inv(Cinv)


    @staticmethod
    def __standard_block_structure(C):
        n = C.shape[0]
        CC = np.zeros((2*n,2*n))
        CC[0:n,0:n] = C
        CC[n:2*n,n:2*n] = np.linalg.inv(C)
        
        return CC
        
