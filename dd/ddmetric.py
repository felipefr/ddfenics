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
import dolfin as df
import scipy
import fetricks as ft
import copy

from sklearn.decomposition import PCA

class DDMetric:
    def __init__(self, C = None, V = None, dx = None, ddmat = None,
                 block_structure = "standard", C_estimator_method = "eigendecomp"):
        
        self.V = V
        self.block_structure = self.__standard_block_structure if block_structure=="standard" else block_structure
        self.C_estimator_method = C_estimator_method

        if(type(self.V) is not type(None) and type(dx) is type(None)):
            self.dx = self.V.dxm
        else:
            self.dx = dx
        
        self.C = None
        if(not isinstance(C, type(None))): # if it's not None
            self.reset(C)
        else:
            self.reset(self.estimateC(ddmat.DB))
            
        if(type(self.V) != type(None)):
            self.set_fenics_metric()
                
        self.sh0 = self.V.get_scalar_space()
        self.dist_func = df.Function(self.sh0)
        
        
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
        self.CC_fe = df.Constant(self.CC)
        
        if(type(self.dx) == type(None) ):
            self.dx = df.Measure('dx', self.V.mesh())
        
        self.form_inner_prod = lambda dz: df.assemble(df.inner(df.dot(self.CC_fe, dz), dz)*self.dx)
        self.form_energy = lambda dz: df.assemble(2.0*df.inner(dz[0], dz[1])*self.dx)
        
        self.C_fe = df.Constant(self.CC[:n,:n])
        self.Cinv_fe = df.Constant(self.CC[n:2*n, n:2*n])
        
        
        # readded for retrocompatibility
        self.deps = df.Function(self.V)
        self.dsig = df.Function(self.V)
        self.a_metric = (df.inner(df.dot(self.C_fe, self.deps), self.deps)*self.dx + 
                         df.inner(df.dot(self.Cinv_fe, self.dsig), self.dsig)*self.dx)
                         # df.inner(self.dsig, self.deps)*self.dx)

    # readded for retrocompatibility
    def dist_fenics(self, z_mech, z_db = None):
        if(type(z_db) == type(None) ):
            self.deps.assign(z_mech[0])
            self.dsig.assign(z_mech[1])
        else:
            self.deps.assign(z_mech[0] - z_db[0])
            self.dsig.assign(z_mech[1] - z_db[1])
            
        return np.sqrt( df.assemble(self.a_metric) ) 
    
    # removed for retrocompatibilty
    # def dist_fenics(self, z_mech, z_db):
    #     return np.sqrt( self.form_inner_prod(z_mech - z_db)) 
    
    
    # readded for retrocompatibility
    def dist_form_fenics(self, z_mech, z_db):
        n = self.C.shape[0]
        C_fe = df.Constant(self.C)
        Cinv_fe = df.Constant(self.CC[n:2*n,n:2*n]) # takes into account alpha
        
        deps = z_mech[0] - z_db[0]
        dsig = z_mech[1] - z_db[1]
        
        # form = (df.inner(C_fe*deps, deps)*self.dx + 
        #         df.inner(Cinv_fe*dsig, dsig)*self.dx
        #         + df.inner(deps, dsig)*self.dx)
        
        form = (df.inner(C_fe*deps, deps)*self.dx + 
                df.inner(Cinv_fe*dsig, dsig)*self.dx)
            
        return form      
        

    def norm_fenics(self, z): # receives a list of DDFunction
        return np.sqrt( self.form_inner_prod(z.as_vector()) )     

    def dist_energy_fenics(self, z_mech, z_db):
        return np.sqrt(np.abs(self.form_energy(z_mech.diff(z_db))))  

    def norm_energy_fenics(self, z):
        return np.sqrt(np.abs(self.form_energy(z)))  

    
    def distance_distTree(self, distTree):
        self.dist_func.vector().set_local(distTree)
        return np.sqrt(df.assemble((self.dist_func**2)*self.dx)) # L2 norm

    
    def estimateC(self, DB):
        method = self.C_estimator_method
        
        if(method == 'PCA'):
            C = self.estimateC_PCA(DB)
        elif(method == 'eigendecomp'):
            C = self.estimateC_eigen_decomposition(DB)
        elif(method == 'LSQ'):
            C = self.estimateC_leastsquares(DB)
        elif(method == 'sylvester'):
            C = self.estimateC_sylvester(DB)
        elif(method == 'sylvester_C'):
            C = self.estimateC_sylvester_C(DB)
        elif(method == 'sylvester_Cinv'):
            C = self.estimateC_sylvester_Cinv(DB)
        elif(method == 'sylvester_C_isotropy'):
            C = self.estimateC_sylvester_C_isotropy(DB)
        else:
            print("choose a appropriate C estimation method: {0} does not exist".format(method))
            input()
        
        import ddfenics.utils.nearestPD as nearestPD
        if(not nearestPD.isPD(C)):
            print("estimation is not PD --> using last default")
            print("eigenvalues:")
            print(np.linalg.eig(C)[0])
            if(self.C):
                return copy.deepcopy(self.C) # using the last one
            else:
                print("a default value for C should be provided")
                input()
        else:
            return C
        
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
        Id_mandel = ft.conv2d.Id_mandel_np
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


    @staticmethod
    def __standard_block_structure(C):
        n = C.shape[0]
        CC = np.zeros((2*n,2*n))
        CC[0:n,0:n] = C
        CC[n:2*n,n:2*n] = np.linalg.inv(C)
        
        return CC
        
