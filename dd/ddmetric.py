#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:44:55 2023

@author: ffiguere
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:18:12 2022

@author: felipe
"""

import numpy as np
import dolfin as df

from sklearn.decomposition import PCA

class DDMetric:
    def __init__(self, C = None, block_structure = "standard", V = None, dx = None, ddmat = None):
        self.V = V
        self.block_structure = block_structure
        
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
        if(self.block_structure == 'standard'):    
            self.CC = self.__standard_block_structure(self.C)
        else:
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
      
    def dist_fenics(self, z_mech, z_db):
        return np.sqrt( self.form_inner_prod(z_mech - z_db)) 
    
    def norm_fenics(self, z): # receives a list of DDFunction
        return np.sqrt( self.form_inner_prod(z.as_vector()) )     

    def dist_energy_fenics(self, z_mech, z_db):
        return np.sqrt(np.abs(self.form_energy(z_mech.diff(z_db))))  

    def norm_energy_fenics(self, z):
        return np.sqrt(np.abs(self.form_energy(z)))  

    
    def distance_distTree(self, distTree):
        self.dist_func.vector().set_local(distTree)
        return np.sqrt(df.assemble((self.dist_func**2)*self.dx)) # L2 norm

    
    def estimateC(self, DB, method = 'eigendecomp'):
        if(method == 'PCA'):
            return self.__estimateC_PCA(DB)
        elif(method == 'eigendecomp'):
            return self.__estimateC_eigen_decomposition(DB)


    def __estimateC_PCA(self, DB):
        strain_dim = DB.shape[-1]
        pca = PCA(n_components = strain_dim)
        pca.fit(DB.reshape((-1,2*strain_dim)))
        
        Cest = pca.components_[:,strain_dim:]@np.linalg.inv(pca.components_[:,:strain_dim])    
        
        C = 0.5*(Cest + Cest.T) 
    
        import ddfenics.utils.nearestPD as nearestPD
        if(not nearestPD.isPD(C)):
            return self.C # using the last one
        
        return C
    
    def __estimateC_eigen_decomposition(self, DB):
        strain_dim = DB.shape[-1]
        
        Corr = DB.reshape((-1,2*strain_dim)).T@DB.reshape((-1,2*strain_dim))
        sig, U = np.linalg.eigh(Corr)
    
        asort = np.argsort(sig)
        sig = sig[asort[::-1]]
        U = U[:,asort[::-1]]
        
        Cest = U[strain_dim:2*strain_dim,:strain_dim]@np.linalg.inv(U[:strain_dim,:strain_dim])    
        
        C = 0.5*(Cest + Cest.T) 
    
        import ddfenics.utils.nearestPD as nearestPD
        if(not nearestPD.isPD(C)):
            return self.C # using the last one
        
        return C
    
    @staticmethod
    def __standard_block_structure(C):
        n = C.shape[0]
        CC = np.zeros((2*n,2*n))
        CC[0:n,0:n] = C
        CC[n:2*n,n:2*n] =np.linalg.inv(C)
        
        return CC
        