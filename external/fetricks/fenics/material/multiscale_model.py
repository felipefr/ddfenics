#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 00:23:16 2022

@author: felipe
"""

import sys
import dolfin as df
import numpy as np
from .material_model import materialModel, materialModelExpression 

import fetricks as ft

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF


as_sym_tensor = lambda a: df.as_tensor( [ [ a[0], a[1], a[2]] , [a[1] , a[3], a[4]] , [a[2] , a[4], a[5]] ])
ind_sym_tensor = np.array([0, 1, 2, 4, 5, 8])

collect_stress = lambda m, e: np.array( [ m[i].getStress(e[i,:]) for i in range(len(m))] ).flatten()
collect_tangent = lambda m, e: np.array( [ m[i].getTangent(e[i,:]).flatten()[ind_sym_tensor] for i in range(len(m))] ).flatten()



class multiscaleModel(materialModel):
    
    def __init__(self, W, Wtan, dxm, micromodels):
        
        self.micromodels = micromodels
        self.mesh = W.mesh()
        self.__createInternalVariables(W, Wtan, dxm)

    def __createInternalVariables(self, W, Wtan, dxm):
        self.stress = df.Function(W)
        self.eps = df.Function(W)
        self.tangent = df.Function(Wtan)
        
        self.size_tan = Wtan.num_sub_spaces()
        self.size_strain = W.num_sub_spaces()
    
        self.ngauss = int(W.dim()/self.size_strain)
        
        self.projector_eps = ft.LocalProjector(W, dxm)
        
        self.Wdofmap = W.dofmap()
        self.Wtandofmap = Wtan.dofmap()
        
        
    def tangent_op(self, de):
        return df.dot(as_sym_tensor(self.tangent), de) 

    # def update(self, epsnew):
        
    #     self.projector_eps(epsnew ,  self.eps) 
        
    #     for m in self.micromodels:
    #         m.setUpdateFlag(False)
    
    #     strains = self.eps.vector().get_local()[:].reshape( (-1, self.size_strain) )
        
    #     self.stress.vector().set_local( collect_stress(self.micromodels, strains) ) 
    #     self.tangent.vector().set_local( collect_tangent(self.micromodels, strains) ) 

    
    # parallel version
    def update(self, epsnew):
        self.projector_eps(epsnew ,  self.eps) 
            
        for e in df.cells(self.mesh):
            # print("hello, I'm cell ", e.index(), e.global_index(), comm.Get_rank() )
            dofs = self.Wdofmap.cell_dofs(e.index())
            dofs_tan = self.Wtandofmap.cell_dofs(e.index())
            # print(dofs, dofs_tan )
            m = self.micromodels[e.index()]
            m.setUpdateFlag(False)
            self.stress.vector().vec()[dofs] = m.getStress(self.eps.vector().get_local()[dofs])
            self.tangent.vector().vec()[dofs_tan] = m.getTangent(self.eps.vector().get_local()[dofs]).flatten()[ind_sym_tensor]
            
            
            
            
class multiscaleModelExpression(materialModelExpression):
    
    def __init__(self, W, Wtan, dxm, micromodels):
        self.micromodels = micromodels
        super().__init__(W, Wtan, dxm)
    
    def pointwise_stress(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format        
        return self.micromodels[cell.index].getStress(e)
    
    def pointwise_tangent(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format
        return self.micromodels[cell.index].getTangent(e).flatten()[ind_sym_tensor]
    
    def tangent_op(self, de):
        return df.dot(as_sym_tensor(self.tangent), de) 
    
    def update(self, e):
        super().update(e)
        
        for m in self.micromodels:
            m.setUpdateFlag(False)
    