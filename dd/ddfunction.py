#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 20:06:23 2022

@author: felipe
"""

# from numba import jit
import dolfin as df 
import numpy as np
import ufl
from fetricks.fenics.la.wrapper_solvers import local_project_given_sol

typesForProjection = (ufl.tensors.ComponentTensor, ufl.tensors.ListTensor, ufl.differentiation.VariableDerivative,
                      ufl.algebra.Sum, ufl.tensoralgebra.Dot, ufl.differentiation.Grad, df.Expression)

# Class to convert raw data to fenics tensorial function objects (and vice-versa)    
class DDFunction(df.Function):
    
    def __init__(self, V, dxm = None, name = ''):
        super().__init__(V, name = name)  
        self.mesh = V.mesh()
        self.V = self.function_space()
        
        if(type(dxm) == type(None)):
            self.dxm = V.dxm
        else:
            self.dxm = dxm
                    
        self.n = self.V.num_sub_spaces()
        self.nel = self.mesh.num_cells()
        self.m = int(self.V.dim()/(self.n*self.nel))
        self.ng = self.nel*self.m 
        
        print("number of gauss points = ", self.m )
                
        self.d = self.__getBlankData()

        self.dofmap = V.dofmap() 
        
        # only needed for the fast data query
        self.map = []
        self.MAP = self.__createMapping()

        
# More general implementation, but possibly slower
    # def data(self):        
    #     for e in df.cells(self.mesh):
    #         dofs = self.dofmap.cell_dofs(e.index())
    #         self.d[e.index(), :, :] = self.vector().vec()[dofs].reshape((-1, self.n)) 
            
    #     return self.d.reshape((-1, self.n))

# Original implementatio (less general), possibly faster
    def data(self):        
        for i in range(self.n):
            self.d[self.map[i,0,:], 0, i] = self.vector().get_local()[self.map[i,1,:]]
            
        return self.d.reshape((-1,self.n))

# Only needed for the faster data() query
    def __createMapping(self): 
        
        for i in range(self.n):
            mapTemp = self.V.sub(i).collapse(True)[1]
            self.map.append( [np.array(list(x)) for x in [mapTemp.keys(), mapTemp.values()]])   
            
        self.map = np.array(self.map)

        MAP = np.zeros((self.map.shape[2], self.n), dtype = int)        
        for i in range(self.n):
            MAP[self.map[i,0,:], i] = self.map[i,1,:]
        
        return MAP
        
                
    def __data2tensorVec(self, d, tenVec = None):
        if(type(tenVec) == type(None)):
            tenVec = np.zeros(self.dim())
            
        dd = d.reshape((-1, self.m, self.n))
        for e in df.cells(self.mesh):
            ie = e.index()
            dofs = self.dofmap.cell_dofs(ie)
            tenVec[dofs] = dd[ie, : , :].flatten()
        
        return tenVec
    
    def update(self, d): # Maybe improve perfomance
        
        if isinstance(d, np.ndarray):
            self.vector().set_local(self.__data2tensorVec(d, self.vector().get_local()[:]))
            
        elif isinstance(d, typesForProjection):
            local_project_given_sol(d, self.V, u = self, dxm = self.dxm)
        
        elif isinstance(d, df.Function) and d.function_space() == self.V:
            self.assign(d)
        
        elif isinstance(d, df.Function) and d.function_space() != self.V:
            local_project_given_sol(d, self.V, u = self, dxm = self.dxm)
        else:
            print("DDFunction.update: Invalid type")
            print(type(d))
            input()
            
            

        # for e in df.cells(self.mesh):
        #     # print("hello, I'm cell ", e.index(), e.global_index(), comm.Get_rank() )
        #     dofs = self.Wdofmap.cell_dofs(e.index())
        #     dofs_tan = self.Wtandofmap.cell_dofs(e.index())
        #     # print(dofs, dofs_tan )
        #     m = self.micromodels[e.index()]
        #     m.setUpdateFlag(False)
        #     self.stress.vector().vec()[dofs] = m.getStress(self.eps.vector().get_local()[dofs])
        #     self.tangent.vector().vec()[dofs_tan] = m.getTangent(self.eps.vector().get_local()[dofs]).flatten()[ind_sym_tensor]
            
        
    def __getBlankData(self):
         return np.zeros(self.V.dim()).reshape((-1, self.m, self.n)) 
