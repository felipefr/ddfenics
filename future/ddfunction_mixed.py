#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 00:19:03 2023

@author: ffiguere
"""

# from numba import jit
import dolfin as df 
import numpy as np
import ufl
from ddfenics.dd.ddfunction import DDFunction
from fetricks.fenics.la.wrapper_solvers import LocalProjector
from functools import singledispatch

class DDFunctionMixed(df.Function):
    
    def __init__(self, Vlist, dxm = None, name = ''):
        self.mesh = Vlist[0].mesh()
        self.dxm = Vlist[0].dxm
        self.Vlist = Vlist
        
        We = df.MixedElement([X.ufl_element() for X in Vlist])
        Wh = df.FunctionSpace(self.mesh, We)
        
        super().__init__(Wh, name = name)  
        
        self.V = self.function_space() # V is mixed
        self.projector = LocalProjector(self.V, self.dxm, sol = self)
        
        self.n = self.V.num_sub_spaces()
        self.nn = self.V.sub(0).num_sub_spaces()
        self.nel = self.mesh.num_cells()
        self.m = int(self.V.dim()/(self.nn*self.n*self.nel)) # local number gauss points
        self.ng = self.nel*self.m # global number gauss points
        
        print(self.n, self.nel, self.m, self.ng)

        self.fa_V2UU = df.FunctionAssigner(self.Vlist, self.V) # receiver, assigner
        self.fa_UU2V = df.FunctionAssigner(self.V, self.Vlist) # receiver, assigner
        
        self.update = singledispatch(self.update)
        self.update.register(np.ndarray, self.__update_with_array)
        self.update.register(df.Function, self.__update_with_function)

    def data(self):        
        return self.vector().get_local()[:].reshape((-1, self.n)) 

    def update(self, d): # Maybe improve perfomance
        self.projector(d)
    
    def __update_with_array(self, d):
        self.vector().set_local(d.flatten())

    def __update_with_function(self, d):
        self.assign(d)
        
    @staticmethod
    def get_cartesian_product(Vlist, dxm = None):
        input()
        return DDFunctionMixed(Vlist, dxm) 
         