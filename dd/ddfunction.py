#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:23 2023

@author: felipe
"""

# from numba import jit
import dolfin as df 
import numpy as np
import ufl
from fetricks.fenics.la.wrapper_solvers import LocalProjector
from functools import singledispatch

# Class to convert raw data to fenics tensorial function objects (and vice-versa)    
class DDFunction(df.Function):
    
    def __init__(self, V, dxm = None, name = ''):
        super().__init__(V, name = name)  
        self.mesh = V.mesh()
        self.V = self.function_space()
        self.projector = LocalProjector(self.V, dxm if dxm else V.dxm, sol = self)
        
        self.n = self.V.num_sub_spaces()

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
         
