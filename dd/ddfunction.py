#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:23:23 2023

@author: felipe
This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""


# from numba import jit
import dolfin as df 
import numpy as np
from fetricks.fenics.la.wrapper_solvers import LocalProjector
from functools import singledispatch
import fetricks as ft

# Class to convert raw data to fenics tensorial function objects (and vice-versa)    
class DDFunction(df.Function):
    
    def __init__(self, V, dxm = None, name = '', 
                 inner_representation = 'quadrature', outer_representation = 'uflacs'):
        super().__init__(V, name = name)  
        self.mesh = V.mesh()
        self.V = self.function_space()
        self.dxm = dxm if dxm else V.dxm
        self.projector = LocalProjector(self.V, self.dxm, self, inner_representation, outer_representation)
        
        self.n = self.V.num_sub_spaces()
        self.nel = self.mesh.num_cells()
        self.m = int(self.V.dim()/(self.n*self.nel)) # local number gauss points
        self.ng = self.nel*self.m # global number gauss points
                
        self.update = singledispatch(self.update)
        self.update.register(np.ndarray, self.__update_with_array)
        self.update.register(df.Function, self.__update_with_function)

    def data(self):
        return self.vector().get_local()[:].reshape((-1, self.n)) 

    def update(self, d):
        self.projector(d)
    
    def __update_with_array(self, d):
        # self.vector().set_local(d.flatten())
        ft.setter(self, d.flatten()) # supposed to be faster, but no difference noticed

    def __update_with_function(self, d):
        
        if(d.function_space().ufl_element().family() == self.V.ufl_element().family()):
            self.assign(d)
        else: 
            self.__update_with_array(d.vector().get_local())
        
    # @staticmethod
    # def split(self):
    #     return self[0], self[1]
    
    def get_cartesian_product(Vlist):
        return [DDFunction(V) for V in Vlist]
         