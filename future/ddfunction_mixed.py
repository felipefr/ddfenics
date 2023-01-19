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

class DDFunctionMixed(DDFunction):
    
    def __init__(self, V, dxm = None, name = ''):
        super().__init__(V, dxm, name = name)  

    @staticmethod
    def get_cartesian_product(Vlist, dxm):
        
        X, Y = Vlist
        
        mesh = X.mesh()
        
        We = df.MixedElement(X.ufl_element(), Y.ufl_element())
        Wh = df.FunctionSpace(mesh, We)
        
        wh = DDFunction(Wh, dxm = dxm)
        
        fa_W2UU = df.FunctionAssigner([X, Y], Wh) # receiver, assigner
        fa_UU2W = df.FunctionAssigner(Wh, [X, Y]) # receiver, assigner
        
        
        x = df.Function(X)
        y = df.Function(X)
        
        fa_W2UU.assign([x,y], wh)
        
        return [x, y, wh] 