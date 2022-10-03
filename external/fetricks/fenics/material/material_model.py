#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:09:55 2022

@author: felipe
"""

# TO DO: Compatibilise intervaces

import dolfin as df
from .generic_gausspoint_expression import genericGaussPointExpression
import fetricks as ft


class materialModel:
    
    def sigma_op(self, e):
        pass
    
    def tangent_op(self, e):
        pass
    
    def __createInternalVariables(self, W, Wtan, dxm):
        pass
    
    def update(self, e):
        pass
    
    def project_var(self, AA):
        for label in AA.keys(): 
            self.projector_list[label](AA[label], self.varInt[label])


class materialModelExpression(materialModel):
    def __init__(self, W, Wtan, dxm):
        self.strain = df.Function(W) 
        self.projector = ft.LocalProjector(W, dxm)
    
        self.size_tan = Wtan.num_sub_spaces()
        self.size_strain = W.num_sub_spaces()
        
        self.stress = genericGaussPointExpression(self.strain, self.pointwise_stress , (self.size_strain,))
        self.tangent = genericGaussPointExpression(self.strain, self.pointwise_tangent , (self.size_tan,))
    
    def pointwise_stress(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format
        pass
    
    def pointwise_tangent(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format
        pass
    
    def update(self, e):
        self.projector(e, self.strain)
