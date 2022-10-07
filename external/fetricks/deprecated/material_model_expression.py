#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:48:27 2022

@author: felipe
"""

import sys
import dolfin as df
import numpy as np

from .generic_gausspoint_expression import genericGaussPointExpression
import fetricks as ft

class materialModelExpression:
    def __init__(self, W, dxm):
        self.strain = df.Function(W) 
        self.projector = ft.LocalProjector(W, dxm)
        
        self.stress = genericGaussPointExpression(self.strain, self.stress_op , (3,))
        self.tangent = genericGaussPointExpression(self.strain, self.tangent_op , (3,3,))
        
    def stress_op(self, e, cell = None):
        pass
    
    def tangent_op(self,e, cell = None):
        pass
    
    def update(self, e):
        self.projector(e, self.strain)
