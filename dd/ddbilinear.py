#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 5 17:41:09 2023

@author: felipe

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import dolfin as df 
 
class DDBilinear:
    
    def __init__(self, metric, grad, uh, vh, dx):
        self.C = metric.C_fe
        self.grad = grad
        self.dx = dx
        self.uh = uh
        self.vh = vh    
        self.a_gen = lambda X: df.inner( X , self.grad(self.vh))*self.dx   
        
        self.a_uv = self(self.uh)
        
    def eq_stress(self, q, op):
        if(op == 'u'):
            return df.dot(self.C,self.grad(q))
        elif(op == 'strain'):
            return self.C*q
        elif(op == 'stress'):
            return q
            
    def __call__(self, q, op = 'u'):
        return self.a_gen(self.eq_stress(q,op))  
    