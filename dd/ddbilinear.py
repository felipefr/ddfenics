#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:41:09 2022

@author: felipe
"""

import dolfin as df 
 
class DDBilinear:
    
    def __init__(self, ddmat, dx, uh, vh):
        self.ddmat = ddmat
        self.dx = dx
        self.uh = uh
        self.vh = vh    
        self.a_gen = lambda X: df.inner( X , self.ddmat.grad(self.vh))*self.dx   
        
        self.bilinear = self.action_disp(self.uh)
        
    def action_strain(self, e):
        return self.a_gen(self.ddmat.sigC_e(e))
    
    def action_stress(self, s):
        return self.a_gen(s)

    def action_disp(self, u):
        return self.a_gen(self.ddmat.sigC(u))