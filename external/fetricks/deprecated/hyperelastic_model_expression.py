#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:50:48 2022

@author: felipe
"""
import dolfin as df
import numpy as np

from .material_model_expression import materialModelExpression 

from fetricks import *

# Constant materials params
class hyperelasticModelExpression(materialModelExpression):
    
    def __init__(self, W, dxm, param):
        
        if('lamb' in param.keys()):
            self.lamb = param['lamb']
            self.mu = param['mu']
            
        else: 
            E = param['E']
            nu = param['nu']
            self.lamb = E*nu/(1+nu)/(1-2*nu)
            self.mu = E/2./(1+nu)
            
        self.alpha = param['alpha']  if 'alpha' in param.keys() else 0.0
        
        super().__init__(W, dxm)
    
    def stress(self, e, cell = None): # in mandel format
    
        ee = np.dot(e,e)
        tre2 = (e[0] + e[1])**2.0
        
        lamb_star = self.lamb*( 1 + self.alpha*tre2)
        mu_star = self.mu*( 1 + self.alpha*ee ) 
        
        return lamb_star*(e[0] + e[1])*Id_mandel_np + 2*mu_star*e
    
    
    def tangent(self, e, cell = None): # in mandel format
        
        ee = np.dot(e,e)
        tre2 = (e[0] + e[1])**2.0
        
        lamb_star = self.lamb*( 1 + 3*self.alpha*tre2)
        mu_star = self.mu*( 1 + self.alpha*ee ) 
        
        D = 4*self.mu*self.alpha*np.outer(e,e)
    
        D[0,0] += lamb_star + 2*mu_star
        D[1,1] += lamb_star + 2*mu_star
        D[0,1] += lamb_star
        D[1,0] += lamb_star
        D[2,2] += 2*mu_star

        return D
