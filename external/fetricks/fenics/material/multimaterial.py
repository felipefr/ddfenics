#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 20:17:35 2022

@author: felipe
"""

import numpy as np

from fetricks.mechanics.elasticity_conversions import eng2mu, eng2lamb, eng2lambPlaneStress
from .wrapper_expression import getMyCoeff

def getMultimaterialExpression(param, M, op='cpp'):    
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    return getMyCoeff(materials, param, op = op, mesh = M)


def getLameExpression(nu1,E1,nu2,E2,M, op= 'cpp', plane_stress = False ):
    
    eng2lamb_ = eng2lambPlaneStress if plane_stress else eng2lamb
    
    mu1 = eng2mu(nu1,E1)
    lamb1 = eng2lamb_(nu1,E1)
    mu2 = eng2mu(nu2,E2)
    lamb2 = eng2lamb_(nu2,E2)

    param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1,mu1], [lamb2,mu2]])
    
    return getMultimaterialExpression(param, M, op = op)