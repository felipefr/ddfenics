#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:40:56 2023

@author: ffiguere

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""

import ufl

class DDState(object):
    
    def __init__(self, list_z):
        self.z = list_z
        self.n = len(self.z)
        self.ngrad = self.z[0].function_space.value_shape[0]

    def as_vector(self):
        z = self.z
        return ufl.as_vector([z[0][i] for i in range(self.ngrad)] + [z[1][i] for i in range(self.ngrad)])

    def diff(self, z2):
        return [ self.z[i] - z2[i] for i in range(self.n) ] 
    
    def __sub__(self, z2):
        return self.as_vector() - z2.as_vector()
    
    def __getitem__(self, i):
        return self.z[i]
    
    def copy(self, deepcopy = True):
        return DDState([z_i.copy(deepcopy = True) for z_i in self.z])