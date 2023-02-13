#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:40:56 2023

@author: ffiguere
"""

import dolfin as df

class DDState(object):
    
    def __init__(self, list_z):
        self.z = list_z
        self.n = len(self.z)

    def as_vector(self):
        z = self.z
        return df.as_vector([z[0][0], z[0][1], z[0][2], z[1][0], z[1][1], z[1][2] ])

    def diff(self, z2):
        return [ self.z[i] - z2[i] for i in range(self.n) ] 
    
    def __sub__(self, z2):
        return self.as_vector() - z2.as_vector()
    
    def __getitem__(self, i):
        return self.z[i]