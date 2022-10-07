#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 17:03:12 2022

@author: felipefr
"""
import dolfin as df
import numpy as np
from fetricks.fenics.misc import symgrad

sqrt2 = np.sqrt(2)
halfsqrt2 = 0.5*np.sqrt(2)

Id_mandel_df = df.as_vector([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
Id_mandel_np = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

def mandel2tensor_np(X):
    return np.array([[X[0], halfsqrt2*X[5], halfsqrt2*X[4]],
                     [halfsqrt2*X[5], X[1], halfsqrt2*X[3]],
                     [halfsqrt2*X[4], halfsqrt2*X[3], X[2]] ])

def tensor2mandel_np(X):
    return np.array([X[0,0], X[1,1], X[2,2], 
                     halfsqrt2*(X[1,2] + X[2,1]), 
                     halfsqrt2*(X[0,2] + X[2,0]),
                     halfsqrt2*(X[0,1] + X[1,0])])

def mandel2tensor(X):
    return df.as_tensor( [X[0], halfsqrt2*X[5], halfsqrt2*X[4]],
                         [halfsqrt2*X[5], X[1], halfsqrt2*X[3]],
                         [halfsqrt2*X[4], halfsqrt2*X[3], X[2]])

def tensor2mandel(X):
    return df.as_vector([X[0,0], X[1,1], X[2,2], 
                         halfsqrt2*(X[1,2] + X[2,1]), 
                         halfsqrt2*(X[0,2] + X[2,0]),
                         halfsqrt2*(X[0,1] + X[1,0])])


def tensor4th2mandel(X):
    return df.as_tensor([ [X[0,0,0,0], X[0,0,1,1], X[0,0,2,2], sqrt2*X[0,0,1,2], sqrt2*X[0,0,2,0], sqrt2*X[0,0,0,1]],
                          [X[1,1,0,0], X[1,1,1,1], X[1,1,2,2], sqrt2*X[1,1,1,2], sqrt2*X[1,1,2,0], sqrt2*X[1,1,0,1]],
                          [X[2,2,0,0], X[2,2,1,1], X[2,2,2,2], sqrt2*X[2,2,1,2], sqrt2*X[2,2,2,0], sqrt2*X[2,2,0,1]],
                          [sqrt2*X[1,2,0,0], sqrt2*X[1,2,1,1], sqrt2*X[1,2,2,2], 2*X[1,2,1,2], 2*X[1,2,2,0], 2*X[1,2,0,1]],
                          [sqrt2*X[2,0,0,0], sqrt2*X[2,0,1,1], sqrt2*X[2,0,2,2], 2*X[2,0,1,2], 2*X[2,0,2,0], 2*X[2,0,0,1]],
                          [sqrt2*X[0,1,0,0], sqrt2*X[0,1,1,1], sqrt2*X[0,1,2,2], 2*X[0,1,1,2], 2*X[0,1,2,0], 2*X[0,1,0,1]] ])
                      
def tr_mandel(X):
    return X[0] + X[1] + X[2]


def symgrad_mandel(v):
    return tensor2mandel(symgrad(v))
    
# this is in mandel
def macro_strain_mandel(i): 
    Eps_Mandel = np.zeros((6,))
    Eps_Mandel[i] = 1
    return mandel2tensor_np(Eps_Mandel)


