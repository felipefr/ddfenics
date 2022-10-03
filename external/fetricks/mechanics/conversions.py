#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:18:10 2021

@author: felipefr
"""
import dolfin as df
import numpy as np
from fetricks.fenics.misc import symgrad

# VOIGT NOTATION
def symgrad_voigt(v):
    return df.as_vector([v[0].dx(0), v[1].dx(1), v[0].dx(1) + v[1].dx(0)])


def stress2Voigt(s):
    return df.as_vector([s[0, 0], s[1, 1], s[0, 1]])


def strain2Voigt(e):
    return df.as_vector([e[0, 0], e[1, 1], 2*e[0, 1]])

def voigt2strain(e):
    return df.as_tensor([[e[0], 0.5*e[2]], [0.5*e[2], e[1]]])

def voigt2stress(s):
    return df.as_tensor([[s[0], s[2]], [s[2], s[1]]])
    
 ## this is voigt
def macro_strain(i):
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return np.array([[Eps_Voigt[0], Eps_Voigt[2]/2.],
                    [Eps_Voigt[2]/2., Eps_Voigt[1]]])
                    
                    
# VOIGT NOTATION: Generic backend
def stress2Voigt_gen(s, backend = df.as_vector):
    return backend([s[0, 0], s[1, 1], s[0, 1]])


def strain2Voigt_gen(e, backend = df.as_vector):
    return backend([e[0, 0], e[1, 1], 2*e[0, 1]])

def voigt2Strain_gen(e, backend = df.as_vector):
    return backend([[e[0], 0.5*e[2]], [0.5*e[2], e[1]]])

def voigt2Stress_gen(s, backend = df.as_vector):
    return backend([[s[0], s[2]], [s[2], s[1]]])

    


# MANDEL NOTATION RELATED FUNCTIONS

sqrt2 = np.sqrt(2)
halfsqrt2 = 0.5*np.sqrt(2)

Id_mandel_df = df.as_vector([1.0, 1.0, 0.0])
Id_mandel_np = np.array([1.0, 1.0, 0.0])

def mandel2tensor_np(X):
    return np.array([[X[0], halfsqrt2*X[2]],
                     [halfsqrt2*X[2], X[1]]])

def tensor2mandel_np(X):
    return np.array([X[0,0], X[1,1], halfsqrt2*(X[0,1] + X[1,0])])


def tensor2mandel(X):
    return df.as_vector([X[0,0], X[1,1], halfsqrt2*(X[0,1] + X[1,0])])


def mandel2tensor(X):
    return df.as_tensor([[X[0], halfsqrt2*X[2]],
                        [halfsqrt2*X[2], X[1]]])

def tensor4th2mandel(X):
    
    return df.as_tensor([ [X[0,0,0,0], X[0,0,1,1], sqrt2*X[0,0,0,1]],
                          [X[0,0,0,0], X[0,0,1,1], sqrt2*X[0,0,0,1]],
                          [sqrt2*X[0,1,0,0], sqrt2*X[0,1,1,1], 2*X[1,1,1,1]] ] )
                      
def tr_mandel(X):
    return X[0] + X[1]


def symgrad_mandel(v): # it was shown somehow to have better performance than doing it explicity
    return tensor2mandel(symgrad(v))
    
# this is in mandel
def macro_strain_mandel(i): 
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return np.array([[Eps_Voigt[0], halfsqrt2*Eps_Voigt[2]],
                    [halfsqrt2*Eps_Voigt[2], Eps_Voigt[1]]])



# STRESS RELATED FUNCTIONS
def sigmaLame(u, lame):
    return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*symgrad(u)

def vonMises(sig):
    s = sig - (1./3)*df.tr(sig)*df.Identity(2)
    return df.sqrt((3./2)*df.inner(s, s)) 






