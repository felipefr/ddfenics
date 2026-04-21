#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 19:48:00 2026

@author: felipe

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Fri Jan  6 11:44:55 2023

@author: ffiguere

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""

import numpy as np
import scipy
import copy
from sklearn.decomposition import PCA


class DDMetric:
    def __init__(self, C, W):
        self.C = C
        self.W = W
        self.Nqp = len(W)
        
        self.C = None
        self.CC = None
        self.L = None
        self.Linv = None        
        self.Cinv = None        
        
        self.reset(C)

    # =========================
    # Core metric logic
    # =========================
    def reset(self, C):
        self.C = C
        self.Cinv = np.linalg.inv(C)
        self.CC = np.block([[C               , np.zeros_like(C)],
                            [np.zeros_like(C), self.Cinv      ]])
        
        self.L = np.linalg.cholesky(self.CC)
        self.Linv = np.linalg.inv(self.L)
        
    # =========================
    # Distances / norms
    # =========================
    def transformL(self, x):
        return x @ self.L
    
    def transformLinv(self, x):
        return x @ self.Linv

    def norm_loc(self, x):
        return np.linalg.norm(self.transformL(x), axis = 1)
    
    def norm_from_normloc(self, normloc):
        return np.sqrt(np.dot(self.W, normloc**2))

    def norm(self, z):
        return self.norm_from_normloc(self.norm_loc(z))

    def dist(self, z1, z2):
        return self.norm(z1 - z2)

    def norm_energy(self, z):
        n = self.C.shape[0]
        e = np.abs(np.sum(z[:,:n]*z[:,n:], axis=1))
        return self.norm_from_normloc(e)

# =========================
# Estimation interface
# =========================
def estimate(self, DB, estimator_method):
    C = self.estimate_static(DB, estimator_method)
    self.C = self.check_pd(C, self.C)
    return self.C

@staticmethod
def check_pd(C, fallback=None):
    eigvals = np.linalg.eigvalsh(C)
    if np.any(eigvals <= 0):
        if fallback is not None:
            return copy.deepcopy(fallback)
        raise ValueError("Matrix is not positive definite")
    return C
# =========================
# Estimators
# =========================
@staticmethod
def estimate_static(DB, method):
    methods = {
        "PCA": DDMetric.estimate_pca,
        "eigendecomp": DDMetric.estimate_eig,
        "LSQ": DDMetric.estimate_lsq,
        "sylvester": DDMetric.estimate_sylvester,
    }
    return methods[method](DB)

@staticmethod
def estimate_pca(DB):
    d = DB.shape[-1]
    pca = PCA(n_components=d)
    X = DB.reshape((-1, 2*d))
    pca.fit(X)

    Cest = pca.components_[:, d:] @ np.linalg.inv(pca.components_[:, :d])
    return 0.5 * (Cest + Cest.T)

@staticmethod
def estimate_eig(DB):
    d = DB.shape[-1]
    X = DB.reshape((-1, 2*d))

    Corr = X.T @ X
    sig, U = np.linalg.eigh(Corr)

    idx = np.argsort(sig)[::-1]
    U = U[:, idx]

    Cest = U[d:, :d] @ np.linalg.inv(U[:d, :d])
    return 0.5 * (Cest + Cest.T)

@staticmethod
def estimate_lsq(DB):
    EE = DB[:, 0, :].T @ DB[:, 0, :]
    SE = DB[:, 1, :].T @ DB[:, 0, :]

    EE = 0.5 * (EE + EE.T)
    SE = 0.5 * (SE + SE.T)

    C = SE @ np.linalg.inv(EE)
    return 0.5 * (C + C.T)

@staticmethod
def estimate_sylvester(DB):
    EE = DB[:, 0, :].T @ DB[:, 0, :]
    SE = DB[:, 1, :].T @ DB[:, 0, :]

    return scipy.linalg.solve_sylvester(EE, EE, SE + SE.T)
    
