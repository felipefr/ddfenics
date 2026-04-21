#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:59:40 2026

@author: ffiguere

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2026, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or
<f.rocha.felipe@gmail.com>
"""

"""
This demo compares the fenics-based and non-intrusive mode usages. 
The problem solved is a 2d rectangular holed plate, clamped on the left and loaded on the right.
The non-intrusive mode only uses fenicsx to generate FEM stiffness matrices, 
load vector and discrete gradient operator. It could be replaced by any other user's solver. 
Mesh reading and visualisation also uses fenicsx, but it can be easily replaced.
"""

import pyvista as pv
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from timeit import default_timer as timer 

import ddfenicsx as dd 
ddni = dd.non_intrusive_mode

from fenics_funcs import (read_mesh, get_problem, get_nitsche_terms, 
                         get_KFmat, get_Bmat, pyvista_warp_plot)

def noisy_dataset(DB, sig = 0.01, n_copies = 1):
    DB_list = []
    for i in range(n_copies):
        fac = 1.0 + sig*np.random.randn(DB.shape[0], DB.shape[1])
        DB_list.append(DB*fac)
        
    DB = np.array(DB_list).reshape((-1, DB.shape[1]))
    return DB

gdim = 2
porder = 1

BOTTOM_FLAG = "BOTTOM"
TOP_FLAG = "TOP"
LEFT_FLAG = "LEFT"
RIGHT_FLAG = "RIGHT"

# Material properties
msh_file = "four_holes.msh"
E, nu = 210, 0.3
q_right = 0.1
tol_ddcm = 1e-15
Nitmax_ddcm = 100


# Exact Solution 
meshdata = read_mesh(msh_file)
a , L, bcs, V = get_problem(meshdata, E, nu, q_right, LEFT_FLAG, RIGHT_FLAG)
K, F = get_KFmat(a, L, bcs=[])
Kbc, Fbc, a_nitsche, L_nitsche = get_nitsche_terms(meshdata, E, nu, LEFT_FLAG)

K += Kbc
F += Fbc

uh_ex = sp.linalg.spsolve(K, F) 

# DDCM operators
B, WBT, W = get_Bmat(meshdata)

# Creation Database
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
Cmat = np.array([[lmbda + 2*mu, lmbda, 0], [lmbda, lmbda + 2*mu, 0], [0, 0 , 2*mu]])
eps_aux = np.reshape( B@uh_ex, (-1, 3))
sig_aux = eps_aux@Cmat
DB = np.concatenate( (eps_aux, sig_aux), axis = 1)
DB = noisy_dataset(DB, sig = 0.1, n_copies=3)


# Non-intrusive : 
np.random.seed(10)
nmandel = int(0.5*(gdim+1)*gdim)

start = timer()
ddmat = ddni.DDMaterial(DB, sizes = [nmandel, nmandel], addzero = False, shuffle = -1)  # replaces sigma_law = lambda u : ...

metric = ddni.DDMetric(C=Cmat, W = W)
metric2 = ddni.DDMetric(C=Cmat, W = W)
problem = ddni.DDProblemNonIntrusive(2*nmandel, W, B, WBT, F, Kbc, Fbc, metric, acc_op = "APE")

search = ddni.DDSearch(metric, ddmat, algorithm = 'kd_tree', opInit = 'zero', seed = 8)
solver = ddni.DDSolver(problem, search)

solver.solve(tol = tol_ddcm, maxit = Nitmax_ddcm)
sol_u_ni = problem.get_sol()['u'].copy()
end = timer()

time_ni = end - start
hist_ni = solver.hist['relative_distance'].copy()
error_ni = np.linalg.norm(uh_ex - sol_u_ni)


# Fenics-based : 
np.random.seed(10)
start = timer()
ddmat = dd.DDMaterial(DB.reshape((-1,2,nmandel)), addzero = False, shuffle = -1)  # replaces sigma_law = lambda u : ...

Sh = dd.DDSpace(V, nmandel)
spaces = [V, Sh]
metric = dd.DDMetric(Cmat, Sh)
bcs_nitsche = [meshdata, (LEFT_FLAG, bcs[0].g.value)]
problem = dd.DDProblemInfinitesimalStrainNitsche(spaces, L, bcs_nitsche, metric, is_accelerated = True)

search = dd.DDSearch(metric, ddmat, algorithm = 'kd_tree', opInit = 'zero', seed = 8)
solver = dd.DDSolver(problem, search)

solver.solve(tol = tol_ddcm, maxit = Nitmax_ddcm)
sol_u_fenics = problem.get_sol()['u'].x.array[:]
end = timer()

time_fenics = end - start
hist_fenics = solver.hist['relative_distance'].copy()
error_fenics = np.linalg.norm(uh_ex - sol_u_fenics)
error_ni_fenics = np.linalg.norm(sol_u_ni - sol_u_fenics)


print("error ddcm non-intrusive : ", error_ni)
print("error ddcm fenics-based : ", error_fenics)
print("error non-intrusive agaist fenics-based : ", error_ni_fenics)
print("time ddcm non-intrusive : ", time_ni)
print("time ddcm fenics-based : ", time_fenics)

plt.title("relative incremental ddcm error")
plt.plot(hist_ni, '-o', label = "non-intrusive")
plt.plot(hist_fenics, '-o', label = "fenics-based")
plt.legend()
plt.yscale('log')
plt.grid()


pyvista_warp_plot(sol_u_ni, V, scale_fac = 50.0)