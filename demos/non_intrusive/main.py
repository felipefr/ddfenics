#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:59:40 2026

@author: frocha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:22:07 2026

@author: frocha
"""

import pyvista as pv
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from timeit import default_timer as timer 

import ddfenicsx as dd 
ddni = dd.non_intrusive_mode

from fenics_funcs import get_problem, get_nitsche_terms, get_KFmat, get_Bmat, pyvista_warp_plot

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
CRACK_LEFT = "CRACK_LEFT"
CRACK_RIGHT = "CRACK_RIGHT"

# Material properties
msh_file = "cracked_plate.msh"
E, nu = 210, 0.3
q_right = 1.0
tol_ddcm = 1e-15
Nitmax_ddcm = 100


# Exact Solution 
a , L, bcs, V = get_problem(msh_file, E, nu, q_right, LEFT_FLAG, RIGHT_FLAG)
K, F = get_KFmat(a, L, bcs=[])
Kbc, Fbc, _, _ = get_nitsche_terms(msh_file, E, nu, LEFT_FLAG)

K += Kbc
F += Fbc

uh_ex = sp.linalg.spsolve(K, F) 

# DDCM operators
B, WBT, W = get_Bmat(msh_file)

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
ddmat = ddni.DDMaterial(DB, sizes = [nmandel, nmandel], addzero = False, shuffle = -1)  # replaces sigma_law = lambda u : ...

metric = ddni.DDMetric(C=Cmat, W = W)
metric2 = ddni.DDMetric(C=Cmat, W = W)
problem = ddni.DDProblemNonIntrusive(2*nmandel, W, B, WBT, F, Kbc, Fbc, metric, acc_op = "APE")

search = ddni.DDSearch(metric, ddmat, algorithm = 'kd_tree', opInit = 'zero', seed = 8)
solver = ddni.DDSolver(problem, search)

solver.solve(tol = tol_ddcm, maxit = Nitmax_ddcm)

sol_u = problem.get_sol()['u'].copy()

plt.plot(solver.hist['relative_distance'], '-o')
plt.yscale('log')
plt.grid()


# Fenics-based Non-intrusive : 
np.random.seed(10)
ddmat = dd.DDMaterial(DB.reshape((-1,2,nmandel)), addzero = False, shuffle = -1)  # replaces sigma_law = lambda u : ...

Sh = dd.DDSpace(V, nmandel)
spaces = [V, Sh]
metric = dd.DDMetric(Cmat, Sh)

problem = dd.DDProblemInfinitesimalStrain(spaces, L, bcs, metric, is_accelerated = True)

search = dd.DDSearch(metric, ddmat, algorithm = 'kd_tree', opInit = 'zero', seed = 8)
solver = dd.DDSolver(problem, search)

solver.solve(tol = tol_ddcm, maxit = Nitmax_ddcm)
sol_u2 = problem.get_sol()['u'].x.array[:]

plt.plot(solver.hist['relative_distance'], '-o')
plt.yscale('log')
plt.grid()

print(np.linalg.norm(uh_ex - sol_u))
print(np.linalg.norm(uh_ex - sol_u2))
print(np.linalg.norm(sol_u - sol_u2))

pyvista_warp_plot(sol_u, V)