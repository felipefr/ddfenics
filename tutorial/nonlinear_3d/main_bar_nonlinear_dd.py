#!/usr/bin/env python
# coding: utf-8

"""
This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

# 0) **Imports**

# In[1]:


import os, sys
from timeit import default_timer as timer
import dolfin as df # Fenics : dolfin + ufl + FIAT + ...
import numpy as np
import matplotlib.pyplot as plt
import copy 
import fetricks as ft
from fetricks.mechanics.conversions3d import *

# DDfenics imports
from ddfenics.dd.ddmaterial import DDMaterial 
from ddfenics.dd.ddmetric import DDMetric
from ddfenics.dd.ddfunction import DDFunction
from ddfenics.dd.ddproblem_infinitesimalstrain import DDProblemInfinitesimalStrain as DDProblem
from ddfenics.dd.ddsolver import DDSolver
from ddfenics.dd.ddspace import DDSpace


# 1) **Consititutive behaviour Definition**

# In[2]:


# fetricks is a set of utilitary functions to facilitate our lives
from fetricks.mechanics.elasticity_conversions import youngPoisson2lame

database_file = 'database_generated.txt' # database_ref.txt to sanity check

Nd = 100000 # number of points

E = 100.0
nu = 0.3
alpha = 10e4
lamb, mu = youngPoisson2lame(nu, E) 

np.random.seed(1)


eps_range = np.array([[-0.00115314, -0.00559695, -0.00172573, -0.00325391, -0.00162043, -0.0027794 ],
                      [0.00120036, 0.00579082, 0.00196701, 0.00163748, 0.00144286, 0.00310545]])

DD = np.zeros((Nd,2,6))

sigma_law = lambda e: (lamb*(1.0 + alpha*tr_mandel(e)**2)*tr_mandel(e)*Id_mandel_np + 
                      2*mu*(1 + alpha*np.dot(e,e))*e)

for i in range(Nd):
    DD[i,0,:] = eps_range[0,:] + np.random.rand(6)*(eps_range[1,:] - eps_range[0,:])
    DD[i,1,:] = sigma_law(DD[i,0,:])
    
np.savetxt(database_file, DD.reshape((-1,12)), header = '1.0 \n%d 2 6 6'%Nd, comments = '', fmt='%.8e', )

ddmat = DDMaterial(database_file)  # replaces sigma_law = lambda u : ...
# ddmat.plotDB()


# 2) **Mesh** (Unchanged) 

# In[3]:
# gmsh_mesh = ft.GmshIO("bar3d.geo", 3)
# gmsh_mesh.write("xdmf")
mesh = ft.Mesh("bar3d.xdmf")


# 3) **Mesh regions** (Unchanged)

# In[4]:


clampedBndFlag = 1 # left
loadBndFlag = 2 # right
Uh = df.VectorFunctionSpace(mesh, "CG", 1) # Equivalent to CG
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0, 0.0)), mesh.boundaries, clampedBndFlag)
# Space for stresses and strains
Sh0 = DDSpace(Uh, 6 , 'DG') 

spaces = [Uh, Sh0]

# Unchanged
tz = -0.1
traction = df.Constant((0.0, 0.0, tz))

b = lambda w: df.inner(traction,w)*mesh.ds(loadBndFlag)

# replaces df.LinearVariationalProblem(a, b, uh, bcs = [bcL])
metric = DDMetric(ddmat = ddmat, V = Sh0, dx = Sh0.dxm)
problem = DDProblem(spaces, symgrad_mandel, b, [bcL], metric = metric ) 
sol = problem.get_sol()

start = timer()
#replaces df.LinearVariationalSolver(problem)
solver = DDSolver(problem, ddmat, opInit = 'zero')
tol_ddcm = 1e-7
hist = solver.solve(tol = tol_ddcm, maxit = 100);

end = timer()

uh = sol["u"]
normL2 = df.assemble(df.inner(uh,uh)*mesh.dx)
norm_energy = df.assemble(df.inner(sol['state_mech'][0],sol['state_mech'][1])*mesh.dx)

print("Time spent: ", end - start)
print("Norm L2: ", normL2)
print("Norm energy: ", norm_energy)

# Convergence in 21 iterations
# assert np.allclose(normL2, 0.00044305280541032056)
# assert np.allclose(norm_energy, 0.0021931246408032315)


# 7) **Postprocessing**

# a) *Convergence*

# In[8]:


hist = solver.hist

fig = plt.figure(2)
plt.title('Minimisation')
plt.plot(hist['relative_energy'], 'o-')
plt.xlabel('iterations')
plt.ylabel('energy gap')
plt.legend(loc = 'best')
plt.yscale('log')
plt.grid()

fig = plt.figure(3)
plt.title('Relative distance (wrt k-th iteration)')
plt.plot(hist['relative_distance'], 'o-', label = "relative_distance")
plt.plot([0,len(hist['relative_energy'])],[tol_ddcm,tol_ddcm])
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('rel. distance')
plt.legend(loc = 'best')
plt.grid()


# b) *scatter plot*

# In[9]:


state_mech = sol["state_mech"]
state_db = sol["state_db"]

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.set_xlabel(r'$\epsilon_{xx}+\epsilon_{yy}$')
ax1.set_ylabel(r'$\sigma_{xx}+\sigma_{yy}$')
ax1.scatter(ddmat.DB[:, 0, 0] + ddmat.DB[:, 0, 1], ddmat.DB[:, 1, 0] + ddmat.DB[:, 1, 1], c='gray')
ax1.scatter(state_db[0].data()[:,0]+state_db[0].data()[:,1],state_db[1].data()[:,0]+state_db[1].data()[:,1], c='blue')
ax1.scatter(state_mech[0].data()[:,0]+state_mech[0].data()[:,1],state_mech[1].data()[:,0]+state_mech[1].data()[:,1], marker = 'x', c='black' )

ax2.set_xlabel(r'$\epsilon_{xy}$')
ax2.set_ylabel(r'$\sigma_{xy}$')
ax2.scatter(ddmat.DB[:, 0, 2], ddmat.DB[:, 1, 2], c='gray')
ax2.scatter(state_db[0].data()[:,2], state_db[1].data()[:,2], c='blue')
ax2.scatter(state_mech[0].data()[:,2], state_mech[1].data()[:,2], marker = 'x', c='black')


