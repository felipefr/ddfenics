"""
This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

#!/usr/bin/env python
# coding: utf-8

# 0) **Imports**

# In[11]:


import os, sys
from timeit import default_timer as timer
# import dolfin as df # Fenics : dolfin + ufl + FIAT + ...
import dolfinx, ufl
from dolfinx import fem,mesh,plot
import dolfinx.fem.petsc
from petsc4py import PETSc
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np
import matplotlib.pyplot as plt
# import fetricksx as ft
import ddfenicsx as dd
from ddproblem_poisson import DDProblemPoisson



# 1) **Consititutive behaviour Definition**

# In[12]:

database_file = 'database_generated.txt'

Nd = 1000 # number of points
noise = 0.01
g_range = np.array([[-0.09, -0.05], [0.05,0.15]])

qL = 100.0
qT = 10.0
c1 = 6.0
c2 = 3.0
c3 = 500.0

alpha_0 = 1000.0
beta = 1e2

# alpha : Equivalent to sig = lamb*ufl.div(u)*ufl.Identity(2) + mu*(ufl.grad(u) + ufl.grad(u).T)
def flux(g):
    g2 = np.dot(g,g)
    alpha = alpha_0*(1+beta*g2)
    return alpha*g

np.random.seed(1)
DD = np.zeros((Nd,2,2))

for i in range(Nd):
    DD[i,0,:] = g_range[0,:] + np.random.rand(2)*(g_range[1,:] - g_range[0,:])
    DD[i,1,:] = flux(DD[i,0,:]) + noise*np.random.randn(2)
    
np.savetxt(database_file, DD.reshape((-1,4)), header = '1.0 \n%d 2 2 2'%Nd, comments = '', fmt='%.8e', )

ddmat = dd.DDMaterial(database_file)  # replaces sigma_law = lambda u : ...

# plt.plot(DD[:,0,0], DD[:,1,0], 'o')
# plt.xlabel('$g_x$')
# plt.ylabel('$q_x$')
# plt.grid()


# # # 2) **Mesh** (Unchanged) 

# # # In[13]:


Nx =  20 
Ny =  20 
Lx = 1.0
Ly = 1.0
# mesh = ufl.RectangleMesh(ufl.Point(0.0,0.0) , ufl.Point(Lx,Ly), Nx, Ny, 'left/right');
domain = mesh.create_rectangle(MPI.COMM_SELF,[[0.0, 0.0],[Lx,Ly]],[Nx,Ny],mesh.CellType.triangle);



# # 3) **Mesh regions** (Unchanged)

# # In[14]:
    
# leftBnd = ufl.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
# rightBnd = ufl.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx=Lx)
# bottomBnd = ufl.CompiledSubDomain('near(x[1], 0.0) && on_boundary')
# topBnd = ufl.CompiledSubDomain('near(x[1], Ly) && on_boundary', Ly=Ly)

leftFlag = 4
rightFlag = 2
bottomFlag = 1
topFlag = 3

def left_boundary_foo(x): 
    return np.isclose(x[0],0.0)

def right_boundary_foo(x):
    return np.isclose(x[0],Lx)

def bottom_boundary_foo(x): 
    return np.isclose(x[1],0.0)

def top_boundary_foo(x):
    return np.isclose(x[1],Ly)


fdim = domain.topology.dim-1
clamped_facets = mesh.locate_entities_boundary(domain,fdim,clamped_boundary)
uD = np.array([0.0,0.0],dtype=PETSc.ScalarType)
bc = fem.dirichletbc(uD,fem.locate_dofs_topological(V,fdim,clamped_facets),V)


# preparing right and bottom boundaries for dirichlet conditions
tdim = domain.topology.dim
fdim = tdim-1
right_facets = mesh.locate_entities_boundary(domain,fdim,right_boundary_foo)
bottom_facets = mesh.locate_entities_boundary(domain,fdim,bottom_boundary_foo)

# # preparing left and top boundaries for neumann conditions
left_facets = mesh.locate_entities(domain,fdim,left_boundary_foo)
top_facets = mesh.locate_entities(domain,fdim,top_boundary_foo)

left_facets_tag = mesh.meshtags(domain,fdim,left_facets,np.full_like(left_facets,leftFlag))
top_facets_tag = mesh.meshtags(domain,fdim,top_facets,np.full_like(top_facets,topFlag))

ds = ufl.Measure("ds",domain=domain,subdomain_data=[left_facets_tag, top_facets_tag])
dx = ufl.Measure('dx', domain=domain)

# # 4) **Spaces** (Unchanged)

# # In[15]:


# from ddfenics.dd.ddspace import DDSpace

# Uh = ufl.FunctionSpace(mesh, "CG", 1) # Equivalent to CG
Uh = fem.FunctionSpace(domain,("Lagrange",1))

zero_petsc = np.array([0.0],dtype=PETSc.ScalarType)[0]
# bcBottom = ufl.DirichletBC(Uh, ufl.Constant(0.0), boundary_markers, bottomFlag)
bcBottom = fem.dirichletbc(ScalarType(0.0), dofs = bottom_facets, V = Uh)
# bcRight = ufl.DirichletBC(Uh, ufl.Constant(0.0), boundary_markers, rightFlag)
bcRight = fem.dirichletbc(ScalarType(0.0), dofs = right_facets, V = Uh)
bcs = [bcBottom, bcRight]


# Space for stresses and strains
Sh = dd.DDSpace(Uh, 2, 'DG') 
spaces = [Uh, Sh]

# # Unchanged
x = ufl.SpatialCoordinate(domain)
flux_left = fem.Constant(domain, ScalarType(qL))
flux_top = fem.Constant(domain, ScalarType(qT))
# source = ufl.sin(x[0])*ufl.cos(x[1])
source = fem.Constant(domain, ScalarType(1.0))

# P_ext = lambda w : source*w*dx + flux_left*w*ds(leftFlag) + flux_top*w*ds(topFlag)
P_ext = lambda w : source*w*dx


metric = dd.DDMetric(ddmat = ddmat, V = Sh.space, dx = dx)


z_mech = dd.DDState([dd.DDFunction(Sh.space, Sh.dxm, name = "strain_mech"), 
                     dd.DDFunction(Sh.space, Sh.dxm, name = "stress_mech")])

z_db = dd.DDState([dd.DDFunction(Sh.space, Sh.dxm, name = "strain_db"), 
                   dd.DDFunction(Sh.space, Sh.dxm, name = "stress_db")]) 

# # 6) **Statement and Solving the problem** <br> 
# # - DDProblem : States DD equilibrium subproblem and updates.
# # - DDSolver : Implements the alternate minimization using SKlearn NearestNeighbors('ball_tree', ...) searchs for the projection onto data.
# # - Stopping criteria: $\|d_k - d_{k-1}\|/energy$

# # In[17]:

# P_ext = fem.form(vh*ds(topFlag))
# l2 = fem.petsc.create_vector(P_ext)

# # replaces ufl.LinearVariationalProblem(a, b, uh, bcs = [bcL])
# problem = DDProblemPoisson(spaces, ufl.grad, L = P_ext, bcs = bcs, metric = metric) 


# mimicking DDProblem

u = fem.Function(Uh)
eta = fem.Function(Uh)

uh = ufl.TrialFunction(Uh)
vh = ufl.TestFunction(Uh)

# # bcs_eta = [fem.bcs.DirichletBC(b) for b in self.bcs] # creates a copy
# # [b.homogenize() for b in bcs_eta] # in the case tested its already homogeneous

C = metric.C_fe

a = ufl.inner(C*ufl.grad(uh), ufl.grad(vh))*dx



b1 = ufl.inner(C*z_db[0] , ufl.grad(vh))*dx     
b2 = P_ext(vh)


bilinear_form_dd = fem.form(a)
A_dd = fem.petsc.assemble_matrix(bilinear_form_dd, bcs = [bcRight])
A_dd.assemble()

# solver_dd = PETSc.KSP().create(domain.comm)
# solver_dd.setOperators(A_dd)
# solver_dd.setType(PETSc.KSP.Type.PREONLY)
# solver_dd.getPC().setType(PETSc.PC.Type.LU)


L1 = fem.form(b1)
L1_ = fem.petsc.create_vector(L1)

L2 = fem.form(b2)
L2_ = fem.petsc.create_vector(L2)


z = [ufl.grad(u), z_db[1] + C*ufl.grad(eta)] # z_mech symbolic




#sol = problem.get_sol()

# start = timer()

# #replaces ufl.LinearVariationalSolver(problem)
# search = dd.DDSearch(metric, ddmat, opInit = 'zero', seed = 2)
# solver = dd.DDSolver(problem, search)
# tol_ddcm = 1e-8
# solver.solve(tol = tol_ddcm, maxit = 100);

# end = timer()

# uh = sol["u"]
# normL2 = ufl.assemble(ufl.inner(uh,uh)*dx)
# norm_energy = ufl.assemble(ufl.inner(sol['state_mech'][0],sol['state_mech'][1])*dx)

# print("Time spent: ", end - start)
# print("Norm L2: ", normL2)
# print("Norm energy: ", norm_energy)


# # 7) **Plotting**

# # a) *Minimisation*

# # In[18]:


# hist = solver.hist

# fig = plt.figure(2)
# plt.title('Minimisation')
# plt.plot(hist['relative_energy'], 'o-')
# plt.xlabel('iterations')
# plt.ylabel('energy')
# plt.legend(loc = 'best')
# plt.yscale('log')
# plt.grid()

# fig = plt.figure(3)
# plt.title('Relative distance (wrt k-th iteration)')
# plt.plot(hist['relative_distance'], 'o-', label = "relative_distance")
# plt.plot([0,len(hist['relative_energy'])],[tol_ddcm,tol_ddcm], label = "threshold")
# plt.yscale('log')
# plt.xlabel('iterations')
# plt.ylabel('rel. distance')
# plt.legend(loc = 'best')
# plt.grid()


# # In[19]:


# state_mech = sol["state_mech"]
# state_db = sol["state_db"]

# fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

# ax1.set_xlabel(r'$g_{x}$')
# ax1.set_ylabel(r'$q_{x}$')
# ax1.scatter(ddmat.DB[:, 0, 0], ddmat.DB[:, 1, 0], c='gray')
# ax1.scatter(state_db[0].data()[:,0], state_db[1].data()[:,0], c='blue')
# ax1.scatter(state_mech[0].data()[:,0], state_mech[1].data()[:,0], marker = 'x', c='black')

# ax2.set_xlabel(r'$g_{y}$')
# ax2.set_ylabel(r'$q_{y}$')
# ax2.scatter(ddmat.DB[:, 0, 1], ddmat.DB[:, 1, 1], c='gray')
# ax2.scatter(state_db[0].data()[:,1], state_db[1].data()[:,1], c='blue')
# ax2.scatter(state_mech[0].data()[:,1], state_mech[1].data()[:,1], marker = 'x', c='black')


# ax3.set_xlabel(r'$g_{x}$')
# ax3.set_ylabel(r'$q_{y}$')
# ax3.scatter(ddmat.DB[:, 0, 0], ddmat.DB[:, 1, 1], c='gray')
# ax3.scatter(state_db[0].data()[:,0], state_db[1].data()[:,1], c='blue')
# ax3.scatter(state_mech[0].data()[:,0], state_mech[1].data()[:,1], marker = 'x', c='black')

# ax4.set_xlabel(r'$g_{y}$')
# ax4.set_ylabel(r'$q_{x}$')
# ax4.scatter(ddmat.DB[:, 0, 1], ddmat.DB[:, 1, 0], c='gray')
# ax4.scatter(state_db[0].data()[:,1], state_db[1].data()[:,0], c='blue')
# ax4.scatter(state_mech[0].data()[:,1], state_mech[1].data()[:,0], marker = 'x', c='black')

# plt.tight_layout()


# # 

# # In[20]:


# from ddfenics.utils.postprocessing import *
# output_sol_ref = "square_fe_sol.xdmf"
# errors = comparison_with_reference_sol(sol, output_sol_ref, labels = ['u','g','q'])

# output_vtk = "square_dd_vtk.xdmf"
# generate_vtk_db_mech(sol, output_vtk, labels = ["u", "g", "q"])


# # I don't know why the error in the gradient is so large
# assert( np.allclose(np.array(list(errors.values())),
#                     np.array([4.623696e-02, 8.813123e-02, 1.974890e+00, 2.179316e-02, 1.982428e+00]) ))




