#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:53:57 2024

Updated to fenicsx 0.10 - Tue Fev 12 2026

@author: felipe
"""

import numpy as np

from petsc4py import PETSc
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import dolfinx, ufl
print(f"DOLFINx version: {dolfinx.__version__}")
print(f"UFL version: {ufl.__version__}")

from dolfinx import fem, mesh, plot
import dolfinx.fem.petsc
import ddfenicsx as dd
from ddfenicsx.utils.fetricks import symgrad_mandel, tr_mandel, Id_mandel

# Parameters
E = 100.0
nu = 0.3
L = 5.0
H = 0.5
Nx = 50
Ny =  5
nmandel = 3

# Geometrie/Mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD,[[0.0, 0.0],[L,H]],[Nx,Ny], mesh.CellType.triangle, diagonal = mesh.DiagonalType.right)
# domain = mesh.create_rectangle(MPI.COMM_WORLD,[[0.0, 0.0],[L,H]],[Nx,Ny], mesh.CellType.quadrilateral)
tdim = domain.topology.dim

# Functional space
V = fem.functionspace(domain,("Lagrange",1,(tdim,)))


# In[4]:


# Dirichlet boundary conditions
def clamped_boundary(x):
    return np.isclose(x[0],0.0)

fdim = domain.topology.dim-1
clamped_facets = mesh.locate_entities_boundary(domain,fdim,clamped_boundary)
uD = np.array([0.0,0.0],dtype=PETSc.ScalarType)
bc = fem.dirichletbc(uD,fem.locate_dofs_topological(V,fdim,clamped_facets),V)

# In[5]:


# Neumann boundary conditions
def traction_boundary(x):
    return np.isclose(x[0],L)

traction_facets = mesh.locate_entities(domain,fdim,traction_boundary)
traction_facets_tag = mesh.meshtags(domain,fdim,traction_facets,np.full_like(traction_facets,1))
ds = ufl.Measure("ds",domain=domain,subdomain_data=traction_facets_tag)

T = fem.Constant(domain,np.array([0.1e-3,-1.0e-3],dtype=PETSc.ScalarType))


# ## Model-based solution

# In[6]:


# Constitutive model (as expressions)
lam = nu*E/(1.0+nu)/(1.0-2*nu)
mu2 = E/(1.0+nu)

def epsilon(u):
    return symgrad_mandel(u)

def sigma_hooke(e):
    return lam*tr_mandel(e)*Id_mandel + mu2*e

# Weak form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma_hooke(epsilon(u)),epsilon(v))*ufl.dx
b = ufl.dot(T,v)*ds


# In[8]:


# Solve the problem
std_problem = fem.petsc.LinearProblem(a,b,bcs=[bc],
                                      petsc_options={"ksp_type": "preonly","pc_type": "lu"},
                                      petsc_options_prefix='elasticity')
std_uh = std_problem.solve()

# Construct database from previous simulation (using ddfunction just to ease the task)
Sh = dd.DDSpace(V, nmandel, 'DG') 

eps_all = dd.DDFunction(Sh, Sh.dxm, name = "strain_db")
eps_all.update(epsilon(std_uh))
# print(eps_all.x.array.reshape((-1,nmandel)))
sig_all = dd.DDFunction(Sh, Sh.dxm, name = "stress_db")
sig_all.update(sigma_hooke(epsilon(std_uh)))
# print(sig_all.x.array.reshape((-1,nmandel)))

DB_ref = np.concatenate((eps_all.x.array.reshape((-1,nmandel)),sig_all.x.array.reshape((-1,nmandel))),axis=1)
np.savetxt('database.txt', DB_ref, header = "1.0\n {0} 2 {1} {1}".format(DB_ref.shape[0],nmandel), comments = '')

print("std-fem u norm:", fem.assemble_scalar(fem.form(ufl.inner(std_uh, std_uh)*ufl.dx)))

