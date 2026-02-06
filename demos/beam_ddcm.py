#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:22:49 2024

@author: felipe
"""
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import dolfinx, ufl
print(f"DOLFINx version: {dolfinx.__version__}")
print(f"UFL version: {ufl.__version__}")

from dolfinx import fem,mesh,plot
import dolfinx.fem.petsc
import ddfenicsx as dd
import fetricksx as ft

# Geometrie/Mesh
L = 5.0
H = 0.5
Nx = 50
Ny =  5
nmandel = 3
tol_ddcm = 1e-9
database_file = 'database.txt'

domain = mesh.create_rectangle(MPI.COMM_WORLD,[[0.0, 0.0],[L,H]],[Nx,Ny],mesh.CellType.triangle, diagonal = mesh.DiagonalType.left)
# domain = mesh.create_rectangle(MPI.COMM_WORLD,[[0.0, 0.0],[L,H]],[Nx,Ny], mesh.CellType.quadrilateral)
tdim = domain.topology.dim

V = fem.FunctionSpace(domain,("Lagrange",1,(tdim,)))

# Dirichlet boundary conditions
def clamped_boundary(x):
    return np.isclose(x[0],0.0)

fdim = domain.topology.dim-1
clamped_facets = mesh.locate_entities_boundary(domain,fdim,clamped_boundary)
uD = np.array([0.0,0.0],dtype=PETSc.ScalarType)
bc = fem.dirichletbc(uD,fem.locate_dofs_topological(V,fdim,clamped_facets),V)

# Neumann boundary conditions
def traction_boundary(x):
    return np.isclose(x[0],L)

traction_facets = mesh.locate_entities(domain,fdim,traction_boundary)
traction_facets_tag = mesh.meshtags(domain,fdim,traction_facets,np.full_like(traction_facets,1))
ds = ufl.Measure("ds",domain=domain,subdomain_data=traction_facets_tag)

T = fem.Constant(domain,np.array([0.1e-3,-1.0e-3],dtype=PETSc.ScalarType))

# loading database
ddmat = dd.DDMaterial(database_file, addzero = False, shuffle =-1)  # replaces sigma_law = lambda u : ...

# declaring and solving ddcm problem
P_ext = lambda w: ufl.dot(T,w)*ds

Sh = dd.DDSpace(V, nmandel, 'DG') 
spaces = [V, Sh]
metric = dd.DDMetric(ddmat = ddmat , V = Sh)
problem = dd.DDProblemInfinitesimalStrain(spaces, ft.symgrad_mandel, L = P_ext, bcs = [bc], metric = metric)

search = dd.DDSearch(metric, ddmat, algorithm = 'kd_tree', opInit = 'zero', seed = 8)
solver = dd.DDSolver(problem, search)

solver.solve(tol = tol_ddcm, maxit = 100); 

print("dd u norm:", fem.petsc.assemble.assemble_scalar(fem.form(ufl.inner(problem.u, problem.u)*ufl.dx)))
