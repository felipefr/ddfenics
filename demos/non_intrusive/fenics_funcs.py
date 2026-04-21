#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:03:04 2026

@author: frocha
"""


import pyvista as pv
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import mesh, fem, io, plot
import dolfinx.fem.petsc as petsc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import ufl
import scipy.sparse as sp
from timeit import default_timer as timer 


facm = 0.5*np.sqrt(2.)
def eps(u):
    return ufl.sym(ufl.grad(u))

def ten2man(A):
    return ufl.as_vector([A[0,0], A[1,1], facm*(A[0,1] + A[1,0])])

def petsc2scipy(A, shape = None):
    Ai, Aj, Av = A.getValuesCSR()
    A_scipy = sp.csr_matrix((Av, Aj, Ai), shape = shape)
    return A_scipy



def read_mesh(msh_file, gdim = 2):
    # Usage of meshdata
    #domain = meshdata.mesh
    #cell_tags = meshdata.cell_tags
    #facet_tags = meshdata.facet_tags
    #physical = meshdata.physical_groups
    
    meshdata = io.gmsh.read_from_msh(msh_file, MPI.COMM_WORLD, gdim=gdim)
    return meshdata

def get_nitsche_terms(meshdata, E, nu, CLAMPED_FLAG, gdim = 2, porder = 1):
    gamma = 100
    uD = np.array([0.0,0.0])
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    gamma = gamma*(lmbda + 2*mu)
    
    domain = meshdata.mesh
    cell_tags = meshdata.cell_tags
    facet_tags = meshdata.facet_tags
    physical = meshdata.physical_groups
    
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dsD = ds(physical[CLAMPED_FLAG].tag)
    
    n = ufl.FacetNormal(domain)
    h = ufl.CellDiameter(domain)
    
    u_D_ = fem.Constant(domain, uD)
    lmbda_ = fem.Constant(domain, lmbda)
    mu_ = fem.Constant(domain, mu)

    # -------------------------
    # Variational formulation
    # -------------------------
    V = fem.functionspace(domain, ("CG", porder, (gdim,))) 
    
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    sigma = lambda w : lmbda_ * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2.0 * mu_ * eps(w)
    
    # Nitsche terms
    a = gamma / h * ufl.inner(u, v) * dsD
    a += - ufl.inner(ufl.dot(sigma(u), n), v) * dsD
    a += - ufl.inner(ufl.dot(sigma(v), n), u) * dsD
    
    # Linear form
    L = gamma / h * ufl.inner(u_D_, v) * dsD
    L += - ufl.inner(ufl.dot(sigma(v), n), u_D_) * dsD
    
    A = assemble_matrix(fem.form(a))
    A.assemble()
    
    f = assemble_vector(fem.form(L))
    f.assemble()
    
    Nh = V.dofmap.index_map.size_global

    return petsc2scipy(A, shape = (gdim*Nh, gdim*Nh)), f.array, a, L
   



def get_problem(meshdata, E, nu, q_load, CLAMPED_FLAG, LOAD_FLAG, gdim = 2, porder = 1):
    domain = meshdata.mesh
    cell_tags = meshdata.cell_tags
    facet_tags = meshdata.facet_tags
    physical = meshdata.physical_groups
    
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    
    # -------------------------
    # Function space (vector)
    # -------------------------
    V = fem.functionspace(domain, ("CG", porder, (gdim,)))
    
    # -------------------------
    # Boundary conditions
    # -------------------------
    # Left: fully fixed
    facets = facet_tags.find(physical[CLAMPED_FLAG].tag)
    dofs = fem.locate_dofs_topological(V, physical[CLAMPED_FLAG].dim, facets)
    u_left = np.array([0.0, 0.0], dtype=np.float64)
    bc_left = fem.dirichletbc(u_left, dofs, V)
    
    bcs = [bc_left]
    
    # -------------------------
    # Material model
    # -------------------------
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    lmbda_ = fem.Constant(domain, lmbda)
    mu_ = fem.Constant(domain, mu)
    
    # -------------------------
    # Variational formulation
    # -------------------------
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    sigma = lmbda_ * ufl.tr(eps(u)) * ufl.Identity(gdim) + 2.0 * mu_ * eps(u)
    a = ufl.inner(sigma, eps(v)) * dx
    
    # External tractions
    t_right = fem.Constant(domain, np.array([q_load, 0.0], dtype=np.float64))
    L = ufl.dot(t_right, v) * ds(physical[LOAD_FLAG].tag)


    return a, L, bcs, V

def get_Bmat(meshdata, gdim = 2, porder = 1):
    domain = meshdata.mesh
    
    dx = ufl.Measure("dx", domain=domain)
    
    nmandel = int((gdim+1)*gdim/2.)
    V = fem.functionspace(domain, ("CG", porder, (gdim,)))
    S = fem.functionspace(domain, ("DG", porder-1, (nmandel,)))
    S0 = fem.functionspace(domain, ("DG", porder-1))
    
    s = ufl.TrialFunction(S)
    v = ufl.TestFunction(V)
    p = ufl.TestFunction(S0)
    
    B_form = ufl.inner(s, ten2man(eps(v))) * dx
    W_form = p*dx 
    
    W = assemble_vector(fem.form(W_form))
    W.assemble()
    W = W.array
    
    WBT = assemble_matrix(fem.form(B_form))
    WBT.assemble()
    WBT = petsc2scipy(WBT)
    
    B = sp.diags(np.repeat(1.0/W, nmandel)) @ WBT.T 
    
    return B, WBT, W


def get_KFmat(a,L, bcs):
    k = fem.form(a)
    f = fem.form(L)
    
    K = assemble_matrix(k, bcs)
    K.assemble()
    
    F = assemble_vector(f)
    
    # Convert matrix
    K_scipy = petsc2scipy(K)
    
    # Convert vector
    F_scipy = F.array
    
    return K_scipy, F_scipy




def pyvista_warp_plot(sol_u, V, gdim = 2, scale_fac = 1.0):
    uh = fem.Function(V)
    uh.x.array[:] = sol_u
    
    topology, cell_types, geometry = plot.vtk_mesh(uh.function_space)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    # Displacement vector field
    u_vec = uh.x.array.reshape((-1, gdim))

    # Magnitude
    u_mag = np.linalg.norm(u_vec, axis=1)
    grid.point_data["|u|"] = u_mag    
    grid.point_data["Displacement"] = np.hstack((u_vec , np.zeros_like(u_vec[:,0]).reshape((-1,1))))

    warped = grid.warp_by_vector("Displacement", factor=scale_fac)

    plotter = pv.Plotter()

    plotter.add_mesh(
        warped,
        scalars="|u|",
        show_edges=True,
        cmap="viridis"
    )
        
    plotter.add_mesh(
        grid,
        style="wireframe",
        color="gray",
        line_width=0.2,
        label="Undeformed"
    )
    

    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.show()
