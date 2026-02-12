#!/usr/bin/env python
# coding: utf-8

# 0) **Imports**

# In[1]:


import os, sys
from timeit import default_timer as timer
import numpy as np

from mpi4py import MPI
from dolfinx import mesh

import ufl
from petsc4py.PETSc import ScalarType

# 1) **Consititutive behaviour**

# In[2]:

youngPoisson2lame = lambda nu,E : [ nu * E/((1. - 2.*nu)*(1.+nu)) , E/(2.*(1. + nu)) ]


nu = 0.3 
E = 100.0 
lamb, mu = youngPoisson2lame(nu, E) 

sigma_law = lambda u: lamb*ufl.div(u)*ufl.Identity(2) + mu*(ufl.grad(u) + ufl.grad(u).T)


# 2) **Mesh**  

# In[31]:


Nx =  50 
Ny =  10 
Lx = 2.0
Ly = 0.5
domain = mesh.create_rectangle(MPI.COMM_WORLD,  [np.array([0.0, 0.0]), np.array([Lx, Ly])], [Nx, Ny]);


# 2*) Plot mesh
# In[31]:

from dolfinx import plot
import pyvista
tdim = 2
topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

pyvista.set_jupyter_backend("pythreejs")

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.show()


# 4) **Spaces**

# In[33]:

from dolfinx import fem

Uh = fem.VectorFunctionSpace(domain, ("CG", 1))


# 3) **Mesh regions** 

# In[32]:

clampedBndFlag = 1
loadBndFlag = 2
fdim = domain.topology.dim - 1 # facets topological dimension

bndLeft_lamb = lambda x: np.isclose(x[0], 0.)
bndRight_lamb = lambda x: np.isclose(x[0], Lx)
              
facetsRight = mesh.locate_entities(domain, fdim, bndRight_lamb)
facet_markers = np.full_like(facetsRight, loadBndFlag)
# sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain, fdim, facetsRight, facet_markers)
    
dx = ufl.Measure('dx', domain=domain)
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag)


u_D = fem.Function(Uh)

u_D_c = fem.Constant(domain, np.zeros(2))
# u_D.interpolate(u_D_c)

dofs_L = fem.locate_dofs_geometrical(Uh, bndLeft_lamb)

bcL = fem.dirichletbc(np.zeros(2), dofs_L, Uh)

ty = -0.1
traction = fem.Constant(domain, np.array([0.0, ty]))



# In[35]:


u = ufl.TrialFunction(Uh) 
v = ufl.TestFunction(Uh)
a = ufl.inner(sigma_law(u),  ufl.grad(v))*dx
b = ufl.inner(traction,v)*ds(loadBndFlag)


# 6) **Solving**

# In[41]:

problem = fem.petsc.LinearProblem(a, b, bcs=[bcL], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

print("Norm L2: ", np.linalg.norm(uh.vector.array) )