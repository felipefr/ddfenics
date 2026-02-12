#!/usr/bin/env python
# coding: utf-8

# 0) **Imports**

# In[51]:


import os, sys
sys.path.append("/home/frocha/sources") # parent folder of ddfenicsx

from timeit import default_timer as timer
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import dolfinx, ufl
print(f"DOLFINx version: {dolfinx.__version__}")
print(f"UFL version: {ufl.__version__}")

from dolfinx import fem, mesh, io
import dolfinx.fem.petsc
from ddfenicsx.utils.fetricks import symgrad_mandel, tensor2mandel, interpolate_quadrature

import pyvista as pv
from dolfinx.plot import vtk_mesh

## Parameters
ty = -0.1
nu = 0.3 
E = 100.0 
alpha = 10e4
Nx =  50 
Ny =  10 
Lx = 2.0
Ly = 0.5
nmandel = 3

# 1) **Consititutive behaviour Definition**

# Strain Energy
# $$
# \psi(\varepsilon(u)) = \frac{\lambda}{2}
# ( tr {\varepsilon}^2 + \frac{\alpha}{2} tr {\varepsilon}^4) + 
# \mu ( |\varepsilon|^2  + \frac{\alpha}{2} |\varepsilon|^4).
# $$

# In[52]:



lamb = nu*E/(1.0+nu)/(1.0-2*nu)
mu = E/(1.0+nu)

# alpha : Equivalent to sig = lamb*df.div(u)*df.Identity(2) + mu*(df.grad(u) + df.grad(u).T)
def psi_e(e):
    tr_e = ufl.tr(e)
    e2 = ufl.inner(e,e)
    
    return (0.5*lamb*(tr_e**2 + 0.5*alpha*tr_e**4) +
           mu*(e2 + 0.5*alpha*e2**2))

psi = lambda w: psi_e(ufl.sym(ufl.grad(w)))


# 2) **Mesh**  

# In[53]:



domain = mesh.create_rectangle(MPI.COMM_WORLD,[[0.0, 0.0],[Lx,Ly]],[Nx,Ny], mesh.CellType.triangle, diagonal = mesh.DiagonalType.right)
tdim = domain.topology.dim
dx = ufl.Measure('dx', domain)

topology, cell_types, geometry = vtk_mesh(domain)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# Plot
# plotter = pv.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
# plotter.enable_parallel_projection()
# plotter.show()


#    

# 3) **Mesh regions** 

# In[54]:


def clamped_boundary(x):
    return np.isclose(x[0],0.0)

fdim = domain.topology.dim-1
clamped_facets = mesh.locate_entities_boundary(domain,fdim,clamped_boundary)

# Neumann boundary conditions
def traction_boundary(x):
    return np.isclose(x[0],Lx)

traction_facets = mesh.locate_entities(domain,fdim,traction_boundary)
traction_facets_tag = mesh.meshtags(domain,fdim,traction_facets,np.full_like(traction_facets,1))
ds = ufl.Measure("ds",domain=domain,subdomain_data=traction_facets_tag)


# 4) **Spaces**

# In[55]:


Uh = fem.functionspace(domain,("Lagrange",1,(tdim,)))
uD = np.array([0.0,0.0],dtype=PETSc.ScalarType)
bcs = [fem.dirichletbc(uD,fem.locate_dofs_topological(Uh,fdim,clamped_facets),Uh)]


# 5. **Variational Formulation**: Minimisation 
# \begin{align} 
# \min_{u \in U} \left ( J(u):=\int_{\Omega} \psi(u) dx - \Pi_{ext}(u) \right) \\
# F(u; v) = \delta J(u;v) = 0 \quad \forall v \in V , \\
# \delta F(u, du; v) = \delta^2 J(u, du;v) \quad \forall v \in V ,
# \end{align} 
# <br>
# 

# In[56]:


du = ufl.TrialFunction(Uh)            # Incremental displacement
v  = ufl.TestFunction(Uh)             # Test function
uh  = fem.Function(Uh, name = "u")                 # Displacement from previous iteration


traction = np.array([0.0, ty], dtype=PETSc.ScalarType)
T = fem.Constant(domain, traction)

P_ext = ufl.inner(T,uh)*ds
P_int = psi(uh)*dx

J = P_int - P_ext

F = ufl.derivative(J, uh, v)
DF = ufl.derivative(F, uh, du) # Optional


# 6) **Solving**

# In[57]:


# Compute solution
start = timer()
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_monitor": None,
    "snes_atol": 1e-9,
    "snes_rtol": 1e-9,
    "snes_stol": 0.0,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

problem = fem.petsc.NonlinearProblem(F, uh, bcs=bcs, J = DF,  petsc_options = petsc_options, 
                                     petsc_options_prefix='nonlinear_elasticity')

problem.solve()

end = timer()

print("Time spent: ", end - start)
print("Norm L2: ", fem.assemble_scalar(fem.form(ufl.inner(uh,uh)*dx)))

print("Norm l2: ", np.linalg.norm(uh.x.array) )
# 0.5066642165089559


# 7. **Pos-Processing**

def sigma_law(w):    
    e = ufl.variable(ufl.sym(ufl.grad(w)))
    return ufl.diff(psi_e(e),e)


# Construct database from previous simulation (using ddfunction just to ease the task)
Sh_scalar = fem.functionspace(domain, ("DG", 0))
eps = symgrad_mandel(uh)
sig = tensor2mandel(sigma_law(uh))
                    
epsh = [ fem.Function(Sh_scalar, name = "strain_{0}".format(i)) for i in range(nmandel)]                     
sigh = [ fem.Function(Sh_scalar, name = "stress_{0}".format(i)) for i in range(nmandel)]

for i in range(nmandel):
    interpolate_quadrature(eps[i], epsh[i])
    interpolate_quadrature(sig[i], sigh[i])


# with XDMF 
# out_file = "./vtk/reference.xdmf"
# with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(uh, t = 0.0)
#     for f in epsh + sigh:
#         xdmf.write_function(f, t = 0.0)


# with Adios
out_file = "./vtk/reference.bp"
with io.VTXWriter(domain.comm, out_file, [uh] + sigh + epsh) as f:
    f.write(0.0)


data = np.vstack([e.x.array for e in epsh] + [s.x.array for s in sigh] ).T
np.savetxt("reference_phase_space.txt", data)

np.savetxt("reference_disp.txt", uh.x.array.reshape((-1,2)))

