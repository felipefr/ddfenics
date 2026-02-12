#!/usr/bin/env python
# coding: utf-8

# 0) **Imports**

# In[1]:
import os, sys
from timeit import default_timer as timer
import copy 
import numpy as np
import matplotlib.pyplot as plt

from petsc4py import PETSc
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import dolfinx, ufl
print(f"DOLFINx version: {dolfinx.__version__}")
print(f"UFL version: {ufl.__version__}")

from dolfinx import fem, mesh, io
import dolfinx.fem.petsc
from ddfenicsx.utils.fetricks import symgrad_mandel, tensor2mandel_np, mandel2tensor_np, interpolate_quadrature
import ddfenicsx as dd

import pyvista as pv
from dolfinx.plot import vtk_mesh

def L2norm(w):
    return np.sqrt( fem.assemble_scalar(fem.form(ufl.inner(w, w)*ufl.dx)))
    

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
tol_ddcm = 1e-9
Nitmax_ddcm = 100
Nd = 100000 # number of points
database_file = 'database_generated.txt' # database_ref.txt to sanity check
eps_range = np.array([[-0.01,0.01], [-0.002, 0.002], [-0.004, 0.001]]).T

# 1) **Consititutive behaviour Definition**

# In[30]:

lamb = nu*E/(1.0+nu)/(1.0-2*nu)
mu = E/(1.0+nu)


np.random.seed(1)
DD = np.zeros((Nd,2,3))

sigma_law = lambda e: (lamb*(1.0 + alpha*np.linalg.trace(e)**2)*np.linalg.trace(e)*np.eye(2) + 
                      2*mu*(1 + alpha*np.dot(e.flatten(),e.flatten()))*e)

sigma_law_mandel = lambda em: tensor2mandel_np(sigma_law(mandel2tensor_np(em)))

for i in range(Nd):
    DD[i,0,:] = eps_range[0,:] + np.random.rand(3)*(eps_range[1,:] - eps_range[0,:])
    DD[i,1,:] = sigma_law_mandel(DD[i,0,:])
    
np.savetxt(database_file, DD.reshape((-1,6)), header = '1.0 \n%d 2 3 3'%Nd, comments = '', fmt='%.8e', )

ddmat = dd.DDMaterial(database_file)  # replaces sigma_law = lambda u : ...
ddmat.plotDB(fig_id = 5)


# 2) **Mesh** (Unchanged) 

# In[31]:


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

# 3) **Mesh regions** (Unchanged)

# In[32]:


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
Sh = dd.DDSpace(Uh, nmandel, 'Quadrature') 
spaces = [Uh, Sh]


# In[40]:

# 5) **Statement and Solving the problem** <br> 
# - DDProblem : States DD equilibrium subproblem and updates.
# - DDSolver : Implements the alternate minimization using SKlearn NearestNeighbors('ball_tree', ...) searchs for the projection onto data.
# - Stopping criteria: $\|d_k - d_{k-1}\|/energy$

traction = np.array([0.0, ty], dtype=PETSc.ScalarType)
T = fem.Constant(domain, traction)

# loading database
ddmat = dd.DDMaterial(database_file, addzero = False, shuffle = -1)  # replaces sigma_law = lambda u : ...

# declaring external load
P_ext = lambda w: ufl.dot(T,w)*ds

metric = dd.DDMetric(ddmat = ddmat , V = Sh)
problem = dd.DDProblemInfinitesimalStrain(spaces, L = P_ext, bcs = bcs, metric = metric)

search = dd.DDSearch(metric, ddmat, algorithm = 'kd_tree', opInit = 'zero', seed = 8)
solver = dd.DDSolver(problem, search)

solver.solve(tol = tol_ddcm, maxit = Nitmax_ddcm)

print("dd u norm:", L2norm(problem.u) )

# In[41]:

# 7) **Postprocessing**

# a) *Convergence*

# In[42]:


hist = solver.hist

fig = plt.figure(1)
plt.title('Minimisation')
plt.plot(hist['relative_energy'], 'o-')
plt.xlabel('iterations')
plt.ylabel('energy gap')
plt.legend(loc = 'best')
plt.yscale('log')
plt.grid()

fig = plt.figure(2)
plt.title('Relative distance (wrt k-th iteration)')
plt.plot(hist['relative_distance'], 'o-', label = "relative_distance")
plt.plot([0,len(hist['relative_energy'])],[tol_ddcm,tol_ddcm])
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('rel. distance')
plt.legend(loc = 'best')
plt.grid()


#  

# c) *Convergence with data*

# In[37]:


relative_norm = lambda x1, x0: L2norm(x1 - x0)/L2norm(x0)

Nd_list = [10,100, 1000, 10000, 50000, 100000] 
hist_list = []
error_u = []
error_eps = []
error_sig = []


data_S =  np.loadtxt("reference_phase_space.txt")
data_U =  np.loadtxt("reference_disp.txt")
sol_ref = {"state" : [dd.DDFunction(Sh, dxm = Sh.dxm), dd.DDFunction(Sh, dxm = Sh.dxm)], "u" : fem.Function(Uh)}   
sol_ref["state"][0].x.array[:] = data_S[:,0:3].flatten()
sol_ref["state"][1].x.array[:] = data_S[:,3:].flatten()
sol_ref["u"].x.array[:] = data_U.flatten()

np.random.seed(1)

for Nd_i in Nd_list:    
    indexes = np.arange(0,Nd).astype('int')
    np.random.shuffle(indexes)
    DD_i = DD[indexes[:Nd_i], : , : ]
    ddmat_i = dd.DDMaterial(DD_i)
    metric_i = dd.DDMetric(ddmat = ddmat_i , V = Sh)
    problem_i = dd.DDProblemInfinitesimalStrain(spaces, L = P_ext, bcs = bcs, metric = metric_i)    
    search_i = dd.DDSearch(metric_i, ddmat_i, algorithm = 'kd_tree', opInit = 'zero', seed = 8)
    solver_i = dd.DDSolver(problem_i, search_i)
    solver_i.solve(tol = tol_ddcm, maxit = Nitmax_ddcm)
    
    hist_list.append(copy.deepcopy(solver_i.hist))
    
    error_u.append(relative_norm(solver_i.sol["u"], sol_ref["u"])) 
    error_eps.append(relative_norm(solver_i.sol["state_mech"][0], sol_ref["state"][0])) 
    error_sig.append(relative_norm(solver_i.sol["state_mech"][1], sol_ref["state"][1]))
    
    
    
plt.figure(8)
plt.title('Energy gap VS database size')
plt.plot(Nd_list, [hist_list[i]['relative_energy'][-1] for i in range(len(Nd_list))], '-o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nd')
plt.ylabel('energy gap')
plt.grid()

plt.figure(9)
plt.title('Relative L2 errors VS database size (wrt classical solutions)')
plt.plot(Nd_list, error_u, '-o', label = 'u')
plt.plot(Nd_list, error_eps, '-o', label = 'eps')
plt.plot(Nd_list, error_sig, '-o', label = 'sig')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nd')
plt.ylabel('Errors')
plt.legend(loc = 'best')
plt.grid()


# 8. **Sanity check:** : Recovering reference database

# In[43]:


state_mech = solver.sol["state_mech"]
state_db = solver.sol["state_db"]

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



# 9. Post-processing
# In[39]:

out_file = "./vtk/ddcm.bp"
Sh_scalar = fem.functionspace(domain, ("DG", 0))

uh = solver.sol['u']                   
epsh = [ fem.Function(Sh_scalar, name = "strain_{0}".format(i)) for i in range(nmandel)]                     
sigh = [ fem.Function(Sh_scalar, name = "stress_{0}".format(i)) for i in range(nmandel)]

for i in range(nmandel):
    epsh[i].x.array[:] = solver.sol['state_mech'][0].x.array[i::nmandel]
    sigh[i].x.array[:] = solver.sol['state_mech'][1].x.array[i::nmandel]
    
with io.VTXWriter(domain.comm, out_file, [uh] + sigh + epsh) as f:
    f.write(0.0)

# In[ ]:




