#!/usr/bin/env python
# coding: utf-8

# 0) **Imports**

# In[1]:

import os, sys
from timeit import default_timer as timer
import numpy as np

from mpi4py import MPI
from dolfinx import mesh
from dolfinx import fem
from dolfinx import nls

import ufl
from petsc4py.PETSc import ScalarType


# 1) **Consititutive behaviour Definition**

# Strain Energy
# $$
# \psi(\varepsilon(u)) = \frac{\lambda}{2}
# ( tr {\varepsilon}^2 + \frac{\alpha}{2} tr {\varepsilon}^4) + 
# \mu ( |\varepsilon|^2  + \frac{\alpha}{2} |\varepsilon|^4).
# $$

# In[8]:

youngPoisson2lame = lambda nu,E : [ nu * E/((1. - 2.*nu)*(1.+nu)) , E/(2.*(1. + nu)) ]

nu = 0.3 
E = 100.0 
alpha = 10e4
lamb, mu = youngPoisson2lame(nu, E)

# alpha : Equivalent to sig = lamb*df.div(u)*df.Identity(2) + mu*(df.grad(u) + df.grad(u).T)
def psi_e(e, domain):
    tr_e = ufl.tr(e)
    e2 = ufl.inner(e,e)
    
    lamb_ = fem.Constant(domain, ScalarType(lamb))
    mu_ = fem.Constant(domain, ScalarType(mu))
    alpha_ = fem.Constant(domain, ScalarType(alpha))
    
    return (0.5*lamb_*(tr_e**2 + 0.5*alpha_*tr_e**4) +
           mu_*(e2 + 0.5*alpha*e2**2))

# 2) **Mesh**  

# In[9]:

Nx =  50 
Ny =  10 
Lx = 2.0
Ly = 0.5
domain = mesh.create_rectangle(MPI.COMM_WORLD,  [np.array([0.0, 0.0]), np.array([Lx, Ly])], [Nx, Ny]);

#    

# 4) **Spaces**

# In[33]:


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


# 5. **Variational Formulation**: Minimisation 
# \begin{align} 
# \min_{u \in U} \left ( J(u):=\int_{\Omega} \psi(u) dx - \Pi_{ext}(u) \right) \\
# F(u; v) = \delta J(u;v) = 0 \quad \forall v \in V , \\
# \delta F(u, du; v) = \delta^2 J(u, du;v) \quad \forall v \in V ,
# \end{align} 
# <br>
# 

# In[16]:

psi = lambda u: psi_e(ufl.sym(ufl.grad(u)) , domain)

v  = ufl.TestFunction(Uh)             # Test function
uh  = fem.Function(Uh)                 # Displacement from previous iteration
du  = ufl.TrialFunction(Uh)             # Test function

J = psi(uh)*dx - ufl.inner(traction,uh)*ds(loadBndFlag)

F = ufl.derivative(J, uh) # Optional
DF = ufl.derivative(F, uh) # Optional

# 6) **Solving**

# In[17]:


# Compute solution
start = timer()

problem = fem.petsc.NonlinearProblem(F, uh, bcs = [bcL], J = DF)

solver = nls.petsc.NewtonSolver(domain.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.report = 1
solver.convergence_criterion = "incremental"

solver.solve(uh)


end = timer()


print("Norm L2: ", np.linalg.norm(uh.vector.array) )