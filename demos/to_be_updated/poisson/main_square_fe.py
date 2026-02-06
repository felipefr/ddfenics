#!/usr/bin/env python
# coding: utf-8

# 0) **Imports**

# In[65]:


import os, sys
from timeit import default_timer as timer
import dolfin as df # Fenics : dolfin + ufl + FIAT + ...
import numpy as np


# 1) **Consititutive behaviour Definition**

# Negative Flux
# $$
# \mathbf{q} = \alpha(|\nabla u|) \nabla u
# $$

# In[66]:


qL = 100.0
qT = 10.0
c1 = 6.0
c2 = 3.0
c3 = 500.0

alpha_0 = 1000.0
beta = 1e2

# Indeed negative flux for numerical reasons
def flux(g):
    g2 = df.inner(g,g)
    alpha = alpha_0*(1+beta*g2)
    return alpha*g

# def flux(g):
#     g2 = df.inner(g,g)
#     alpha = alpha_0
#     return alpha*g


# 2) **Mesh**  

# In[67]:


Nx =  20 
Ny =  20 
Lx = 1.0
Ly = 1.0
mesh = df.RectangleMesh(df.Point(0.0,0.0) , df.Point(Lx,Ly), Nx, Ny, 'left/right');
df.plot(mesh);


#    

# 3) **Mesh regions** 

# In[60]:


leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx=Lx)

bottomBnd = df.CompiledSubDomain('near(x[1], 0.0) && on_boundary')
topBnd = df.CompiledSubDomain('near(x[1], Ly) && on_boundary', Ly=Ly)

leftFlag = 4
rightFlag = 2
bottomFlag = 1
topFlag = 3

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, leftFlag)
rightBnd.mark(boundary_markers, rightFlag)
bottomBnd.mark(boundary_markers, bottomFlag)
topBnd.mark(boundary_markers, topFlag)


dx = df.Measure('dx', domain=mesh)
ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)


# 4) **Spaces**

# In[61]:


Uh = df.FunctionSpace(mesh, "CG", 1) # Equivalent to CG
bcBottom = df.DirichletBC(Uh, df.Constant(0.0), boundary_markers, bottomFlag)
bcRight = df.DirichletBC(Uh, df.Constant(0.0), boundary_markers, rightFlag)
bcs = [bcBottom, bcRight]


# 5. **Variational Formulation**: Minimisation 
# \begin{align} 
# \min_{u \in U} \left ( J(u):=\int_{\Omega} \psi(u) dx - \Pi_{ext}(u) \right) \\
# F(u; v) = \delta J(u;v) = 0 \quad \forall v \in V , \\
# \delta F(u, du; v) = \delta^2 J(u, du;v) \quad \forall v \in V ,
# \end{align} 
# <br>
# 

# In[62]:


du = df.TrialFunction(Uh)            # Incremental displacement
v  = df.TestFunction(Uh)             # Test function
uh  = df.Function(Uh)                 # Displacement from previous iteration
x = df.SpatialCoordinate(mesh)

flux_left = df.Constant(qL)
flux_top = df.Constant(qT)
source = c3*df.sin(c1*x[0])*df.cos(c2*x[1])

P_ext = source*v*dx + flux_left*v*ds(leftFlag) + flux_top*v*ds(topFlag)
P_int = df.inner(flux(df.grad(uh)), df.grad(v))*dx

Res = P_int - P_ext

Jac = df.derivative(Res, uh, du) 


# 6) **Solving**

# In[63]:


# Compute solution
start = timer()
df.solve(Res == 0, uh, bcs, J = Jac)
end = timer()

print("Time spent: ", end - start)
print("Norm L2: ", df.assemble(df.inner(uh,uh)*dx))


# 7. **Pos-Processing**

# In[64]:


import fetricks.fenics.postprocessing.wrapper_io as iofe

Sh = df.VectorFunctionSpace(mesh, "DG", 0, dim = 2) 
gh = df.project( -df.grad(uh), Sh)
q = flux(df.grad(uh))
qh = df.project( q, Sh)

uh.rename("u",'')
gh.rename("g",'')
qh.rename("q",'')

print(np.min(gh.vector().get_local()[:].reshape((-1,2)), axis = 0))
print(np.max(gh.vector().get_local()[:].reshape((-1,2)), axis = 0))

iofe.exportXDMF_gen("square_fe_vtk.xdmf", fields={'vertex': [uh], 'cell_vector': [gh, qh] })
iofe.exportXDMF_checkpoint_gen("square_fe_sol.xdmf", fields={'vertex': [uh], 'cell': [gh, qh]})


# In[ ]:




