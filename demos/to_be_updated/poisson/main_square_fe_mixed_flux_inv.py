#!/usr/bin/env python
# coding: utf-8

# 0) **Imports**

# In[65]:


import os, sys
from timeit import default_timer as timer
import dolfin as df # Fenics : dolfin + ufl + FIAT + ...
import numpy as np
import fetricks as ft

import fetricks.fenics.postprocessing.wrapper_io as iofe
import ddfenics as dd

class BoundarySource(df.UserExpression):
    def __init__(self, mesh, g, **kwargs):
        self.mesh = mesh
        self.g = g
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        cell = df.Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        values[0] = self.g*n[0]
        values[1] = self.g*n[1]
    def value_shape(self):
        return (2,)


qL = 100.0
qT = 10.0
c1 = 6.0
c2 = 3.0
c3 = 500.0

alpha_0 =  0.001
beta = 10
gamma = 0.5

# Indeed negative flux for numerical reasons
# def flux_inv(q):
#     q2 = df.inner(q,q)
#     alpha = alpha_0/(beta*q2+1)
#     return alpha*q

def flux_inv(q):
    q2 = df.inner(q,q)
    alpha = alpha_0
    return alpha*q

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
BDM_e = df.FiniteElement("BDM", mesh.ufl_cell(), 1)
DG_e = df.FiniteElement("DG", mesh.ufl_cell(), 0)

We = df.MixedElement(BDM_e, DG_e)
Wh = df.FunctionSpace(mesh, We)

w = df.Function(Wh)

dw = df.TestFunction(Wh)
w_ = df.TrialFunction(Wh)

tau, v = df.split(dw)
q, u = df.split(w)

x = df.SpatialCoordinate(mesh)

flux_left = df.Constant(qL)
flux_top = df.Constant(qT)
source = df.Constant(c3)*df.sin(c1*x[0])*df.cos(c2*x[1])

g_left = BoundarySource(mesh, flux_left,  degree=2)
g_top = BoundarySource(mesh, flux_top,  degree=2)

# g_left = ft.NeumannVectorSourceCpp(mesh, flux_left, 2)
# g_top = ft.NeumannVectorSourceCpp(mesh, flux_top, 2)

bcLeft = df.DirichletBC(Wh.sub(0), g_left , boundary_markers, leftFlag)
bcTop =  df.DirichletBC(Wh.sub(0), g_top , boundary_markers, topFlag)

bcs = [bcLeft, bcTop]

normal = df.FacetNormal(mesh)

P_ext = source*v*dx + df.Constant(0.0)*df.inner(tau, normal)*ds
P_int = df.inner(flux_inv(q), tau)*dx + df.inner(df.div(tau), u)*dx + df.inner(df.div(q), v)*dx  

Res = P_int + P_ext

Jac = df.derivative(Res, w, w_) 

# 6) **Solving**

# In[63]:


# Compute solution
start = timer()
df.solve(Res == 0, w, bcs, J = Jac)
end = timer()

print("Time spent: ", end - start)


# 7. **Pos-Processing**

# In[64]:


q, u = w.split(deepcopy = True)

Sh = df.VectorFunctionSpace(mesh, "DG", 0, dim = 2) 
Uh = df.FunctionSpace(mesh, "CG", 1) 

uh = df.project(u, Uh)
gh = df.project(df.grad(uh), Sh)
# q = flux(df.grad(u))
qh = df.project( q, Sh)

uh.rename("u",'')
gh.rename("g",'')
qh.rename("q",'')
u.rename('u', '')
q.rename('q', '')

print(np.min(gh.vector().get_local()[:].reshape((-1,2)), axis = 0))
print(np.max(gh.vector().get_local()[:].reshape((-1,2)), axis = 0))


print(np.min(qh.vector().get_local()[:].reshape((-1,2)), axis = 0))
print(np.max(qh.vector().get_local()[:].reshape((-1,2)), axis = 0))

iofe.exportXDMF_gen("square_fe_mixed_flux_inv_vtk.xdmf", fields={'vertex': [uh], 'cell_vector': [gh, qh] })
iofe.exportXDMF_checkpoint_gen("square_fe_mixed_flux_inv_sol.xdmf", fields={'vertex': [u], 'cell': [gh, q]})


# In[ ]:




