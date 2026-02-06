#!/usr/bin/env python
# coding: utf-8

# 0) **Imports**

# In[11]:


import os, sys
from timeit import default_timer as timer
import dolfin as df # Fenics : dolfin + ufl + FIAT + ...
import numpy as np
import matplotlib.pyplot as plt
import ddfenics as dd

# 1) **Consititutive behaviour Definition**

# In[12]:


# fetricks is a set of utilitary functions to facilitate our lives


database_file = 'database_generated_inv_flux.txt'

Nd = 10000 # number of points
noise = 0.0
# q_range = np.array([[-75574.43503827, -80839.71689618], [83086.58554961, 82225.74890162]])
q_range = np.array([[-130., -40.], [30., 280.]])


qL = 100.0
qT = 10.0
c1 = 6.0
c2 = 3.0
c3 = 500.0

alpha_0 = 0.001
beta = 10
gamma = 0.5

# Indeed negative flux for numerical reasons
def flux_inv(q):
    q2 = np.dot(q,q)
    alpha = alpha_0
    return alpha*q

np.random.seed(1)
DD = np.zeros((Nd,2,2))

for i in range(Nd):
    DD[i,1,:] = q_range[0,:] + np.random.rand(2)*(q_range[1,:] - q_range[0,:])
    DD[i,0,:] = flux_inv(DD[i,1,:]) + noise*np.random.randn(2) 
    
np.savetxt(database_file, DD.reshape((-1,4)), header = '1.0 \n%d 2 2 2'%Nd, comments = '', fmt='%.8e', )

ddmat = dd.DDMaterial(database_file)  # replaces sigma_law = lambda u : ...

plt.plot(DD[:,0,0], DD[:,1,0], 'o')
plt.xlabel('$g_x$')
plt.ylabel('$q_x$')
plt.grid()
plt.plot()


# 2) **Mesh** (Unchanged) 

# In[13]:


Nx =  20 
Ny =  20 
Lx = 1.0
Ly = 1.0
mesh = df.RectangleMesh(df.Point(0.0,0.0) , df.Point(Lx,Ly), Nx, Ny, 'left/right');
# df.plot(mesh);


# 3) **Mesh regions** (Unchanged)

# In[14]:


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


# 4) **Spaces** (Unchanged)

# In[15]:



Uh = df.FunctionSpace(mesh, "CG", 1) # Equivalent to CG
bcBottom = lambda Wh : df.DirichletBC(Wh, df.Constant(0.0), boundary_markers, bottomFlag)
bcRight = lambda Wh : df.DirichletBC(Wh, df.Constant(0.0), boundary_markers, rightFlag)
bcs = [bcBottom, bcRight]

# Space for stresses and strains
Sh0 = dd.DDSpace(Uh, 2, 'DG') 

spaces = [Uh, Sh0]

# Unchanged
x = df.SpatialCoordinate(mesh)

flux_left = df.Constant(qL)
flux_top = df.Constant(qT)
source = c3*df.sin(c1*x[0])*df.cos(c2*x[1])
    
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

g_left = BoundarySource(mesh, flux_left,  degree=2)
g_top = BoundarySource(mesh, flux_top,  degree=2)

bcLeft = lambda Wh : df.DirichletBC(Wh, g_left , boundary_markers, leftFlag)
bcTop = lambda Wh : df.DirichletBC(Wh, g_top , boundary_markers, topFlag)
bcs_W = [bcLeft, bcTop]

# changed 
P_ext = lambda w : source*w*dx 

dddist = dd.DDMetric(ddmat = ddmat, V = Sh0, dx = dx)

# replaces df.LinearVariationalProblem(a, b, uh, bcs = [bcL])
problem = dd.DDProblemPoissonMixed(spaces, df.grad, L = P_ext, bcs = [bcs, bcs_W], metric = dddist) 
sol = problem.get_sol()

start = timer()

#replaces df.LinearVariationalSolver(problem)
search = dd.DDSearch(dddist, ddmat, opInit = 'zero', seed = 2)
solver = dd.DDSolver(problem, search)
tol_ddcm = 3e-6
solver.solve(tol = tol_ddcm, maxit = 100);

end = timer()

uh = sol["u"]
normL2 = df.assemble(df.inner(uh,uh)*dx)
norm_energy = df.assemble(df.inner(sol['state_mech'][0],sol['state_mech'][1])*dx)

print("Time spent: ", end - start)
print("Norm L2: ", normL2)
print("Norm energy: ", norm_energy)


# 7) **Plotting**

# a) *Minimisation*

# In[18]:


hist = solver.hist

fig = plt.figure(2)
plt.title('Minimisation')
plt.plot(hist['relative_energy'], 'o-')
plt.xlabel('iterations')
plt.ylabel('energy')
plt.legend(loc = 'best')
plt.yscale('log')
plt.grid()

fig = plt.figure(3)
plt.title('Relative distance (wrt k-th iteration)')
plt.plot(hist['relative_distance'], 'o-', label = "relative_distance")
plt.plot([0,len(hist['relative_energy'])],[tol_ddcm,tol_ddcm], label = "threshold")
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('rel. distance')
plt.legend(loc = 'best')
plt.grid()


# In[19]:


state_mech = sol["state_mech"]
state_db = sol["state_db"]

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

ax1.set_xlabel(r'$g_{x}$')
ax1.set_ylabel(r'$q_{x}$')
ax1.scatter(ddmat.DB[:, 0, 0], ddmat.DB[:, 1, 0], c='gray')
ax1.scatter(state_db[0].data()[:,0], state_db[1].data()[:,0], c='blue')
ax1.scatter(state_mech[0].data()[:,0], state_mech[1].data()[:,0], marker = 'x', c='black')

ax2.set_xlabel(r'$g_{y}$')
ax2.set_ylabel(r'$q_{y}$')
ax2.scatter(ddmat.DB[:, 0, 1], ddmat.DB[:, 1, 1], c='gray')
ax2.scatter(state_db[0].data()[:,1], state_db[1].data()[:,1], c='blue')
ax2.scatter(state_mech[0].data()[:,1], state_mech[1].data()[:,1], marker = 'x', c='black')


ax3.set_xlabel(r'$g_{x}$')
ax3.set_ylabel(r'$q_{y}$')
ax3.scatter(ddmat.DB[:, 0, 0], ddmat.DB[:, 1, 1], c='gray')
ax3.scatter(state_db[0].data()[:,0], state_db[1].data()[:,1], c='blue')
ax3.scatter(state_mech[0].data()[:,0], state_mech[1].data()[:,1], marker = 'x', c='black')

ax4.set_xlabel(r'$g_{y}$')
ax4.set_ylabel(r'$q_{x}$')
ax4.scatter(ddmat.DB[:, 0, 1], ddmat.DB[:, 1, 0], c='gray')
ax4.scatter(state_db[0].data()[:,1], state_db[1].data()[:,0], c='blue')
ax4.scatter(state_mech[0].data()[:,1], state_mech[1].data()[:,0], marker = 'x', c='black')

plt.tight_layout()


# 

# In[20]:


from ddfenics.utils.postprocessing import *
print("Errors (ref: fe_mixed_flux_inv)")
output_sol_ref = "square_fe_mixed_flux_inv_sol.xdmf"
errors = comparison_with_reference_sol(sol, output_sol_ref, labels = ['u','g','q'])

# Just comparable if the fluxes law are the same (I only manage to do it linear)
# print("Errors (ref: fe)")
# output_sol_ref = "square_fe_sol.xdmf"
# errors = comparison_with_reference_sol(sol, output_sol_ref, labels = ['u','g','q'])

output_vtk = "square_dd_mixed_flux_inv_vtk.xdmf"
generate_vtk_db_mech(sol, output_vtk, labels = ["u", "g", "q"])

assert( np.allclose(np.array(list(errors.values())),
                    np.array([1.764306e-02, 3.292391e-02, 1.701336e-01, 2.061865e-02, 1.700429e-01]) ))



# In[ ]:




