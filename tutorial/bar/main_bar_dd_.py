#!/usr/bin/env python
# coding: utf-8

# 0) **Imports**

# In[1]:


import os, sys
from timeit import default_timer as timer
import dolfin as df # Fenics : dolfin + ufl + FIAT + ...
import numpy as np
import matplotlib.pyplot as plt

# DDfenics imports
from ddfenics.dd.ddmaterial import DDMaterial 
from ddfenics.dd.ddmetric import DDMetric
from ddfenics.dd.ddfunction import DDFunction
from ddfenics.dd.ddbilinear import DDBilinear
from ddfenics.dd.ddproblem import DDProblem
from ddfenics.dd.ddsolver import DDSolver


# 1) **Consititutive behaviour Definition**

# 2) **Mesh** (Unchanged) 

# In[3]:


# fetricks is a set of utilitary functions to facilitate our lives
# from fetricks import symgrad_mandel
from fetricks.mechanics.elasticity_conversions import youngPoisson2lame
database_file = 'database_dd_test.txt' # previous computed

Nd = 100 # number of points

E = 100.0
nu = 0.3
lamb, mu = youngPoisson2lame(nu, E) 
print(lamb,mu)

Cmat = np.array( [[lamb + 2*mu, lamb, 0], [lamb, lamb + 2*mu, 0], [0, 0, mu]] )

np.random.seed(0)

eps_range = np.array([[-0.015,0.015], [-0.007, 0.007], [-0.1,0.1]]).T
DD = np.zeros((Nd,2,3))

for i in range(Nd):
    DD[i,0,:] = eps_range[0,:] + np.random.rand(3)*(eps_range[1,:] - eps_range[0,:])
    DD[i,1,:] = Cmat@DD[i,0,:]
    
np.savetxt(database_file, DD.reshape((-1,6)), header = '1.0 \n%d 2 3 3'%Nd, comments = '', fmt='%.8e', )

ddmat = DDMaterial(database_file)  # replaces sigma_law = lambda u : ...

# ddmat = DDMaterial(database_file)  # replaces sigma_law = lambda u : ...


print(Cmat)
ddmat.estimateC()
print(ddmat.C)
print(ddmat.grad)

ddmat.plotDB()



# In[4]:


Nx =  10 # x10
Ny =  5 # x10
Lx = 2.0
Ly = 0.5
mesh = df.RectangleMesh(df.Point(0.0,0.0) , df.Point(Lx,Ly), Nx, Ny, 'left/right');
df.plot(mesh);


#    

# 3) **Mesh regions** (Unchanged)

# In[5]:


leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx=Lx)

clampedBndFlag = 1
loadBndFlag = 2
boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, clampedBndFlag)
rightBnd.mark(boundary_markers, loadBndFlag)

dx = df.Measure('dx', domain=mesh)
ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)


# 4) **Spaces**

# In[6]:


Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1) # Unchanged
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0)), boundary_markers, clampedBndFlag) # Unchanged

# Space for stresses and strains
Sh0 = df.VectorFunctionSpace(mesh, 'DG', degree = 0 , dim = 3) 


# 5) **Variational Formulation**: <br>
# 
# - Strong format: 
# $$
# \begin{cases}
# div \sigma = 0  \text{in} \, \Omega \\
# u = 0 \quad \text{on} \, \Gamma_1 \\
# \varepsilon = \nabla^s u \quad \text{in} \, \Omega \\
# \sigma  n  = t \quad \text{on} \, \Gamma_2 \\
# \end{cases}
# $$ 
# - DD equilibrium subproblem: Given $(\varepsilon^*, \sigma^*) \in Z_h$, solve for $(u,\eta) \in U_h$  
# $$
# \begin{cases}
# (\mathbb{C} \nabla^s u , \nabla^s v ) = (\mathbb{C} \varepsilon^* , \nabla^s v ) \quad \forall v \in U_h, \\
# (\mathbb{C} \nabla^s \eta , \nabla^s \xi ) = \Pi_{ext}(\xi) - (\sigma^* , \nabla^s \xi ) \quad \forall \xi \in U_h \\
# \end{cases}
# $$
# - Updates:
# $$
# \begin{cases}
# \varepsilon = \nabla^s u \\
# \sigma = \sigma^* + \mathbb{C} \nabla^s \eta
# \end{cases}
# $$
# - DD ''bilinear'' form : $(\bullet , \nabla^s v)$ or sometimes  $(\mathbb{C} \nabla^s \bullet, \nabla^s v)$ 

# In[7]:


# Unchanged
ty = -0.1
traction = df.Constant((0.0, ty))

u = df.TrialFunction(Uh) 
v = df.TestFunction(Uh)
b = df.inner(traction,v)*ds(loadBndFlag)

a = DDBilinear(ddmat, dx, u, v) # replaces df.inner(sig(u) , df.grad(v))*dx


# 6) **Statement and Solving the problem** <br> 
# - DDProblem : States DD equilibrium subproblem and updates.
# - DDSolver : Implements the alternate minimization using SKlearn NearestNeighbors('ball_tree', ...) searchs for the projection onto data.
# - Stopping criteria: $\|d_k - d_{k-1}\|/energy$

# In[8]:


# Extended solution : replaces u = df.Function(Uh)
sol = {"state_mech" : [DDFunction(Sh0), DDFunction(Sh0)], # mechanical states (eps, sig)
       "state_db": [DDFunction(Sh0), DDFunction(Sh0)],  # database states (eps, sig)
       "u" : df.Function(Uh)} # displacemens

# replaces df.LinearVariationalProblem(a, b, uh, bcs = [bcL])
problem = DDProblem(a, b, sol, [bcL], metric = DDMetric(ddmat.C, V = Sh0, dx = dx)) 

start = timer()
#replaces df.LinearVariationalSolver(problem)
solver = DDSolver(problem, opInit = 'random', seed = 1)
tol = 1e-7
hist = solver.solve(tol = tol, maxit = 100);

end = timer()

uh = sol["u"]
normL2 = df.assemble(df.inner(uh,uh)*dx)
normL2_ref = 0.00046895042573293067

print("Time spent: ", end - start)
print("Norm L2: ", normL2)
print("Norm L2 (Ref)",  normL2_ref)
print("Relative Error",  np.abs(normL2_ref - normL2)/normL2_ref)


# 7) **Postprocessing**

# a) *Convergence*

# In[9]:


hist = solver.hist

fig = plt.figure(1)
plt.title('d_conv = {0}'.format(hist['distance'][-1]))
plt.plot(hist['distance'], 'o-', label = "distance")
plt.xlabel('iterations')
plt.ylabel('distance')
plt.legend(loc = 'best')
plt.grid()

fig = plt.figure(2)
plt.title('d_conv = {0}'.format(hist['relative_energy'][-1]))
plt.plot(hist['relative_distance'], 'o-', label = "relative_distance")
plt.plot(hist['classical_relative_energy'], 'o-' , label = "classical_relative_energy")
plt.plot([0,len(hist['relative_energy'])],[tol,tol])
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('distance')
plt.legend(loc = 'best')
plt.grid()


# b) *Scatter*

# In[10]:


state_mech = sol["state_mech"]
state_db = sol["state_db"]

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


#  

# c) *Pos-Processing*

# In[11]:


import fetricks.fenics.postprocessing.wrapper_io as iofe
from fetricks.fenics.la.wrapper_solvers import local_project

uh.rename('uh', '')
state_mech[1].rename('sigma_mech', '')
state_db[1].rename('sigma_db', '')

iofe.exportXDMF_gen("bar_dd.xdmf" , {'vertex': [uh], 'cell': [state_mech[1], state_db[1]]})


# In[ ]:




