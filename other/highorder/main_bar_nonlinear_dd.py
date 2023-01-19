import os, sys
from timeit import default_timer as timer
import dolfin as df # Fenics : dolfin + ufl + FIAT + ...
import numpy as np
import matplotlib.pyplot as plt
import copy 
import fetricks as ft

# DDfenics imports
from ddfenics.dd.ddmaterial import DDMaterial 
from ddfenics.dd.ddmetric import DDMetric
from ddfenics.dd.ddfunction import DDFunction
from ddfenics.dd.ddproblem_infinitesimalstrain import DDProblemInfinitesimalStrain as DDProblem
from ddfenics.dd.ddsolver import DDSolver


from fetricks.fenics.la.wrapper_solvers import local_project_given_sol



# 1) **Consititutive behaviour Definition**

# In[2]:


# fetricks is a set of utilitary functions to facilitate our lives
from fetricks.mechanics.elasticity_conversions import youngPoisson2lame
from fetricks import Id_mandel_np, tr_mandel

database_file = 'database_generated.txt' # database_ref.txt to sanity check

Nd = 100000 # number of points
degree = 1 # Uh
DG_degree = 1 # Normally DG_degree = degree - 1 
Sh_representantion = 'DG' # To DG work in high-dimension we need more control points

# E = 100.0
# nu = 0.3
# alpha = 10e4
# lamb, mu = youngPoisson2lame(nu, E) 

# np.random.seed(1)

# eps_range = np.array([[-0.008,0.008], [-0.002, 0.002], [-0.0025, 0.00025]]).T
# DD = np.zeros((Nd,2,3))

# sigma_law = lambda e: (lamb*(1.0 + alpha*tr_mandel(e)**2)*tr_mandel(e)*Id_mandel_np + 
#                       2*mu*(1 + alpha*np.dot(e,e))*e)

# for i in range(Nd):
#     DD[i,0,:] = eps_range[0,:] + np.random.rand(3)*(eps_range[1,:] - eps_range[0,:])
#     DD[i,1,:] = sigma_law(DD[i,0,:])
    
# np.savetxt(database_file, DD.reshape((-1,6)), header = '1.0 \n%d 2 3 3'%Nd, comments = '', fmt='%.8e', )

ddmat = DDMaterial(database_file)  # replaces sigma_law = lambda u : ...
DD = ddmat.DB
# ddmat.plotDB()


# 2) **Mesh** (Unchanged) 

# In[3]:


Nx =  50 # x10
Ny =  10 # x10
Lx = 2.0
Ly = 0.5
mesh = df.RectangleMesh(df.Point(0.0,0.0) , df.Point(Lx,Ly), Nx, Ny, 'left/right');
# df.plot(mesh);


#    

# 3) **Mesh regions** (Unchanged)

# In[4]:


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

# In[5]:


from ddfenics.dd.ddspace import DDSpace


Uh = df.VectorFunctionSpace(mesh, "Lagrange", degree) # Unchanged
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0)), boundary_markers, clampedBndFlag) # Unchanged

# Space for stresses and strains
Sh0 = DDSpace(Uh, 3 , Sh_representantion, degree_quad = DG_degree + 1) # Quadrature degree = DG + 1  
# Sh0 = df.VectorFunctionSpace(mesh, 'DG', degree = 0 , dim = 3) 

spaces = [Uh, Sh0]


# Unchanged
ty = -0.1
traction = df.Constant((0.0, ty))

b = lambda w: df.inner(traction,w)*ds(loadBndFlag)


# 6) **Statement and Solving the problem** <br> 
# - DDProblem : States DD equilibrium subproblem and updates.
# - DDSolver : Implements the alternate minimization using SKlearn NearestNeighbors('ball_tree', ...) searchs for the projection onto data.
# - Stopping criteria: $\|d_k - d_{k-1}\|/energy$

# In[7]:


# replaces df.LinearVariationalProblem(a, b, uh, bcs = [bcL])
dx = Sh0.dxm

start = timer()
start_total = timer()

metric = DDMetric(ddmat = ddmat, V = Sh0, dx = Sh0.dxm)
problem = DDProblem(spaces, ft.symgrad_mandel, b, [bcL], metric = metric ) 
sol = problem.get_sol()


#replaces df.LinearVariationalSolver(problem)
solver = DDSolver(problem, ddmat, opInit = 'zero')
tol_ddcm = 1e-7
hist = solver.solve(tol = tol_ddcm, maxit = 100);

end = timer()

uh = sol["u"]
normL2 = df.assemble(df.inner(uh,uh)*dx)
norm_energy = df.assemble(df.inner(sol['state_mech'][0],sol['state_mech'][1])*dx)

print("Time spent: ", end - start)
print("Norm L2: ", normL2)
print("Norm energy: ", norm_energy)

relative_error = lambda x1, x0: np.sqrt(df.assemble( df.inner(x1 - x0, x1 - x0)*Sh0.dxm ) )/np.sqrt(df.assemble( df.inner(x0, x0)*Sh0.dxm ) )

# relative_error = lambda x0, x1: df.errornorm(x0, x1, norm_type = 'L2', degree_rise = 2)/df.norm(x0, norm_type = 'L2')

Sh_ref = df.VectorFunctionSpace(mesh, "DG", DG_degree, dim = 3) 
sol_ref_file =  df.XDMFFile("bar_nonlinear_P{0}_DG1_sol.xdmf".format(degree))
sol_ref = {"state" : [DDFunction(Sh_ref, dxm = Sh0.dxm), DDFunction(Sh_ref, dxm = Sh0.dxm)], "u" : df.Function(Uh)}   
sol_ref_file.read_checkpoint(sol_ref["u"],"u")
sol_ref_file.read_checkpoint(sol_ref["state"][0],"eps")
sol_ref_file.read_checkpoint(sol_ref["state"][1],"sig")

error_u = relative_error(sol_ref["u"], sol["u"])
error_eps = relative_error(sol_ref["state"][0], sol["state_mech"][0])
error_sig = relative_error(sol_ref["state"][1], sol["state_mech"][1])

# Convergence in 21 iterations
# assert np.allclose(normL2, 0.0004429280923646128) # small change from 0.00044305280541032056
# assert np.allclose(norm_energy, 0.0021931246408032315)


print(error_u)
print(error_eps)
print(error_sig)


if(degree == 3):
    assert np.allclose(error_u, 0.00924497320057915)
    assert np.allclose(error_eps, 0.047388834633471724)
    assert np.allclose(error_sig, 0.031010810382930278)
    
elif(degree == 2):
    assert np.allclose(error_u, 0.00942447814261604)
    assert np.allclose(error_eps, 0.045433857591044735)
    assert np.allclose(error_sig, 0.012733014790579809)

elif(degree == 1 and DG_degree == 0):
    assert np.allclose(error_u, 0.00803377328109256)
    assert np.allclose(error_eps, 0.04152470807124927)
    assert np.allclose(error_sig, 0.014129803896221652)    

elif(degree == 1 and DG_degree == 1): # difference is not huge
    assert np.allclose(error_u, 0.008034437290880379)
    assert np.allclose(error_eps, 0.041524708071111076)
    assert np.allclose(error_sig, 0.014129803896213156)


elif(degree == 1 and DG_degree == 1 and Sh_representantion  == "DG"): # difference is not huge
    assert np.allclose(error_u, 0.008034437295984955)
    assert np.allclose(error_eps, 0.041524708071250895)
    assert np.allclose(error_sig, 0.014129803896222728)    
    

# 7) **Postprocessing**

# a) *Convergence*

# In[8]:


hist = solver.hist

fig = plt.figure(2)
plt.title('Minimisation')
plt.plot(hist['relative_energy'], 'o-')
plt.xlabel('iterations')
plt.ylabel('energy gap')
plt.legend(loc = 'best')
plt.yscale('log')
plt.grid()

fig = plt.figure(3)
plt.title('Relative distance (wrt k-th iteration)')
plt.plot(hist['relative_distance'], 'o-', label = "relative_distance")
plt.plot([0,len(hist['relative_energy'])],[tol_ddcm,tol_ddcm])
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('rel. distance')
plt.legend(loc = 'best')
plt.grid()


# b) *scatter plot*

# In[9]:


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

# In[11]:

end_time_total = timer()

print("time total : ", end_time_total - start_total)