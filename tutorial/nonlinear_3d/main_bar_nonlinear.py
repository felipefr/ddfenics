import os, sys
from timeit import default_timer as timer
import dolfin as df # Fenics : dolfin + ufl + FIAT + ...
import numpy as np
import fetricks as ft
from fetricks.mechanics.conversions3d import *

# fetricks is a set of utilitary functions to facilitate our lives
from fetricks.mechanics.elasticity_conversions import youngPoisson2lame


nu = 0.3 
E = 100.0 
alpha = 10e4
lamb, mu = youngPoisson2lame(nu, E)

# alpha : Equivalent to sig = lamb*df.div(u)*df.Identity(2) + mu*(df.grad(u) + df.grad(u).T)
def psi_e(e):
    tr_e = df.tr(e)
    e2 = df.inner(e,e)
    
    return (0.5*lamb*(tr_e**2 + 0.5*alpha*tr_e**4) +
           mu*(e2 + 0.5*alpha*e2**2))

psi = lambda w: psi_e(0.5*(df.grad(w) + df.grad(w).T))


# 2) **Mesh**  
gmsh_mesh = ft.GmshIO("bar3d.geo", 3)
gmsh_mesh.write("xdmf")

mesh = ft.Mesh("bar3d.xdmf")


# 3) **Mesh regions** 

clampedBndFlag = 1 # left
loadBndFlag = 2 # right
Uh = df.VectorFunctionSpace(mesh, "CG", 1) # Equivalent to CG
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0, 0.0)), mesh.boundaries, clampedBndFlag)

# 5. **Variational Formulation**: Minimisation 

du = df.TrialFunction(Uh)            # Incremental displacement
v  = df.TestFunction(Uh)             # Test function
uh  = df.Function(Uh)                 # Displacement from previous iteration

tz = -0.1
traction = df.Constant((0.0, 0.0, tz))

P_ext = df.inner(traction,uh)*mesh.ds(loadBndFlag)
P_int = psi(uh)*mesh.dx

J = P_int - P_ext

F = df.derivative(J, uh, v)
DF = df.derivative(F, uh, du) # Optional


# 6) **Solving**

# Compute solution
start = timer()
df.solve(F == 0, uh, bcL, J = DF)
end = timer()

print("Time spent: ", end - start)
print("Norm L2: ", df.assemble(df.inner(uh,uh)*mesh.dx))

assert np.allclose(df.assemble(df.inner(uh,uh)*mesh.dx),  2851.936202397678) 

# 7. **Pos-Processing**

# In[31]:
import fetricks.fenics.postprocessing.wrapper_io as iofe

def sigma_law(w):    
    e = 0.5*(df.grad(w) + df.grad(w).T) 
    e = df.variable(e)
    return df.diff(psi_e(e),e)

Sh = df.VectorFunctionSpace(mesh, "DG", 0, dim = 6) 
epsh = df.project( symgrad_mandel(uh), Sh)
sig = sigma_law(uh)
sigh = df.project( tensor2mandel(sig), Sh)

uh.rename("u",'')
epsh.rename("eps",'')
sigh.rename("sig",'')

iofe.exportXDMF_gen("bar_nonlinear_vtk.xdmf", fields={'vertex': [uh], 'cell_vector': [epsh, sigh] })
iofe.exportXDMF_checkpoint_gen("bar_nonlinear_sol.xdmf", fields={'vertex': [uh], 'cell': [epsh, sigh]})




