import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
from fetricks.fenics.mesh.mesh import Mesh 

import fetricks as ft 

from timeit import default_timer as timer
from functools import partial 

df.parameters["form_compiler"]["representation"] = 'uflacs'
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

E = 100.0
nu = 0.3
alpha = 200.0
ty = 5.0

mesh = Mesh("./meshes/mesh_40.xdmf")

start = timer()

clampedBndFlag = 2 
LoadBndFlag = 1 
traction = df.Constant((0.0,ty ))
    
deg_u = 1
deg_stress = 0
V = df.VectorFunctionSpace(mesh, "CG", deg_u)
We = df.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=3, quad_scheme='default')
W = df.FunctionSpace(mesh, We)
W0e = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W0 = df.FunctionSpace(mesh, W0e)

bcL = df.DirichletBC(V, df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
bc = [bcL]

def F_ext(v):
    return df.inner(traction, v)*mesh.ds(LoadBndFlag)


metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
dxm = df.dx(metadata=metadata)

model = ft.hyperelasticModelExpression(W, dxm, {'E': E, 'nu': nu, 'alpha': alpha})

u = df.Function(V, name="Total displacement")
du = df.Function(V, name="Iteration correction")
v = df.TestFunction(V)
u_ = df.TrialFunction(V)

a_Newton = df.inner(ft.tensor2mandel(ft.symgrad(u_)), df.dot(model.tangent, ft.tensor2mandel(ft.symgrad(v))) )*dxm
res = -df.inner(ft.tensor2mandel(ft.symgrad(v)), model.stress )*dxm + F_ext(v)

file_results = df.XDMFFile("cook.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True


callbacks = [lambda w: model.updateStrain(ft.tensor2mandel(ft.symgrad(w))) ]

ft.Newton(a_Newton, res, bc, du, u, callbacks , Nitermax = 10, tol = 1e-8)

## Solve here Newton Raphson

file_results.write(u, 0.0)

end = timer()
print(end - start)
