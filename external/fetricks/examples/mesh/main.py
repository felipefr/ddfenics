import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import fetricks as ft 


# from timeit import default_timer as timer
# from functools import partial 

# df.parameters["form_compiler"]["representation"] = 'uflacs'
# import warnings
# from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
# warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

# from fetricks.fenics.mesh.mesh import Mesh 

# from fetricks.fenics.mesh.wrapper_gmsh_new import Gmsh
# import numpy as np

# class CookMembrane(Gmsh):
#     def __init__(self, lcar = 1.0):
#         super().__init__()    
              
#         Lx = 48.0
#         Hy = 44.0
#         hy = 16.0
        
#         hsplit = int(np.sqrt(Lx**2 + Hy**2)/lcar)
#         vsplit = int(0.5*(Hy + hy)/lcar)
        
#         p0 = self.add_point([0.0,0.0,0.0], lcar = lcar)
#         p1 = self.add_point([Lx,Hy,0.0], lcar = lcar)
#         p2 = self.add_point([Lx,Hy+hy,0.0], lcar = lcar)
#         p3 = self.add_point([0.0,Hy,0.0], lcar = lcar)
        
#         l0 = self.add_line(p0,p1)
#         l1 = self.add_line(p1,p2)
#         l2 = self.add_line(p2,p3)
#         l3 = self.add_line(p3,p0)
        
#         self.l = [l0,l1,l2,l3]
#         a = self.add_line_loop(lines = self.l)
#         self.s = self.add_surface(a)
        
#         self.set_transfinite_lines(self.l[0::2], hsplit)
#         self.set_transfinite_lines(self.l[1::2], vsplit)
#         self.set_transfinite_surface(self.s,orientation = 'alternate')

#         self.physicalNaming()
        
#     def physicalNaming(self):
#         self.add_physical(self.l[1], 1)
#         self.add_physical(self.l[3], 2)
#         self.add_physical(self.s,0)

# # elastic parameters

# E = 100.0
# nu = 0.3
# alpha = 200.0
# ty = 5.0

# model = ft.hyperelasticModel({'E': E, 'nu': nu, 'alpha': alpha})

# mesh = Mesh("./meshes/mesh_40.xdmf")

# start = timer()

# clampedBndFlag = 2 
# LoadBndFlag = 1 
# traction = df.Constant((0.0,ty ))
    
# deg_u = 1
# deg_stress = 0
# V = df.VectorFunctionSpace(mesh, "CG", deg_u)
# We = df.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=3, quad_scheme='default')
# W = df.FunctionSpace(mesh, We)
# W0e = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
# W0 = df.FunctionSpace(mesh, W0e)

# bcL = df.DirichletBC(V, df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
# bc = [bcL]

# def F_ext(v):
#     return df.inner(traction, v)*mesh.ds(LoadBndFlag)


# metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
# dxm = df.dx(metadata=metadata)

# model.createInternalVariables(W, W0, dxm)
# u = df.Function(V, name="Total displacement")
# du = df.Function(V, name="Iteration correction")
# v = df.TestFunction(V)
# u_ = df.TrialFunction(V)

# a_Newton = df.inner(ft.tensor2mandel(ft.symgrad(u_)), model.tangent(ft.tensor2mandel(ft.symgrad(v))) )*dxm
# res = -df.inner(ft.tensor2mandel(ft.symgrad(v)), model.sig )*dxm + F_ext(v)

# file_results = df.XDMFFile("cook.xdmf")
# file_results.parameters["flush_output"] = True
# file_results.parameters["functions_share_mesh"] = True


# callbacks = [lambda w: model.update_alpha(ft.tensor2mandel(ft.symgrad(w))) ]

# ft.Newton(a_Newton, res, bc, du, u, callbacks , Nitermax = 10, tol = 1e-8)

# ## Solve here Newton Raphson

# file_results.write(u, 0.0)

# end = timer()
# print(end - start)