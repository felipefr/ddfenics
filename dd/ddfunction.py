#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:02:58 2024

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:23:23 2023

@author: felipe
This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""

# import dolfin as df # Fenics : dolfin + ufl + FIAT + ...
import dolfinx, ufl
from dolfinx import fem,mesh,plot
import dolfinx.fem.petsc
from petsc4py import PETSc
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

# from numba import jit
import numpy as np
from functools import singledispatch


# basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
# quadrature_points, weights = basix.make_quadrature(basix_celltype, deg_quad)

# map_c = domain.topology.index_map(domain.topology.dim)
# num_cells = map_c.size_local + map_c.num_ghosts
# cells = np.arange(0, num_cells, dtype=np.int32)


def interpolate_quadrature(ufl_expr, function):
    expr_expr = fem.Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(domain, cells)
    function.x.array[:] = expr_eval.flatten()[:]

def get_indexes_cells(mesh):
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    return np.arange(0, num_cells, dtype=np.int32)
    

def interpolate_quadrature(ufl_expr, function, domain, quadrature_points, cells):
    expr_expr = fem.Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(domain, cells)
    function.x.array[:] = expr_eval.flatten()[:]

# Class to convert raw data to fenics tensorial function objects (and vice-versa)    
class DDFunction(fem.Function):
    
    def __init__(self, V, dxm = None, name = ''):
        super().__init__(V.space, name = name)
        self.V = self.function_space
        self.mesh = self.V.mesh
        self.dxm = dxm if dxm else V.dxm
        
        self.eval_points = V.eval_points
        self.cells = get_indexes_cells(self.mesh)
        
        self.n = self.V.num_sub_spaces
        tdim = self.mesh.topology.dim
        self.nel = self.mesh.topology.index_map(tdim).size_global
        self.m = int(self.x.array.shape[0]/(self.n*self.nel)) # local number gauss points
        self.ng = self.nel*self.m # global number gauss points
                
        self.update = singledispatch(self.update)
        self.update.register(np.ndarray, self.__update_with_array)
        self.update.register(fem.Function, self.__update_with_function)

    def data(self):
        return self.x.array[:].reshape((-1, self.n)) 

    def update(self, d):
        # self.interpolate(fem.Expression(d,self.eval_points))
        interpolate_quadrature(d, self, self.mesh, self.eval_points, self.cells) 
    
    def __update_with_array(self, d):
        # self.vector().set_local(d.flatten())
        self.x.array[:] = d.flatten() # supposed to be faster, but no difference noticed

    def __update_with_function(self, d):
        self.__update_with_array(d.array)
        
    # @staticmethod
    # def split(self):
    #     return self[0], self[1]
    
    def get_cartesian_product(Vlist):
        return [DDFunction(V) for V in Vlist]
         