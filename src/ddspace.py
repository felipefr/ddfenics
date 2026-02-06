#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:17:03 2022

@author: felipefr

This file is part of ddfenics, a FEniCs-based (Model-Free) Data-driven 
Computational Mechanics implementation.

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""

import dolfinx, ufl
from dolfinx import fem,mesh,plot
import basix

class DDSpace:
    
    def __init__(self, Uh, dim, representation = 'Quadrature', degree_quad = None): # degree_quad seems a keyword

        self.representation = representation
        self.degree_quad = degree_quad if degree_quad else Uh.ufl_element().degree
        self.mesh = Uh.mesh
        self.basix_cell = self.mesh.basix_cell()
        
    
        if(representation == 'DG'):
            self.dxm = ufl.Measure('dx', self.mesh)
            self.W0e = basix.ufl.element("DG", self.basix_cell, degree= self.degree_quad)
            self.We = basix.ufl.element("DG", self.basix_cell, degree= self.degree_quad, shape = (dim,))
            self.space = fem.functionspace(self.mesh, self.We)       
            self.scalar_space = fem.functionspace(self.mesh, self.W0e)
            self.eval_points = self.space.element.interpolation_points()

        elif(representation == 'Quadrature'):
            self.dxm = ufl.Measure("dx", domain=self.mesh, metadata={"quadrature_degree": self.degree_quad, "quadrature_scheme": "default"})
            self.W0e = basix.ufl.quadrature_element(self.basix_cell, degree=self.degree_quad, scheme = "default", value_shape= ())
            self.We = basix.ufl.quadrature_element(self.basix_cell, degree=self.degree_quad, scheme = "default", value_shape = (dim,))
            self.space = fem.functionspace(self.mesh, self.We)       
            self.scalar_space = fem.functionspace(self.mesh, self.W0e)
            basix_celltype = getattr(basix.CellType, self.mesh.topology.cell_type.name)
            points, weights = basix.make_quadrature(basix_celltype, self.degree_quad)
            # basix_celltype = getattr(basix.CellType, self.mesh.topology.cell_type.name)
            # points, weights = basix.make_quadrature(self.basix_celltype, self.degree_quad)
            self.eval_points = points
            self.weights = weights
            
            

        # quadrature_points, weights = basix.make_quadrature(basix_celltype, deg_quad)

