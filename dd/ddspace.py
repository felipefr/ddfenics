#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:17:03 2022

@author: felipefr
"""

import dolfin as df
# CRITICAL = 50 , ERROR = 40 , WARNING = 30, INFO = 20, PROGRESS = 16, TRACE = 13, DBG = 10
df.set_log_level(50)


class DDSpace(df.FunctionSpace):
    
    def __init__(self, Uh, dim, representation = 'Quadrature', degree_quad = None):

        self.representation = representation
        self.degree_quad = degree_quad if degree_quad else Uh.ufl_element().degree()
        
        if(representation == 'Quadrature'):
            
            
            import warnings
            from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
            warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
            
            
            
            df.parameters["form_compiler"]["representation"] = 'quadrature'
            # df.parameters["form_compiler"]["optimize"] = True
            # df.parameters["form_compiler"]["cpp_optimize"] = True
            # df.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3"

            
            self.Qe = df.VectorElement("Quadrature", Uh.ufl_cell(), 
                                       degree = degree_quad, dim = dim, quad_scheme='default')
            
            
            self.metadata = {"quadrature_degree": degree_quad, "quadrature_scheme": "default"}
            self.dxm = df.Measure( 'dx', Uh.mesh(), metadata= self.metadata)
            
        elif(representation == "DG"):
            self.Qe = df.VectorElement("DG", Uh.ufl_cell(), degree = self.degree_quad - 1, dim = dim)
            self.dxm = df.Measure('dx', Uh.mesh())
            

        super().__init__(Uh.mesh(), self.Qe ) # for stress
    
    # scalar space of the present one (necessary to the distance to make sure dist is evaluated on the right spot)
    def get_scalar_space(self):
        if(self.representation == "Quadrature"):
            Qe = df.VectorElement("Quadrature", self.mesh().ufl_cell(), 
                                  degree = self.sub(0).ufl_element().degree(), dim = 1, quad_scheme='default')
            sh0 = df.FunctionSpace(self.mesh(), Qe ) # for stress
        
        elif(self.representation == "DG"):
            sh0 = df.FunctionSpace(self.mesh() , 'DG', self.sub(0).ufl_element().degree())    
            
        return sh0
