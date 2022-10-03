import numpy as np
import meshio
import pygmsh
import os
import dolfin as df
from functools import reduce

import fetricks.fenics.postprocessing.wrapper_io as iofe

class Mesh(df.Mesh):
    def __init__(self, meshFile, comm = df.MPI.comm_world):
        super().__init__(comm)
        
        if(meshFile[-3:] == 'xml'):
            df.File(meshFile) >> self            
            self.subdomains = df.MeshFunction("size_t", self, meshFile[:-4] + "_physical_region.xml")
            self.boundaries = df.MeshFunction("size_t", self, meshFile[:-4] + "_facet_region.xml")
            
        elif(meshFile[-4:] == 'xdmf'):
            self.subdomains, self.boundaries = iofe.readXDMF_with_markers(meshFile, self, comm)
                
        self.ds = df.Measure('ds', domain=self, subdomain_data=self.boundaries)
        self.dx = df.Measure('dx', domain=self, subdomain_data=self.subdomains)
            
        self.V = {}
        self.bcs = {}
        self.dsN = {}
        self.dxR = {}
        
    def createFiniteSpace(self,  spaceType = 'S', name = 'u', spaceFamily = 'CG', degree = 1):
        
        myFunctionSpace = df.TensorFunctionSpace if spaceType =='T' else (df.VectorFunctionSpace if spaceType == 'V' else df.FunctionSpace)
        
        self.V[name] = myFunctionSpace(self, spaceFamily, degree)
        
    def addDirichletBC(self, name = 'default', spaceName = 'u', g = df.Constant(0.0), markerLabel=0, sub = None):
        Vaux = self.V[spaceName] if type(sub)==type(None) else self.V[spaceName].sub(sub)
        self.bcs[name] = df.DirichletBC(Vaux, g , self.boundaries, markerLabel)
    
    def applyDirichletBCs(self,A,b = None):
        if(type(b) == type(None)):
            for bc in self.bcs.values():
                bc.apply(A)
        else:
            for bc in self.bcs.values():
                bc.apply(A,b)
                
    def nameNeumannBoundary(self, name, boundaryMarker):
        self.dsN[name] = reduce(lambda x,y: x+y, [self.ds(b) for b in boundaryMarker] )
        
    def nameRegion(self, name, regionMarker):
        self.dxR[name] = reduce(lambda x,y: x+y, [self.dx(r) for r in regionMarker] )
        
      

    
