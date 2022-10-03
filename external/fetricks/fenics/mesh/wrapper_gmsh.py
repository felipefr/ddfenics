# ============================================================================= 
# The class Gmsh is wrapper for pygmsh Geometry. It provides basic functionalities
# to interect with meshio and fenics. 
# 
# =============================================================================

import numpy as np
import meshio
import pygmsh
import os
import dolfin as df
from functools import reduce

from fetricks.fenics.postprocessing.wrapper_io import exportMeshHDF5_fromGMSH
from fetricks.fenics.mesh.mesh import Mesh


class Gmsh(pygmsh.built_in.Geometry):
    def __init__(self, meshname = "default.xdmf"):
        super().__init__()   
        self.mesh = None
        self.setNameMesh(meshname)
        self.gmsh_opt = '-algo front2d -smooth 2 -anisoMax 1000.0'
        
    # write .geo file necessary to produce msh files using gmsh
    def writeGeo(self):
        savefile = self.radFileMesh.format('geo')
        f = open(savefile,'w')
        f.write(self.get_code())
        f.close()

    # write .xml files (standard for fenics): 
    def writeXML(self):
        meshXMLFile = self.radFileMesh.format('xml')
        meshMshFile = self.radFileMesh.format('msh')
        self.writeMSH()
        
        os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXMLFile))    
    
    def writeMSH(self, gmsh_opt = ''):
        self.writeGeo()
        meshGeoFile = self.radFileMesh.format('geo')
        meshMshFile = self.radFileMesh.format('msh')
    
        os.system('gmsh -2 -format msh2 {0} {1} -o {2}'.format(self.gmsh_opt, meshGeoFile, meshMshFile))  # with del2d, noticed less distortions
        
        self.mesh = meshio.read(meshMshFile)
        
    def write(self, opt = 'meshio'):
        if(type(self.mesh) == type(None)):
            self.generate()
        if(opt == 'meshio'):
            savefile = self.radFileMesh.format('msh')
            meshio.write(savefile, self.mesh)
        elif(opt == 'fenics'):
            savefile = self.radFileMesh.format('xdmf')
            exportMeshHDF5_fromGMSH(self.mesh, savefile)
            
    def generate(self):
        self.mesh = pygmsh.generate_mesh(self, verbose = False, extra_gmsh_arguments = self.gmsh_opt.split(), dim = 2, mesh_file_type = 'msh2') # it should be msh2 cause of tags    
                                          

    # def getEnrichedMesh(self, savefile = ''):
        
    #     if(len(savefile) == 0):
    #         savefile = self.radFileMesh.format(self.format)
        
    #     if(savefile[-3:] == 'xml'):
    #         self.writeXML(savefile)
            
    #     elif(savefile[-4:] == 'xdmf'):
    #         print("exporting to fenics")
    #         self.write(savefile, 'fenics')
        
    #     return Mesh(savefile)
    
    def setNameMesh(self, meshname):
        self.radFileMesh,  self.format = meshname.split('.')
        self.radFileMesh += '.{0}'
        
        
# self.mesh = pygmsh.generate_mesh(self, verbose=False, dim=2, prune_vertices=True, prune_z_0=True,
# remove_faces=False, extra_gmsh_arguments=gmsh_opt,  mesh_file_type='msh4') # it should be msh2 cause of tags
