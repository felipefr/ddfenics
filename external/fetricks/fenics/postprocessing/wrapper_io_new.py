import numpy as np
import dolfin as df 

import meshio
import h5py
import xml.etree.ElementTree as ET

def readXDMF_with_markers(meshFile, mesh, comm = df.MPI.comm_world):

    with df.XDMFFile(comm,meshFile) as infile:
        infile.read(mesh)
    
    mvc = df.MeshValueCollection("size_t", mesh, 1)
    with df.XDMFFile(comm, "{0}_faces.xdmf".format(meshFile[:-5])) as infile:
        infile.read(mvc, "faces")
                
    mf  = df.MeshFunction("size_t", mesh, mvc)
  
    mvc = df.MeshValueCollection("size_t", mesh, 2)
    with df.XDMFFile(comm, "{0}_regions.xdmf".format(meshFile[:-5])) as infile:
        infile.read(mvc, "regions")
    
    mt  = df.MeshFunction("size_t", mesh, mvc)
    
    # return mt, mf
    return mt, mf

# Todo: Export just mesh with meshio then update .h5 file with domains and boundary physical markers. Need of just one xdmf file, and also h5 ==> look below

def exportMeshHDF5_fromGMSH(gmshMesh = 'mesh.msh', meshFile = 'mesh.xdmf', labels = {'line' : 'faces', 'triangle' : 'regions'}): #'meshTemp2.msh'    
    geometry = meshio.read(gmshMesh) if type(gmshMesh) == type('s') else gmshMesh
    
    meshFileRad = meshFile[:-5]
    
    # working on mac, error with cell dictionary
    meshio.write(meshFile, meshio.Mesh(points=geometry.points[:,:2], cells={"triangle": geometry.cells["triangle"]})) 

    mesh = meshio.Mesh(points=np.zeros((1,2)), cells={'line': geometry.cells['line']},
                                                                              cell_data={'line': {'faces': geometry.cell_data['line']["gmsh:physical"]}})
    
    meshio.write("{0}_{1}.xdmf".format(meshFileRad,'faces'), mesh)
        
    mesh = meshio.Mesh(points=np.zeros((1,2)), cells={"triangle": np.array([[1,2,3]])}, cell_data={'triangle': {'regions': geometry.cell_data['triangle']["gmsh:physical"]}})
    
    meshio.write("{0}_{1}.xdmf".format(meshFileRad,'regions'), mesh)
    
    # hack to not repeat mesh information
    f = h5py.File("{0}_{1}.h5".format(meshFileRad,'regions'),'r+')
    del f['data1']
    f['data1'] = h5py.ExternalLink(meshFileRad + ".h5", "data1")
    f.close()
    
    g = ET.parse("{0}_{1}.xdmf".format(meshFileRad,'regions'))
    root = g.getroot()
    root[0][0][2].attrib['NumberOfElements'] = root[0][0][3][0].attrib['Dimensions'] # left is topological in level, and right in attributes level
    root[0][0][2][0].attrib['Dimensions'] = root[0][0][3][0].attrib['Dimensions'] + ' 3'
  
    g.write("{0}_{1}.xdmf".format(meshFileRad,'regions'))
    

def exportXDMF_gen(filename, fields, k = -1):
    with df.XDMFFile(filename) as ofile: 
        ofile.parameters["flush_output"] = True
        ofile.parameters["functions_share_mesh"] = True
    
        
        if('vertex' in fields.keys()):
            for field in fields['vertex']:
                ofile.write(field, k) 
        

        if('cell' in fields.keys()):
            for field in fields['cell']:
                for field_i in field.split():
                    ofile.write(field_i, k) 


def exportXDMF_gen_append(ofile, fields, k = -1):
    # if(k==0):
    ofile.parameters["flush_output"] = True
    ofile.parameters["functions_share_mesh"] = True
        
        
    if('vertex' in fields.keys()):
        for field in fields['vertex']:
            ofile.write_checkpoint(field, field.name(), float(k), df.XDMFFile.Encoding.HDF5, True) 
    

    if('cell' in fields.keys()):
        for field in fields['cell']:
            for field_i in field.split():
                ofile.write_checkpoint(field_i,field_i.name(), float(k), df.XDMFFile.Encoding.HDF5, True) 
           
            
               
def exportXDMF_checkpoint_gen(filename, fields):
    with df.XDMFFile(filename) as ofile: 
        ofile.parameters["flush_output"] = True
        ofile.parameters["functions_share_mesh"] = True
            
        count = 0
        if('vertex' in fields.keys()):
            for field in fields['vertex']:
                ofile.write_checkpoint(field, field.name(), count, append = True)
                # count = count + 1
        

        if('cell' in fields.keys()):
            for field in fields['cell']:
                for field_i in field.split():
                    ofile.write_checkpoint(field_i, field_i.name(), count, append = True) 
                    # count = count + 1