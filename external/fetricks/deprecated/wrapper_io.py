from __future__ import print_function
import numpy as np
from fenics import *
from dolfin import *
from ufl import nabla_div
import matplotlib.pyplot as plt
import sys, os
import copy

from functools import reduce

from timeit import default_timer as timer
import meshio
import h5py
import xml.etree.ElementTree as ET
from deepBND.core.fenics_tools.wrapper_expression import *
import deepBND.core.elasticity.fenics_utils as fela
from deepBND.core.fenics_tools.wrapper_solvers import local_project


def readXDMF_with_markers(meshFile, mesh, comm = MPI.comm_world):

    with XDMFFile(comm,meshFile) as infile:
        infile.read(mesh)
    
    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile(comm, "{0}_faces.xdmf".format(meshFile[:-5])) as infile:
        infile.read(mvc, "faces")
                
    mf  = MeshFunction("size_t", mesh, mvc)
  
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(comm, "{0}_regions.xdmf".format(meshFile[:-5])) as infile:
        infile.read(mvc, "regions")
    
    mt  = MeshFunction("size_t", mesh, mvc)
    
    # return mt, mf
    return mt, mf


def createMeshFromReference(d, referenceGeo = 'reference.geo'):
    with open('head.geo', 'w') as f:
        [f.write('{0} = {1}; \n'.format(key, value)) for key, value in d.items()]
        
    os.system("cat head.geo " + referenceGeo + " > modifiedReference.geo")
    os.system("gmsh modifiedReference.geo -2 -format 'msh2' -o mesh.msh")

def exportMeshXML(d, referenceGeo = 'reference.geo', meshFile = 'mesh.xml'):
    createMeshFromReference(d, referenceGeo)
    os.system("dolfin-convert mesh.msh " + meshFile)
    # os.system("rm head.geo modifiedReference.geo mesh.msh")

def exportMeshXDMF_fromReferenceGeo(d, referenceGeo = 'reference.geo', meshFile = 'mesh.xdmf'):
    createMeshFromReference(d, referenceGeo)
    exportMeshXDMF_fromGMSH('mesh.msh', meshFile , {'line' : 'faces', 'triangle' : 'regions'})
    
def exportMeshXDMF_fromGMSH(gmshMesh = 'mesh.msh', meshFile = 'mesh.xdmf', labels = {'line' : 'faces', 'triangle' : 'regions'}): #'meshTemp2.msh'
    geometry = meshio.read(gmshMesh)
    
    meshFileRad = meshFile[:-5]
    meshio.write(meshFile, meshio.Mesh(points=geometry.points[:,:2], cells={"triangle": geometry.cells["triangle"]}))
    
    for key, value in labels.items():
        meshio.write("{0}_{1}.xdmf".format(meshFileRad,value), meshio.Mesh(points=geometry.points[:,:2], cells={key: geometry.cells[key]},
                                                                              cell_data={key: {value: geometry.cell_data[key]["gmsh:physical"]}}))

def exportMeshHDF5_fromGMSH(gmshMesh = 'mesh.msh', meshFile = 'mesh.xdmf', labels = {'line' : 'faces', 'triangle' : 'regions'}): #'meshTemp2.msh'
    # Todo: Export just mesh with meshio then update .h5 file with domains and boundary physical markers. Need of just one xdmf file, and also h5 ==> look below
    # import dolfin as df

    # mesh = df.Mesh(xml_mesh_name)
    # mesh_file = df.HDF5File(df.mpi_comm_world(), h5_file_name, 'w')
    # mesh_file.write(mesh, '/mesh')
    
    # # maybe you have defined a mesh-function (boundaries, domains ec.)
    # # in the xml_mesh aswell, in this case use the following two lines
    
    # domains = df.MeshFunction("size_t", mesh, 3, mesh.domains())
    # mesh_file.write(domains, "/domains")
    # read in parallel:
    
    # mesh = df.Mesh()
    # hdf5 = df.HDF5File(df.mpi_comm_world(), h5_file_name, 'r')
    # hdf5.read(mesh, '/mesh', False)
    
    # # in case mesh-functions are available ...
    
    # domains = df.CellFunction("size_t", mesh)
    # hdf5.read(domains, "/domains")
    
    geometry = meshio.read(gmshMesh) if type(gmshMesh) == type('s') else gmshMesh
    
    meshFileRad = meshFile[:-5]
    
    meshio.write(meshFile, meshio.Mesh(points=geometry.points[:,:2], cells={"triangle": geometry.cells["triangle"]})) # working on mac, error with cell dictionary
    # meshio.write(meshFile, meshio.Mesh(points=geometry.points[:,:2], cells={"triangle": geometry.cells}))
        
    mesh = meshio.Mesh(points=np.zeros((1,2)), cells={'line': geometry.cells['line']},
                                                                              cell_data={'line': {'faces': geometry.cell_data['line']["gmsh:physical"]}})
    
    meshio.write("{0}_{1}.xdmf".format(meshFileRad,'faces'), mesh)
    
    # f = h5py.File("{0}_{1}.h5".format(meshFileRad,'faces'),'r+')
    # del f['data0']
    # f['data0'] = h5py.ExternalLink(meshFileRad + ".h5", "data0")
    # f.close()
    
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
    
    

def getDefaultParameters():
    d = {}
    d['lc'] = 0.083
    d['Lx1'] = 0.25
    d['Lx2'] = 0.5
    d['Lx3'] = 0.25
    d['Ly1'] = 0.25
    d['Ly2'] = 0.5
    d['Ly3'] = 0.25
    
    d['theta'] = np.pi/6.0 # angle from the middle square axis
    d['alpha'] = 0.0 # angle of the middle square axis
    d['LxE'] = 0.2
    d['LyE'] = 0.1 
    
    d['Nx1'] = 5
    d['Nx2'] = 7
    d['Nx3'] = 5
    d['Ny1'] = 5
    d['Ny2'] = 7
    d['Ny3'] = 5
    
    return d

def getDefaultParameters_given_hRef(hRef):
    d = getDefaultParameters()
    Nref = int(d['Lx1']/hRef) + 1
    hRef = d['Lx1']/(Nref - 1)
    d['lc'] = hRef
    d['Nx1'] = Nref
    d['Nx2'] = 2*Nref - 1
    d['Nx3'] = Nref
    d['Ny1'] = Nref
    d['Ny2'] = 2*Nref - 1
    d['Ny3'] = Nref
    
    return d

def getAffineParameters(param, d):
    
    x0 = [0.5,0.5]
    t = param[:2]
    l = param[2:]
    
    a = np.zeros((3,2))
    b = np.zeros((3,2))
    
    listLAux = [(d[s + '1'],d[s + '1'] + d[s + '2'], d[s + '3'], d[s + '1'] + d[s + '2'] + d[s + '3'] ) for s in ['Lx','Ly'] ]

    for k in range(2):
        a0 = t[k] - l[k]*x0[k]
        b0 = 1.0 + l[k]
        
        L1,L2,L3,L = listLAux[k]
        # leftmost or bottommost
        a[0,k] = 0.0
        b[0,k] = (a0 + b0*L1)/L1 
    
        # rightmost or uppermost
        b[2,k] = -(a0 + b0*L2 - L)/L3
        a[2,k] = L*(1.0 - b[2,k])
    
        # middle
        b[1,k] = b0
        a[1,k] = a0
  
         
    return a, b

def generateParametrisedMesh(meshFile, param, meshDefaultParam):
    
    mesh = EnrichedMesh('mesh.xml')  
    
    a, b = getAffineParameters(param,meshDefaultParam)
    
    listLAux = [(meshDefaultParam[s + '1'],meshDefaultParam[s + '1'] + meshDefaultParam[s + '2']) for s in ['Lx','Ly'] ]
    
    for xy in mesh.coordinates():
        for k in range(2):
            L1,L2 = listLAux[k]
            i = 0 if xy[k] < L1 else (2 if xy[k] > L2 else 1)            
            xy[k] = a[i,k] + b[i,k]*xy[k]
           
    return mesh


def moveMesh(mesh, meshRef, param, meshDefaultParam):
        
    a, b = getAffineParameters(param,meshDefaultParam)
    
    listLAux = [(meshDefaultParam[s + '1'],meshDefaultParam[s + '1'] + meshDefaultParam[s + '2']) for s in ['Lx','Ly'] ]
    
    for xy, xyRef in zip(mesh.coordinates(), meshRef.coordinates()):
        for k in range(2):
            L1,L2 = listLAux[k]
            i = 0 if xyRef[k] < L1 else (2 if xyRef[k] > L2 else 1)            
            xy[k] = a[i,k] + b[i,k]*xyRef[k]
           


def postProcessing(u, mesh, sigma, lame, outputFile):
    s = sigma(u) - (1./3)*tr(sigma(u))*Identity(2)
    von_Mises = sqrt(((3./2)*inner(s, s))) 
    
    W =  FunctionSpace(mesh, 'DG',0)
    von_Mises = project(von_Mises, W)
    
    sigma_xx = project(sigma(u)[0,0], W)
    sigma_xy = project(sigma(u)[0,1], W)
    sigma_yy = project(sigma(u)[1,1], W)
       
    lame0 = project(lame[0],W)
    lame1 = project(lame[1],W)
   
    fileResults = XDMFFile(outputFile)
    fileResults.parameters["flush_output"] = True
    fileResults.parameters["functions_share_mesh"] = True

    u.rename('u', 'displacements at nodes')
    von_Mises.rename('von_Mises', 'Von Mises stress')
    sigma_xx.rename('sigma_xx', 'Cauchy stress in component xx')
    sigma_xy.rename('sigma_xy', 'Cauchy stress in component xy')
    sigma_yy.rename('sigma_yy', 'Cauchy stress in component yy')
    lame0.rename('lamb', 'Lame coefficient 0')
    lame1.rename('mu', 'Lame coefficient 0')
    
    fileResults.write(u,0.)
    fileResults.write(von_Mises,0.)    
    fileResults.write(sigma_xx,0.)
    fileResults.write(sigma_xy,0.)
    fileResults.write(sigma_yy,0.)
    fileResults.write(lame0,0.)
    fileResults.write(lame1,0.)

def postProcessing_simple(u, outputFile, comm = MPI.comm_world):
    fileResults = XDMFFile(comm,outputFile)
    fileResults.parameters["flush_output"] = True
    fileResults.parameters["functions_share_mesh"] = True
    
    u.rename('u', 'displacements at nodes')    
    fileResults.write(u,0.)
    fileResults.close()
    
def postProcessing_temporal(u, outputFile, comm = MPI.comm_world):
    fileResults = XDMFFile(comm,outputFile)
    fileResults.parameters["flush_output"] = True
    fileResults.parameters["functions_share_mesh"] = True
    
    for i, ui in enumerate(u):
        ui.rename('u', 'displacements at nodes')    
        fileResults.write(ui,float(i))

    fileResults.close()

def postProcessing_complete(u, outputFile, labels = [], param = [], rename = True):
    fileResults = XDMFFile(outputFile)
    fileResults.parameters["flush_output"] = True
    fileResults.parameters["functions_share_mesh"] = True
    
    
    if 'u' in labels:
        if(rename):
            u.rename('u', 'displacements at nodes')    
        
        fileResults.write(u,0.)
    
    if 'vonMises' in labels:
        mesh = u.function_space().mesh()
        materials = mesh.subdomains.array().astype('int32')
        materials -= np.min(materials)
        lame = getMyCoeff(materials , param, op = 'python') 
        sigma = lambda u: fela.sigmaLame(u,lame)
        Vsig = FunctionSpace(mesh, "DG", 0)
        von_Mises = Function(Vsig, name="Stress")
        s = sigma(u) - (1./3)*tr(sigma(u))*Identity(2)
        von_Mises_ = sqrt((3./2)*inner(s, s)) 
        von_Mises.assign(local_project(von_Mises_, Vsig))
        fileResults.write(von_Mises,0.)

    if 'lame' in labels:
        mesh = u.function_space().mesh()
        materials = mesh.subdomains.array().astype('int32')
        materials -= np.min(materials)
        lame = getMyCoeff(materials , param, op = 'python') 
        Vlame = FunctionSpace(mesh, "DG", 0)
        lamb = Function(Vlame, name="lamb")
        mu = Function(Vlame, name="mu")
        lamb.assign(local_project(lame[0], Vlame))
        mu.assign(local_project(lame[1], Vlame))
        
        fileResults.write(lamb,0.)
        fileResults.write(mu,0.)
        
    
    fileResults.close()
    
# Computing errors in stresses
# param = 9*[(lamb0, mu0)] + [(lamb1, mu1)]
# i = Nrefine
# print('computing reference ')    


# Uref = Function(meshList[i].V['u'])
# Uref.assign(u[i])


# sig_ref = Function(VsigRef, name="Stress")
# sig_ref.assign(local_project(sigmaRef(Uref), VsigRef))



# Vsig0_ref = FunctionSpace(meshList[i], "DG", 0)


def exportMeshHDF5_fromGMSH2(gmshMesh = 'mesh.msh', meshFile = 'mesh.xdmf', labels = {'line' : 'faces', 'triangle' : 'regions'}): #'meshTemp2.msh'

    print('hello')
    geometry = meshio.read(gmshMesh) if type(gmshMesh) == type('s') else gmshMesh
    
    print('hello')
    meshFileRad = meshFile[:-5]
    meshio.write(meshFile, meshio.Mesh(points=geometry.points[:,:2], cells={"triangle": geometry.cells_dict['triangle']}))
    
    print('hello')
    
    mesh = meshio.Mesh(points=np.zeros((1,2)), cells={'line': geometry.cells_dict['line']},
                       cell_data={'line': {'faces': geometry.cell_data_dict["gmsh:physical"]['line']}})
    
    meshio.write("{0}_{1}.xdmf".format(meshFileRad,'faces'), mesh)
    
    # f = h5py.File("{0}_{1}.h5".format(meshFileRad,'faces'),'r+')
    # del f['data0']
    # f['data0'] = h5py.ExternalLink(meshFileRad + ".h5", "data0")
    # f.close()
    
    mesh = meshio.Mesh(points=np.zeros((1,2)), cells={"triangle": np.array([[1,2,3]])}, cell_data={'triangle': {'regions': geometry.cell_data_dict["gmsh:physical"]['triangle']}})
    
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
    

def export_XDMF_displacement_sigma(uh, sigma, file):
    mesh = uh.function_space().mesh()
    
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True
    
    uh.rename('u','displacements at nodes')
    file.write(uh, 0.)

    voigt2ten = lambda a: as_tensor(((a[0],a[2]),(a[2],a[1])))
    Vsig = FunctionSpace(mesh, "DG", 0)
    von_Mises = Function(Vsig, name="vonMises")
    sig_xx = Function(Vsig, name="sig_xx")
    sig_yy = Function(Vsig, name="sig_yy")
    sig_xy = Function(Vsig, name="sig_xy")
    
    sig = voigt2ten(sigma(uh))
    s = sig - (1./3)*tr(sig)*Identity(2)
    von_Mises_ = sqrt((3./2)*inner(s, s)) 
    von_Mises.assign(local_project(von_Mises_, Vsig))
    sig_xx.assign(local_project(sig[0,0], Vsig))
    sig_yy.assign(local_project(sig[1,1], Vsig))
    sig_xy.assign(local_project(sig[0,1], Vsig))
    
    file.write(von_Mises,0.)
    file.write(sig_xx,0.)
    file.write(sig_yy,0.)
    file.write(sig_xy,0.)
    
def export_checkpoint_XDMF_displacement_sigma(uh, sigma, file):
    mesh = uh.function_space().mesh()
    
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True
    
    voigt2ten = lambda a: as_tensor(((a[0],a[2]),(a[2],a[1])))
    Vsig = FunctionSpace(mesh, "DG", 0)
    von_Mises = Function(Vsig, name="vonMises")
    sig_xx = Function(Vsig, name="sig_xx")
    sig_yy = Function(Vsig, name="sig_yy")
    sig_xy = Function(Vsig, name="sig_xy")
    
    sig = voigt2ten(sigma(uh))
    s = sig - (1./3)*tr(sig)*Identity(2)
    von_Mises_ = sqrt((3./2)*inner(s, s)) 
    von_Mises.assign(local_project(von_Mises_, Vsig))
    sig_xx.assign(local_project(sig[0,0], Vsig))
    sig_yy.assign(local_project(sig[1,1], Vsig))
    sig_xy.assign(local_project(sig[0,1], Vsig))

    file.write_checkpoint(von_Mises,'vonMises', 0, XDMFFile.Encoding.HDF5)
    file.write_checkpoint(sig_xx,'sig_xx', 0, XDMFFile.Encoding.HDF5, True)
    file.write_checkpoint(sig_yy,'sig_yy', 0, XDMFFile.Encoding.HDF5, True)
    file.write_checkpoint(sig_xy,'sig_xy', 0, XDMFFile.Encoding.HDF5, True)
    
    file.write_checkpoint(uh,'u', 0, XDMFFile.Encoding.HDF5, True)
    
    
    