"""
@author: felipe

This is a wrapper for the h5py library. 
The main idea is to create a numpy-like access to hdf5 files.

Obs: default compression is not necessarily optimised but provides a practical
way of saving datasets in a compressed form 
"""


import h5py
import numpy as np
from functools import partial
import os
# import sys; sys.__stdout__ = sys.__stderr__ # workaround of a bug

# defaultCompression = {'dtype' : 'f8',  'compression' : "gzip", 
#                            'compression_opts' : 1, 'shuffle' : False}

defaultCompression = {'dtype' : 'f8',  'compression' : "gzip", 
                      'compression_opts' : 0, 'shuffle' : False, 
                      'chunks': (5,100)} # dummy chunk to be specified next

toChunk = lambda a: tuple([1] + [a[i] for i in range(1,len(a))])

# load hd5 file: returns a specific dataset if provided label, 
# or a list with all datasets otherwise
def loadhd5(filename, label):
    
    with h5py.File(filename, 'r') as f:
        if(type(label) == type('l')):
            X = np.array(f[label])
        else:
            X = []
            for l in label:
                X.append(np.array(f[l]))


    return X

# similar to loadhd5 but the file is kept open and its handler is returned
def loadhd5_openFile(filename, label, mode = 'r'):
    f = h5py.File(filename, mode)
    if(type(label) == type('l')):
        X = f[label]
    else:
        X = []
        for l in label:
            X.append(f[l])

    return X, f

# save a hd5 file provided given dataset (s) X and label (s)
def savehd5(filename, X,label, mode):
    with h5py.File(filename, mode) as f:
        if(type(label) == type('l')):
            defaultCompression['chunks'] = toChunk(X.shape)
            f.create_dataset(label, data = X, **defaultCompression)
        else:
            for i,l in enumerate(label):
                defaultCompression['chunks'] = toChunk(X[i].shape)
                f.create_dataset(l, data = X[i], **defaultCompression)

# Add a new dataset X with some label in a existing file
def addDataset(f,X, label):
    defaultCompression['chunks'] = toChunk(X.shape)
    g = h5py.File(f, 'a') if (type(f) == type('l')) else f
    g.create_dataset(label, data = X , **defaultCompression)

# Opens a new file with dataset (or datasets) filled with zeroes of given 
# dimension (s)
def zeros_openFile(filename, shape, label, mode = 'w-'):
    f = h5py.File(filename, mode)
    
    if(type(label) == type('l')):
        defaultCompression['chunks'] = toChunk(shape)
        f.create_dataset(label, shape =  shape , **defaultCompression)
        X = f[label]
    else:
        X = []
        for i,l in enumerate(label):
            defaultCompression['chunks'] = toChunk(shape[i])
            f.create_dataset(l, shape =  shape[i] , **defaultCompression)
            X.append(f[l])
    
    return X, f


# load a dataset from a file and save it into other
def extractDataset(filenameIn, filenameOut, label, mode):
    X = loadhd5(filenameIn, label)
    savehd5(filenameOut,X,label,mode)

# Merge datasets of list of files into a single hdf5 file 
def merge(filenameInputs, filenameOutput, InputLabels, OutputLabels , 
          axis = 0, mode = 'w-'):
    

    with h5py.File(filenameOutput, mode) as ff:
        for li,lo in zip(InputLabels,OutputLabels):  
            d = []
            for fn in filenameInputs:
                with h5py.File(fn,'r') as f:
                    if('attrs' in li):
                        s = li.split('/attrs/')
                        d.append(np.array([f[s[0]].attrs[s[1]]]))
                    else:
                        d.append(np.array(f[li])) 

            ff.create_dataset(lo, data = np.concatenate(d, axis = axis), 
                              compression = 'gzip', 
                              compression_opts = 1, shuffle = True)


# Split datasets of list into multiples hdf5 files, according to some indexing 
def split(filenameInput, indexesOutput, labels, mode = 'w-'):
    
    n = len(indexesOutput)
    
    os.system("mkdir " + filenameInput[:-4] + "_split")
    
    ddinputs = loadhd5(filenameInput, labels)
    
    for i in range(n): 
        filenameOutput = filenameInput[:-4] + "_split/part_%d.hd5"%i 
        savehd5(filenameOutput, [d[indexesOutput[i]] for d in ddinputs], labels, mode)

# transforms a .txt file into a hdf5 file
def txt2hd5(filenameIn, filenameOut, label, reshapeLast = [False,0], mode = 'w-'):
    with h5py.File(filenameOut, mode) as f:
        if(type(label) == type('l')):
            data = np.loadtxt(filenameIn)
            if(reshapeLast[0]):
                data = np.reshape(data,(data.shape[0],-1,reshapeLast[1]))
            defaultCompression['chunks'] = toChunk(data.shape)
            f.create_dataset(label, data = data, **defaultCompression)
        else:
            for i,l in enumerate(label):
                data = np.loadtxt(filenameIn[i])
                if(reshapeLast[i][0]):
                    data = np.reshape(data,(data.shape[0],-1,reshapeLast[i][1]))
                    
                defaultCompression['chunks'] = toChunk(data.shape)
                f.create_dataset(l, data = data, **defaultCompression)
       
        

# check the existence of a dataset label into a hdf5 file
def checkExistenceDataset(filename, label):
    with h5py.File(filename, 'r') as f:
        return label in f.keys()
            

# decide which file loader is used is used accordingly to the file extension
# Ex: np.loadtxt or loadhd5 
def getLoadfunc(namefile, label):
       
    if(namefile[-3:]=='hd5'):
        loadfunc = partial(loadhd5, label = label)
    else:
        loadfunc = np.loadtxt
            
    return loadfunc

# Use the generic load function given by getLoadfunc and effectively loads 
# the file
genericLoadfile = lambda x, y : getLoadfunc(x,y)(x)
