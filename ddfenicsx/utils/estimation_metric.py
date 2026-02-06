#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:35:15 2024

@author: felipe
"""

import numpy as np
from numpy import linalg as la
import copy
import scipy
from sklearn.decomposition import PCA
import ddfenicsx as dd
import fetricksx as ft

def check_positiveness(C, C_default = None):

    if(not isPD(C)):
        print("estimation is not PD --> using last default")
        print("eigenvalues:")
        print(np.linalg.eig(C)[0])
        if(type(C_default) != type(None)):
            return copy.deepcopy(C_default) # using the last one
        else:
            print("a default value for C should be provided")
            input()
    else:
        return C
    

def get_estimate_C_method(method):

    dict_method = {'PCA': estimateC_PCA,
                   'eigendecomp': estimateC_eigen_decomposition,
                   'LSQ': estimateC_leastsquares,
                   'sylvester': estimateC_sylvester,
                   'sylvester_cov': estimateC_sylvester_cov,
                   'sylvester_cov_C': estimateC_sylvester_cov_C,
                   'sylvester_cov_Cinv':estimateC_sylvester_cov_Cinv,
                   'sylvester_C': estimateC_sylvester,
                   'sylvester_Cinv': estimateC_sylvester_Cinv,
                   'sylvester_C_isotropy': estimateC_sylvester_C_isotropy}

    return dict_method[method]
    

def estimateC_PCA(DB):
    strain_dim = DB.shape[-1]
    pca = PCA(n_components = strain_dim)
    pca.fit(DB.reshape((-1,2*strain_dim)))
    
    Cest = pca.components_[:,strain_dim:]@np.linalg.inv(pca.components_[:,:strain_dim])    
    
    C = 0.5*(Cest + Cest.T) 

    return C


def estimateC_eigen_decomposition(DB):
    strain_dim = DB.shape[-1]
    
    Corr = DB.reshape((-1,2*strain_dim)).T@DB.reshape((-1,2*strain_dim))
    sig, U = np.linalg.eigh(Corr)

    asort = np.argsort(sig)
    sig = sig[asort[::-1]]
    U = U[:,asort[::-1]]
    
    Cest = U[strain_dim:2*strain_dim,:strain_dim]@np.linalg.inv(U[:strain_dim,:strain_dim])    
    
    C = 0.5*(Cest + Cest.T) 
    
    return C


def estimateC_leastsquares(DB):
    Corr_EE = DB[:,0,:].T @ DB[:,0,:]
    Corr_SE = DB[:,1,:].T @ DB[:,0,:]
    Corr_EE = 0.5*(Corr_EE + Corr_EE.T)
    Corr_SE = 0.5*(Corr_SE + Corr_SE.T)
    C = Corr_SE@np.linalg.inv(Corr_EE)
    C = 0.5*(C + C.T) 

    return C

# Sylvester equation is found when performing a leastsquares imposing symmetry for C

def estimateC_sylvester_C(DB):
    Corr_EE = DB[:,0,:].T @ DB[:,0,:]
    Corr_SE = DB[:,1,:].T @ DB[:,0,:]
    
    # AX + BX = Q ==> Corr_EE*C + C*Corr_EE = (Corr_SE + Corr_SE.T) 
    C = scipy.linalg.solve_sylvester(Corr_EE, Corr_EE, Corr_SE + Corr_SE.T)

    return C



def estimateC_sylvester_C_isotropy(DB):
    Corr_EE = DB[:,0,:].T @ DB[:,0,:]
    Corr_SE = DB[:,1,:].T @ DB[:,0,:]
    
    # AX + BX = Q ==> Corr_EE*C + C*Corr_EE = (Corr_SE + Corr_SE.T) 
    C = scipy.linalg.solve_sylvester(Corr_EE, Corr_EE, Corr_SE + Corr_SE.T)

    n = C.shape[0]
    Id_mandel = ft.conv2d.Id_mandel_np
    Basis = [ np.outer(Id_mandel, Id_mandel).flatten(), np.eye(n).flatten()]
    A = np.array([[np.dot(bi,bj) for bj in Basis] for bi in Basis])
    b = np.array([np.dot(C.flatten(), bi) for bi in Basis])
    x = np.linalg.solve(A,b)


    return (x[0]*Basis[0] +  x[1]*Basis[1]).reshape((n,n))

# Sylvester equation is found when performing a leastsquares imposing symmetry for Cinv
def estimateC_sylvester_Cinv(DB):
    Corr_SS = DB[:,1,:].T @ DB[:,1,:]
    Corr_ES = DB[:,0,:].T @ DB[:,1,:]
    
    # AX + BX = Q ==> Corr_SS*Cinv + Cinv*Corr_SS = (Corr_ES + Corr_ES.T) 
    Cinv = scipy.linalg.solve_sylvester(Corr_SS, Corr_SS, Corr_ES + Corr_ES.T)

    return np.linalg.inv(Cinv)

# Average sylvester for C and Cinv
def estimateC_sylvester(DB):
    return 0.5*( estimateC_sylvester_C(DB) + estimateC_sylvester_Cinv(DB))


# Sylvester equation is found when performing a leastsquares imposing symmetry for C

def estimateC_sylvester_cov(DB):
    return 0.5*( estimateC_sylvester_cov_C(DB) + estimateC_sylvester_cov_Cinv(DB))

def estimateC_sylvester_cov_C(DB):
    n_strain = DB.shape[-1]
    cov = np.cov(DB[:,0,:].T, DB[:,1,:].T)
    Corr_EE = cov[0:n_strain, 0:n_strain]
    Corr_SE = cov[n_strain : 2*n_strain, 0:n_strain]
    
    # AX + BX = Q ==> Corr_EE*C + C*Corr_EE = (Corr_SE + Corr_SE.T) 
    C = scipy.linalg.solve_sylvester(Corr_EE, Corr_EE, Corr_SE + Corr_SE.T)

    return C

def estimateC_sylvester_cov_Cinv(DB):
    n_strain = DB.shape[-1]
    cov = np.cov(DB[:,0,:].T, DB[:,1,:].T)
    Corr_SE = cov[n_strain : 2*n_strain, 0:n_strain]
    Corr_SS = cov[n_strain : 2*n_strain, n_strain : 2*n_strain]
    
    # AX + BX = Q ==> Corr_EE*C + C*Corr_EE = (Corr_SE + Corr_SE.T) 
    Cinv = scipy.linalg.solve_sylvester(Corr_SS, Corr_SS, Corr_SE + Corr_SE.T)

    return np.linalg.inv(Cinv)




# alternative method
def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0
    
    alpha = 10.0
    beta = np.max(eigval)/(alpha - 1)

    return eigvec.dot(np.diag(eigval + beta)).dot(eigvec.T)

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


# if __name__ == '__main__':
#     import numpy as np
#     for i in range(10):
#         for j in range(2, 100):
#             A = np.random.randn(j, j)
#             B = nearestPD(A)
#             assert (isPD(B))
#     print('unit test passed!')