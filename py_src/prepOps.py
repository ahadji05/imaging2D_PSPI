
import numpy as np
import cmath
from numpy.linalg import pinv
import time
import numba
import util

#   ------------------------------------------

def makeForwPSoperators(kappa, wavenumbers, dz):
    psOp = np.zeros( (len(kappa), len(wavenumbers)), dtype=np.complex64 )
    i=0
    for k in kappa:
        j=0
        for kx in wavenumbers:
            if np.abs(k) >= np.abs(kx):
                kz = np.sqrt( k**2 - kx**2 )
            else:
                kz = -1j*np.sqrt( kx**2 - k**2 )
            psOp[i,j] = cmath.exp( -1j*kz*dz )
            j += 1
        i += 1
    
    return psOp

#   ------------------------------------------

def makeBackPSoperators(kappa, wavenumbers, dz):
    psOp = np.zeros( (len(kappa), len(wavenumbers)), dtype=np.complex64 )
    i=0
    for k in kappa:
        j=0
        for kx in wavenumbers:
            if np.abs(k) >= np.abs(kx):
                kz = np.sqrt( k**2 - kx**2 )
            else:
                kz = +1j*np.sqrt( kx**2 - k**2 )
            psOp[i,j] = cmath.exp( +1j*kz*dz )
            j += 1
        i += 1

    return psOp
    