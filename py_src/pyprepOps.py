
import numpy as np
import cmath
from numpy.linalg import pinv
import time
import util


# FUNCTION SCOPE: Prepare a 2D table of 1D operators for forward-in-time
# propagation. Operators are designed based on a set of k's given in array
# kappa, and given a set of wavenumbers (kx).
# INPUT parameters:
#    - kappa : array with desired k values (must: kappa[0] = 0.0, kappa[N-1]
# be kmax, no negative values)
#    - wavenumbers : array with wavenumbers (represent the positive Fourier 
# coefficients)
#    - dz : depth step (dz > 0)
# OUTPUT:
#    - psOp : "2D" array of ns 1D operators
def forwOperators(kappa, wavenumbers, dz):
    psOp = np.zeros( (len(kappa), len(wavenumbers)), dtype=np.complex64 )
    i=0
    for k in kappa:
        j=0
        for kx in wavenumbers:
            if np.abs(k) >= np.abs(kx):
                kz = np.sqrt( k**2 - kx**2 )
            else:
                kz = -1j*np.sqrt( kx**2 - k**2 )
            psOp[i,j] = cmath.exp( -1j * (kz - k) * dz )
            j += 1
        i += 1
    
    return psOp



# FUNCTION SCOPE: Prepare a 2D table of 1D operators for backward-in-time
# propagation. Operators are designed based on a set of k's given in array
# kappa, and given a set of wavenumbers (kx).
# INPUT parameters:
#    - kappa : array with desired k values (must: kappa[0] = 0.0, kappa[N-1]
# be kmax, no negative values)
#    - wavenumbers : array with wavenumbers (represent the positive Fourier 
# coefficients)
#    - dz : depth step (dz > 0)
# OUTPUT:
#    - psOp : "2D" array of ns 1D operators
def backOperators(kappa, wavenumbers, dz):
    psOp = np.zeros( (len(kappa), len(wavenumbers)), dtype=np.complex64 )
    i=0
    for k in kappa:
        j=0
        for kx in wavenumbers:
            if np.abs(k) >= np.abs(kx):
                kz = np.sqrt( k**2 - kx**2 )
            else:
                kz = +1j*np.sqrt( kx**2 - k**2 )
            psOp[i,j] = cmath.exp( +1j * (kz - k) * dz )
            j += 1
        i += 1

    return psOp



# FUNCTION SCOPE: Forward Phase-Shift in the frequency-space domain for 
#    a set of ns wavefields/shot-records.
# INPUT parameters:
#    - ns : number of shots
#    - nf : number of frequencies
#    - nx : number of lateral positions
# OUTPUT:
#    - wavefields : "3D" array of ns 2D wavefields to-be-shifted in-place.
def phase_shift_forw(ns, nf, nx, wavefields, velSlide, omega, dz):

    shotSz = nf*nx # size of shot records

    for j in range(nf):
        for i in range(nx):
            
            k = omega[j] / velSlide[i]
            term = cexp ( -1j * k * dz )
            for s in range(ns):
                wavefields[s,j,i] *= term



# FUNCTION SCOPE: Backward Phase-Shift in the frequency-space domain for 
#    a set of ns wavefields/shot-records.
# INPUT parameters:
#    - ns : number of shots
#    - nf : number of frequencies
#    - nx : number of lateral positions
# OUTPUT:
#    - wavefields : "3D" array of ns 2D wavefields to-be-shifted in-place.
def phase_shift_back(ns, nf, nx, wavefields, velSlide, omega, dz):

    shotSz = nf*nx # size of shot records

    for j in range(nf):
        for i in range(nx):
            
            k = omega[j] / velSlide[i]
            term = cexp ( +1j * k * dz )
            for s in range(ns):
                wavefields[s,j,i] *= term



