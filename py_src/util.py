import numpy as np
from numba import njit
from math import exp, pi, ceil

@njit
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

@njit
def ricker_wv_map(t_t0):
    fmax = 30 #Hz
    term = pi*pi*fmax*fmax*(t_t0)*(t_t0)
    return (1-2*term)*exp(-term)

def makeRickerWavelet(isx, isz, nt, nx, tj, xi, init_depth, v):
    wavelet = np.zeros((nt,nx), dtype=np.float32)
    t_t0 = np.zeros(nt,dtype=np.float32)

    for sx in isx:
        for i in range(nx):
            r = np.sqrt((init_depth-isz)**2 + (xi[i]-sx)**2)
            t0 = r/v
            t_t0[:] = tj[:]-t0
            wavelet[:,i] += np.asarray(list(map(ricker_wv_map, t_t0)))/np.sqrt(r)
    
    return wavelet
