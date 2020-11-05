
import numpy as np
import scipy.fftpack as spfft

from math import exp
from cmath import exp as cexp

# pulse_forw_fs: forward propagated pulse (frequency-space domain)
# pulse_back_fs: back propagated shot (frequency-space domain)
def extrapAndImaging(ns, nvel, nz, nextrap, nt, nw, nx, \
    dz, w, kx, refVel, vel_model, pulse_forw_fs, pulse_back_fs):
    
    print(" - wavefields shapes:", pulse_forw_fs.shape)

    image = np.zeros(shape=(ns,nz,nx), dtype=np.float32)

    # proagating wavefields for all reference velocities, shots, and pulses
    ref_channel_forw = np.zeros(shape=(ns,nvel,nw,nx), dtype=np.complex64)
    ref_channel_back = np.zeros(shape=(ns,nvel,nw,nx), dtype=np.complex64)

    for l in range(nextrap):

        if l % 10 ==0:
            print("  - depth:",round((l+1)*dz,1))

        # Phase-shift in f-x domain
        for j in range(nw):
            for i in range(nx):
                u = vel_model[l,i]
                k = w[j]/u
                Q1 = cexp( -1j * k * dz )
                Q2 = cexp( +1j * k * dz )
                for s in range(ns):
                    pulse_forw_fs[s,j,i] *= Q1
                    pulse_back_fs[s,j,i] *= Q2
        
        # FFT -> f-kx domain
        for s in range(ns):
            pulse_forw_fs[s,:,:] = spfft.fft(pulse_forw_fs[s,:,:], axis=1)
            pulse_back_fs[s,:,:] = spfft.fft(pulse_back_fs[s,:,:], axis=1)

        # Propagate with all reference velocities in f-kx domain
        for j in range(nw):
            for i in range(nx):
                for n in range(nvel):
                    k = w[j] / refVel[n]
                    if np.abs(k) > np.abs(kx[i]):
                        kz = +np.sqrt(k**2 - kx[i]**2)
                        Q1 = cexp( -1j * (kz - k) * dz )
                        Q2 = cexp( +1j * (kz - k) * dz )
                        for s in range(ns):
                            ref_channel_forw[s,n,j,i] = pulse_forw_fs[s,j,i] * Q1
                            ref_channel_back[s,n,j,i] = pulse_back_fs[s,j,i] * Q2                                
        
        # IFFT -> f-x domain
        for s in range(ns):
            for n in range(nvel):
                ref_channel_forw[s,n,:,:] = spfft.ifft(ref_channel_forw[s,n,:,:], axis=1)
                ref_channel_back[s,n,:,:] = spfft.ifft(ref_channel_back[s,n,:,:], axis=1)

        # Find coefficients for interpolation and then normalize them
        coeff = find_coeff(vel_model[l,:], refVel)
        coeff = norm_coefficients(coeff)

        pulse_forw_fs[:,:,:] = 0; pulse_back_fs[:,:,:] = 0
        # Interpolation
        for s in range(ns):
            for i in range(nx):
                for n in range(nvel):
                    pulse_forw_fs[s,:,i] += coeff[i,n]*ref_channel_forw[s,n,:,i]
                    pulse_back_fs[s,:,i] += coeff[i,n]*ref_channel_back[s,n,:,i]

        # Imaging condition (cross-correlation)
        for s in range(ns):
            image[s,l,:] = np.add.reduce( (pulse_forw_fs[s,:,:] * \
                np.conj(pulse_back_fs[s,:,:]) ).real, axis=0)
    
    return image


def find_coeff(velocities, ref_velocities):
    nx = len(velocities)
    nvel = len(ref_velocities)
    coeff = np.zeros(shape=(nx,nvel), dtype=np.float32)
    e = 0.001 # a small value to avoid division by 0

    for i in range(nx):
        u = velocities[i]
        sum_ = 0.0
        for n in range(nvel):
            ref_u = ref_velocities[n]
            coeff[i,n] = 1.0 / (np.abs(u-ref_u) + e)
    
    return coeff

def norm_coefficients(coeff):
    for i in range(coeff.shape[0]):
        sum_ = 0.0
        for n in range(coeff.shape[1]):
            sum_ += coeff[i,n]
        A = 1.0 / sum_
        for n in range(coeff.shape[1]):
            coeff[i,n] *= A
    
    return coeff
