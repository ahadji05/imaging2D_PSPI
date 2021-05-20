
from interface.interface_cpu import interpolation_cpu
from interface.interface_cpu import imaging_conditions_cpu
from interface.interface_cpu import phase_shifts_forw_cpu
from interface.interface_cpu import phase_shifts_back_cpu
from interface.interface_cpu import extrap_ref_wavefields_cpu

import numpy as np
import unittest
import random

sdelta = 1e-5

#--------------------------

class Test_Interpolation_coefficients(unittest.TestCase):

    def test_interpolation_1_frequency(self):

        # assuming we have 3 ref. velocities and 4 lateral positions
        # => table coeff will be of shape(4,3)
        coeff = np.array([[0.2,0.5,0.3], [0.25, 0.25, 0.5], \
            [0.3,0.4,0.3], [0.15,0.20,0.65]], dtype=np.float32)

        # since we have 3 ref. velocities we need to assume that we have 3 wavefields
        refWavefields = np.array([[1+0j,0+1j,0+0j,0+0j], \
            [1+1j,3-1j,1+1j,1+1j], [2+2j,2+2j,2+2j,2+0j]], dtype=np.complex64)

        finalWavefield = np.zeros(4, dtype=np.complex64)

        nf = 1 # assuming all wavfields belong to the same frequency
        nx = 4 # because wavefields have 4 values in the contiguous direction.
        nref = 3 # because coefficients have 3 values per position

        interpolation_cpu(1, nf, nx, nref, coeff, refWavefields, finalWavefield)

        # check all values one-by-one
        self.assertAlmostEqual(finalWavefield[0].real , 1.30, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[1].real , 1.75, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[2].real , 1.00, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[3].real , 1.50, delta=sdelta)

        self.assertAlmostEqual(finalWavefield[0].imag , 1.10, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[1].imag , 1.00, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[2].imag , 1.00, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[3].imag , 0.20, delta=sdelta)

    def test_interpolation_2_frequency(self):

        nx = 4
        nref = 3
        
        # assuming we have 3 ref. velocities and 4 lateral positions
        # => table coeff will be of shape(4,3)
        coeff = np.array([[0.2,0.5,0.3], [0.25, 0.25, 0.5], \
            [0.3,0.4,0.3], [0.15,0.20,0.65]], dtype=np.float32)

        nf = 2

        # since we have 3 ref. velocities we need to assume that we have 3 wavefields
        # per frequency => refwavefields.shape = (nref, nf, nx)
        refWavefields = np.array([ \
            [[1+0j,0+1j,0+0j,0+0j], [1+0j,0+1j,0+0j,0+0j]], \
            [[1+1j,3-1j,1+1j,1+1j], [1+1j,3-1j,1+1j,1+1j]], \
            [[2+2j,2+2j,2+2j,2+0j], [2+2j,2+2j,2+2j,2+0j]] \
            ], dtype=np.complex64)

        finalWavefield = np.zeros((nf,nx), dtype=np.complex64)        

        interpolation_cpu(1, nf, nx, nref, coeff, refWavefields, finalWavefield)

        # # since all wavefields over frequencies are the same, check that at the end they
        # # have as expected same values
        for j in range(nf):
            self.assertAlmostEqual(finalWavefield[j,0].real , 1.30, delta=sdelta)
            self.assertAlmostEqual(finalWavefield[j,1].real , 1.75, delta=sdelta)
            self.assertAlmostEqual(finalWavefield[j,2].real , 1.00, delta=sdelta)
            self.assertAlmostEqual(finalWavefield[j,3].real , 1.50, delta=sdelta)

            self.assertAlmostEqual(finalWavefield[j,0].imag , 1.10, delta=sdelta)
            self.assertAlmostEqual(finalWavefield[j,1].imag , 1.00, delta=sdelta)
            self.assertAlmostEqual(finalWavefield[j,2].imag , 1.00, delta=sdelta)
            self.assertAlmostEqual(finalWavefield[j,3].imag , 0.20, delta=sdelta)

#-------------------------------------------------------------------------

class Test_Imaging_Condition(unittest.TestCase):
    
    def test_imag_condition_cpu_1(self):

        ns = 1 # 1 shot
        nx = 5 
        nf = 3
        shotSz = nx*nf #size of wavefield
        imageSz = nx # the image size is just one depth slide

        back = np.zeros((nf,nx), dtype=np.complex64)
        forw = np.zeros_like(back)

        forw[:,0] = 1-1j; forw[:,2] = 2.0+1.0j
        np.copyto(back, forw)
        forw[2,3] = 2-2.9j; back[2,3] = 1.1-0.5j

        image = np.zeros(nx, dtype=np.float32)
        imaging_conditions_cpu(ns, shotSz, nx, nf, forw, back, imageSz, 0, image)

        self.assertAlmostEqual(image[0], 6.0, delta=sdelta)
        self.assertAlmostEqual(image[2], 15.0, delta=sdelta)
        self.assertAlmostEqual(image[3], (forw[2,3]*np.conj(back[2,3])).real, delta=sdelta)


    def test_imag_condition_cpu_2(self):

        ns = 1 # 1 shot
        nx = 5 
        nf = 3
        shotSz = nx*nf #size of wavefield
        imageSz = nx # the image size is just one depth slide

        back = np.zeros((nf,nx), dtype=np.complex64)
        forw = np.zeros_like(back)

        forw[:,0] = 1-1j; forw[:,2] = 2.0+1.0j
        np.copyto(back, forw)
        forw[2,3] = 2-2.9j; back[2,3] = 1.1-0.5j

        image = np.zeros((10,nx), dtype=np.float32)
        imaging_conditions_cpu(ns, shotSz, nx, nf, forw, back, imageSz, 5, image)

        self.assertAlmostEqual(image[5,0], 6.0, delta=sdelta)
        self.assertAlmostEqual(image[5,2], 15.0, delta=sdelta)
        self.assertAlmostEqual(image[5,3], (forw[2,3]*np.conj(back[2,3])).real, delta=sdelta)


    def test_imag_condition_cpu_3(self):

        nz = 10
        ns = 10 # 1 shot
        nx = 5 
        nf = 3
        shotSz = nx*nf #size of wavefield
        imageSz = nz*nx # image size (per shot)

        back = np.zeros((ns,nf,nx), dtype=np.complex64)
        forw = np.zeros_like(back)
        image = np.zeros((ns,nz,nx), dtype=np.float32)

        sIdx = 4
        depthIdx = 5

        forw[sIdx,:,0] = 1-1j; forw[sIdx,:,2] = 2+1j
        np.copyto(back, forw)
        forw[4,2,3] = 2-2.9j; back[4,2,3] = 1.1-0.5j

        imaging_conditions_cpu(ns, shotSz, nx, nf, forw, back, imageSz, depthIdx, image)

        self.assertAlmostEqual(image[sIdx,depthIdx,0], 6.0, delta=sdelta)
        self.assertAlmostEqual(image[sIdx,depthIdx,2], 15.0, delta=sdelta)
        self.assertAlmostEqual(image[sIdx,depthIdx,3], (forw[sIdx,2,3]*np.conj(back[sIdx,2,3])).real, delta=sdelta)

        for i in range(ns):
            for l in range(nz):
                if i != sIdx and l != depthIdx:
                    for ix in range(nx):
                        self.assertAlmostEqual(image[i,l,ix], 0.0, delta=sdelta)


#-------------------------------------------------------------------------

class Test_extrap(unittest.TestCase):
    
    def test_PSforw_all_shots_same_ops_1(self):

        ns = 10 # number of shot records

        velSlide = np.array([1000.0, 1500.0, 1250.0, 1250.0], dtype=np.float32)
        nx = len(velSlide) # number of later positions
        
        omega = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        nf = len(omega) # number of frequencies

        dz = 5.0

        # phase-shift are applied straight to wavefield. in order to test the
        # operation we use wavefields with only values 1+0j only.
        wf = np.ones((ns,nf,nx), dtype=np.complex64)

        phase_shifts_forw_cpu(ns, nf, nx, wf, velSlide, omega, dz)

        # check that all shots have the same values Q
        for j in range(nf):
            for i in range(nx):
                Q = dz*omega[j] / velSlide[i]
                for s in range(ns):
                    self.assertAlmostEqual(wf[s,j,i].real, np.cos(Q), delta=sdelta)
                    self.assertAlmostEqual(wf[s,j,i].imag, -np.sin(Q), delta=sdelta)


    def test_PSforw_all_shots_same_ops_2(self):

        ns = 10 # number of shot records

        velSlide = np.array([1000.0, 1500.0, 1250.0, 1250.0], dtype=np.float32)
        nx = len(velSlide) # number of later positions
        
        omega = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        nf = len(omega) # number of frequencies

        dz = 5.0

        # phase-shift are applied straight to wavefield. in order to test the
        # operation we use wavefields with only values 1+0j only.
        wf = np.ones((ns,nf,nx), dtype=np.complex64)
        wf *= 2.5

        phase_shifts_forw_cpu(ns, nf, nx, wf, velSlide, omega, dz)

        # check that all shots have the same values Q
        for j in range(nf):
            for i in range(nx):
                Q = dz*omega[j] / velSlide[i]
                for s in range(ns):
                    self.assertAlmostEqual(wf[s,j,i].real, 2.5*np.cos(Q), delta=sdelta)
                    self.assertAlmostEqual(wf[s,j,i].imag, -2.5*np.sin(Q), delta=sdelta)


    def test_PSforw_same_velocities_same_ops(self):

        ns = 10 # number of shot records

        velSlide = np.array([1000.0, 1500.0, 1250.0, 1250.0], dtype=np.float32)
        nx = len(velSlide) # number of later positions
        
        omega = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        nf = len(omega) # number of frequencies

        dz = 15.0

        # phase-shift are applied straight to wavefield. in order to test the
        # operation we use wavefields with only values 1+0j only.
        wf = np.ones((ns,nf,nx), dtype=np.complex64)

        phase_shifts_forw_cpu(ns, nf, nx, wf, velSlide, omega, dz)

        # lateral positions 2 & 3 have the same velocities
        for s in range(ns):
            for j in range(nf):
                self.assertAlmostEqual(wf[s,j,2].real, wf[s,j,3].real, delta=sdelta)
                self.assertAlmostEqual(wf[s,j,2].imag, wf[s,j,3].imag, delta=sdelta)


    def test_PSforw_same_omegas_same_ops(self):

        ns = 2 # number of shot records

        velSlide = np.array([1000.0, 1500.0, 1250.0, 1250.0], dtype=np.float32)
        nx = len(velSlide) # number of later positions
        
        omega = np.array([0.0, 0.1, 0.1, 0.3, 0.4], dtype=np.float32)
        nf = len(omega) # number of frequencies

        dz = 15.0

        # phase-shift are applied straight to wavefield. in order to test the
        # operation we use wavefields with only values 1+0j only.
        wf = np.ones((ns,nf,nx), dtype=np.complex64)

        phase_shifts_forw_cpu(ns, nf, nx, wf, velSlide, omega, dz)

        # omega 1 & 2 have the same values
        for s in range(ns):
            for i in range(nx):
                self.assertAlmostEqual(wf[s,1,i].real, wf[s,2,i].real, delta=sdelta)
                self.assertAlmostEqual(wf[s,1,i].imag, wf[s,2,i].imag, delta=sdelta)


    def test_PSback_all_shots_same_ops_1(self):

        ns = 15 # number of shot records

        velSlide = np.array([800.0, 750.0, 250.0, 250.0], dtype=np.float32)
        nx = len(velSlide) # number of later positions
        
        omega = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        nf = len(omega) # number of frequencies

        dz = 10.0

        # phase-shift are applied straight to wavefield. in order to test the
        # operation we use wavefields with only values 1+0j only.
        wf = np.ones((ns,nf,nx), dtype=np.complex64)

        phase_shifts_back_cpu(ns, nf, nx, wf, velSlide, omega, dz)

        # check that all shots have the same values Q
        for j in range(nf):
            for i in range(nx):
                Q = dz*omega[j] / velSlide[i]
                for s in range(ns):
                    self.assertAlmostEqual(wf[s,j,i].real, np.cos(Q), delta=sdelta)
                    self.assertAlmostEqual(wf[s,j,i].imag, np.sin(Q), delta=sdelta)


    def test_PSback_all_shots_same_ops_2(self):

        ns = 15 # number of shot records

        velSlide = np.array([800.0, 750.0, 250.0, 250.0], dtype=np.float32)
        nx = len(velSlide) # number of later positions
        
        omega = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        nf = len(omega) # number of frequencies

        dz = 10.0

        # phase-shift are applied straight to wavefield. in order to test the
        # operation we use wavefields with only values 1+0j only.
        wf = np.ones((ns,nf,nx), dtype=np.complex64)
        wf *= -2.0

        phase_shifts_back_cpu(ns, nf, nx, wf, velSlide, omega, dz)

        # check that all shots have the same values Q
        for j in range(nf):
            for i in range(nx):
                Q = dz*omega[j] / velSlide[i]
                for s in range(ns):
                    self.assertAlmostEqual(wf[s,j,i].real, -2.0*np.cos(Q), delta=sdelta)
                    self.assertAlmostEqual(wf[s,j,i].imag, -2.0*np.sin(Q), delta=sdelta)


    def test_PSback_same_velocities_same_ops(self):

        ns = 200 # number of shot records

        velSlide = np.array([750.0, 250.0, 750.0, 250.0], dtype=np.float32)
        nx = len(velSlide) # number of later positions
        
        omega = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        nf = len(omega) # number of frequencies

        dz = 7.5

        # phase-shift are applied straight to wavefield. in order to test the
        # operation we use wavefields with only values 1+0j only.
        wf = np.ones((ns,nf,nx), dtype=np.complex64)

        phase_shifts_back_cpu(ns, nf, nx, wf, velSlide, omega, dz)

        # lateral positions 1 & 3 have the same velocities
        for s in range(ns):
            for j in range(nf):
                self.assertAlmostEqual(wf[s,j,1].real, wf[s,j,3].real, delta=sdelta)
                self.assertAlmostEqual(wf[s,j,1].imag, wf[s,j,3].imag, delta=sdelta)

        # lateral positions 0 & 2 have the same velocities
        for s in range(ns):
            for j in range(nf):
                self.assertAlmostEqual(wf[s,j,0].real, wf[s,j,2].real, delta=sdelta)
                self.assertAlmostEqual(wf[s,j,0].imag, wf[s,j,2].imag, delta=sdelta)


    def test_PSback_same_omegas_same_ops(self):

        ns = 8 # number of shot records

        velSlide = np.array([750.0, 250.0, 750.0, 250.0], dtype=np.float32)
        nx = len(velSlide) # number of later positions
        
        omega = np.array([0.0, 0.1, 0.2, 0.1, 0.4], dtype=np.float32)
        nf = len(omega) # number of frequencies

        dz = 17.5

        # phase-shift are applied straight to wavefield. in order to test the
        # operation we use wavefields with only values 1+0j only.
        wf = np.ones((ns,nf,nx), dtype=np.complex64)

        phase_shifts_back_cpu(ns, nf, nx, wf, velSlide, omega, dz)

        # omega 1 & 3 have the same values
        for s in range(ns):
            for i in range(nx):
                self.assertAlmostEqual(wf[s,1,i].real, wf[s,3,i].real, delta=sdelta)
                self.assertAlmostEqual(wf[s,1,i].imag, wf[s,3,i].imag, delta=sdelta)


    def test_compare_PSback_and_PSforw(self):

        ns = 8 # number of shot records

        velSlide = np.array([1000.0, 2000.0, 5000.0, 2500.0], dtype=np.float32)
        nx = len(velSlide) # number of later positions
        
        omega = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        nf = len(omega) # number of frequencies

        dz = 17.5

        # phase-shift are applied straight to wavefield. in order to test the
        # operation we use wavefields with only values 1+0j only.
        wf_forw = np.ones((ns,nf,nx), dtype=np.complex64)
        wf_back = np.ones((ns,nf,nx), dtype=np.complex64)

        phase_shifts_forw_cpu(ns, nf, nx, wf_forw, velSlide, omega, dz)
        phase_shifts_back_cpu(ns, nf, nx, wf_back, velSlide, omega, dz)

        # forward and backward PS apply same transfrmation in real parts
        # and exactly opposites in imaginary parts
        for s in range(ns):
            for j in range(nf):
                for i in range(nx):
                    self.assertAlmostEqual(wf_forw[s,j,i].real, wf_back[s,j,i].real, delta=sdelta)
                    self.assertAlmostEqual(wf_forw[s,j,i].imag, -wf_back[s,j,i].imag, delta=sdelta)


