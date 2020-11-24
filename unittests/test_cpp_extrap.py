
import sys
import os

test_src_path = os.path.dirname(__file__)
sys.path.append(test_src_path + '/../cpp_src')
from interface import *

import numpy as np
from scipy import fftpack
import unittest
import random

sdelta = 1e-5

#--------------------------

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

        PSforw(ns, nf, nx, wf, velSlide, omega, dz)

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

        PSforw(ns, nf, nx, wf, velSlide, omega, dz)

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

        PSforw(ns, nf, nx, wf, velSlide, omega, dz)

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

        PSforw(ns, nf, nx, wf, velSlide, omega, dz)

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

        PSback(ns, nf, nx, wf, velSlide, omega, dz)

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

        PSback(ns, nf, nx, wf, velSlide, omega, dz)

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

        PSback(ns, nf, nx, wf, velSlide, omega, dz)

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

        PSback(ns, nf, nx, wf, velSlide, omega, dz)

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

        PSforw(ns, nf, nx, wf_forw, velSlide, omega, dz)
        PSback(ns, nf, nx, wf_back, velSlide, omega, dz)

        # forward and backward PS apply same transfrmation in real parts
        # and exactly opposites in imaginary parts
        for s in range(ns):
            for j in range(nf):
                for i in range(nx):
                    self.assertAlmostEqual(wf_forw[s,j,i].real, wf_back[s,j,i].real, delta=sdelta)
                    self.assertAlmostEqual(wf_forw[s,j,i].imag, -wf_back[s,j,i].imag, delta=sdelta)

