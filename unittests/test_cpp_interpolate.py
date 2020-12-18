
import sys
import os

test_src_path = os.path.dirname(__file__)
sys.path.append(test_src_path + '/../cpp_src')
from interface import *

import numpy as np
import unittest
import random

sdelta = 1e-5

#--------------------------

class Test_Interpolation_coefficients(unittest.TestCase):
    
    def test_find_and_norm_coeff(self):
        
        # define exact velocities in slide
        nx = 3
        nxVels = np.zeros(nx, dtype=np.float32)
        nxVels[0] = 1250; nxVels[1] = 2000; nxVels[2] = 2500 
        
        # define reference velocities
        nref = 4
        refVels = np.zeros(nref, dtype=np.float32)
        refVels[0] = 1000; refVels[1] = 2000
        refVels[2] = 3000; refVels[3] = 4000

        # output coefficients
        coeff = np.zeros((nx, nref), dtype=np.float32)

        # calculate non-normalized coefficients
        find_coeff(nx, nxVels, nref, refVels, coeff)
        
        # examine all coefficient one-by-one
        self.assertAlmostEqual(coeff[0,0], 1.0/250.0 ,delta=sdelta)
        self.assertAlmostEqual(coeff[0,1], 1.0/750.0 ,delta=sdelta)
        self.assertAlmostEqual(coeff[0,2], 1.0/1750.0 ,delta=sdelta)
        self.assertAlmostEqual(coeff[0,3], 1.0/2750.0 ,delta=sdelta)

        self.assertAlmostEqual(coeff[1,0], 1.0/1000.0 ,delta=sdelta)
        self.assertAlmostEqual(coeff[1,1], 1.0/0.0001 ,delta=sdelta) #peak here - see code
        self.assertAlmostEqual(coeff[1,2], 1.0/1000.0 ,delta=sdelta)
        self.assertAlmostEqual(coeff[1,3], 1.0/2000.0 ,delta=sdelta)

        self.assertAlmostEqual(coeff[2,0], 1.0/1500.0 ,delta=sdelta)
        self.assertAlmostEqual(coeff[2,1], 1.0/500.0 ,delta=sdelta)
        self.assertAlmostEqual(coeff[2,2], 1.0/500.0 ,delta=sdelta)
        self.assertAlmostEqual(coeff[2,3], 1.0/1500.0 ,delta=sdelta)

        # normalize coefficients
        norm_coeff(nx, nref, coeff)

        # check that the coefficients of each position sum to 1.0
        sum_0 = np.sum(coeff[0,:])
        self.assertAlmostEqual(sum_0, 1.0, delta=sdelta)
        sum_1 = np.sum(coeff[1,:])
        self.assertAlmostEqual(sum_1, 1.0, delta=sdelta)
        sum_2 = np.sum(coeff[2,:])
        self.assertAlmostEqual(sum_2, 1.0, delta=sdelta)

        # check that no negative coefficients are produced
        for ix in range(nx):
            for n in range(nref):
                self.assertGreater(coeff[ix,n], 0.0)
        
        # some final specific to last lateral velocity checks
        self.assertAlmostEqual(coeff[2,0], coeff[2,3], delta=sdelta)
        self.assertAlmostEqual(coeff[2,1], coeff[2,2], delta=sdelta)
        self.assertAlmostEqual(coeff[2,1], 3.0*coeff[2,0], delta=sdelta)
        self.assertAlmostEqual(coeff[2,2], 3.0*coeff[2,3], delta=sdelta)

    

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

        interpolation(1, nf, nx, nref, coeff, refWavefields, finalWavefield)

        # check all values one-by-one
        self.assertAlmostEqual(finalWavefield[0].real , 1.30, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[1].real , 1.75, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[2].real , 1.00, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[3].real , 1.50, delta=sdelta)

        self.assertAlmostEqual(finalWavefield[0].imag , 1.10, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[1].imag , 1.00, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[2].imag , 1.00, delta=sdelta)
        self.assertAlmostEqual(finalWavefield[3].imag , 0.20, delta=sdelta)


    def test_interpolation_3_frequency(self):

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

        interpolation(1, nf, nx, nref, coeff, refWavefields, finalWavefield)

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
