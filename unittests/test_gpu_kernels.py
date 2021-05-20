
from interface.interface_gpu import interpolation_gpu



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

        interpolation_gpu(1, nf, nx, nref, coeff, refWavefields, finalWavefield)

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

        interpolation_gpu(1, nf, nx, nref, coeff, refWavefields, finalWavefield)

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
