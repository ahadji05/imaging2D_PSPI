
import sys
import os

test_src_path = os.path.dirname(__file__)
sys.path.append(test_src_path + '/../cpp_src')
from interface import *

sys.path.append(test_src_path + '/../py_src')
from pyprepOps import forwOperators, backOperators

import numpy as np
from scipy import fftpack
import unittest
import random

from math import pi

sdelta = 1e-5

#--------------------------

class Test_prepOps(unittest.TestCase):

    def test_class_f_kx_operators_back(self):
        
        Nk = 200; kmax = 0.1785
        k = np.ones(Nk, dtype=np.float32)

        dz = 10.0
        optype = 'b'

        Nkx = 151
        
        xmax = 0; xmin = 2000
        dkx = 2*pi*1/float( xmax - xmin )
        kx = np.fft.fftfreq(Nkx)*Nkx*dkx
        kx = kx.astype(np.float32)
        vals = np.zeros((Nk,Nkx), dtype=np.complex64)

        # run c++ code
        test_class_prepOps(k, kx, vals, Nk, kmax, Nkx, dz, optype)

        # using k and kx as returned from the c++ code, run the python code
        testBack = backOperators(k, kx, dz)

        # compare the operators returned from c++ program with those returned
        # from the python.
        for i in range(Nk):
            for j in range(Nkx):
                self.assertAlmostEqual(vals[i,j].real, testBack[i,j].real , delta=sdelta)
                self.assertAlmostEqual(vals[i,j].imag, testBack[i,j].imag , delta=sdelta)

                
    def test_chooseOperatorIndex_1(self):

        N = 10
        a = np.linspace(0.0, 2.0, N, dtype=np.float32)
        aref = 1.901

        idx = chooseOperatorIndex(N, a, aref)
        self.assertEqual(idx,9)

    def test_chooseOperatorIndex_2(self):

        N = 10
        a = np.linspace(0.0, 2.0, N, dtype=np.float32)
        aref = 0.01

        idx = chooseOperatorIndex(N, a, aref)
        self.assertEqual(idx,0)

    def test_chooseOperatorIndex_3(self):

        N = 100
        a = np.linspace(0.0, 0.2, N, dtype=np.float32)
        aref = 0

        idx = chooseOperatorIndex(N, a, aref)
        self.assertEqual(idx,0)

    def test_chooseOperatorIndex_4(self):

        N = 25
        a = np.linspace(0.0, 40.0, N, dtype=np.float32)
        aref = 40

        idx = chooseOperatorIndex(N, a, aref)
        self.assertEqual(idx,24)

    def test_chooseOperatorIndex_5(self):

        N = 11
        a = np.linspace(0.0, 150, N, dtype=np.float32)
        aref = 50

        idx = chooseOperatorIndex(N, a, aref)
        self.assertEqual(idx,3)