
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

class Testmkl_fft(unittest.TestCase):
    
    def test_src_C_c64fft1dforw(self):

        list_of_N = [10, 33, 155]
        for N in list_of_N:

            A = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.complex64)
            A.imag = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.float32)
            Cmkl = A

            C = np.fft.fft(A)
            Cscipy = fftpack.fft(A)
            
            c64fft1dforw(Cmkl, N)

            #compare all output results one-by-one
            for i in range(N):
                self.assertAlmostEqual(C[i].real, Cmkl[i].real, delta=sdelta)
                self.assertAlmostEqual(C[i].imag, Cmkl[i].imag, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i].real, Cmkl[i].real, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i].imag, Cmkl[i].imag, delta=sdelta)


    def test_src_C_c64fft1dback(self):
        
        list_of_N = [10, 33, 155]
        for N in list_of_N:
            
            A = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.complex64)
            A.imag = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.float32)
            Cmkl = A

            C = np.fft.ifft(A)
            Cscipy = fftpack.ifft(A)

            c64fft1dback(Cmkl, N)

            #compare all output results one-by-one
            for i in range(N):
                self.assertAlmostEqual(C[i].real, Cmkl[i].real, delta=sdelta)
                self.assertAlmostEqual(C[i].imag, Cmkl[i].imag, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i].real, Cmkl[i].real, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i].imag, Cmkl[i].imag, delta=sdelta)
    

    def test_src_C_c64fft1dforwFrom2D_axis_0(self):

        #prepare data
        N1 = 4
        N2 = 3
        A = np.zeros((N1,N2), dtype=np.complex64)
        A[0,0] = 1+1j; A[0,1] = 0.5-1.2j; A[0,2] = 0.8+1j
        A[1,0] = 1-1j; A[1,1] = 2+1j; A[1,2] = 1.5+2j
        A[2,0] = 1.4+1.4j; A[2,1] = 0.9+0.2j; A[2,2] = 1.8-1j
        A[3,0] = 3+1j; A[3,1] = 1.5+0.2j; A[3,2] = -0.1+2.5j
        Cmkl = np.zeros_like(A)
        np.copyto(Cmkl, A)

        # DO FFTs across axis=0
        C = np.fft.fft(A, axis=0)
        Cscipy = fftpack.fft(A, axis=0)

        c64fft1dforwFrom2d(Cmkl, N1, N2, 0) #this is done IN-PLACE
        
        #check all values one-by-one
        for i in range(N1):
            for j in range(N2):
                self.assertAlmostEqual(C[i,j].real, Cmkl[i,j].real, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i,j].real, Cmkl[i,j].real, delta=sdelta)
                self.assertAlmostEqual(C[i,j].imag, Cmkl[i,j].imag, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i,j].imag, Cmkl[i,j].imag, delta=sdelta)


    def test_src_C_c64fft1dforwFrom2D_axis_1(self):

        #prepare data
        N1 = 4
        N2 = 3
        A = np.zeros((N1,N2), dtype=np.complex64)
        A[0,0] = 1+1j; A[0,1] = 0.5-1.2j; A[0,2] = 0.8+1j
        A[1,0] = 1-1j; A[1,1] = 2+1j; A[1,2] = 1.5+2j
        A[2,0] = 1.4+1.4j; A[2,1] = 0.9+0.2j; A[2,2] = 1.8-1j
        A[3,0] = 3+1j; A[3,1] = 1.5+0.2j; A[3,2] = -0.1+2.5j
        Cmkl = np.zeros_like(A)
        np.copyto(Cmkl, A)

        # DO FFTs across axis=1
        C = np.fft.fft(A, axis=1)
        Cscipy = fftpack.fft(A, axis=1)

        c64fft1dforwFrom2d(Cmkl, N1, N2, 1) #this is done IN-PLACE
        
        #check all values one-by-one
        for i in range(N1):
            for j in range(N2):
                self.assertAlmostEqual(C[i,j].real, Cmkl[i,j].real, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i,j].real, Cmkl[i,j].real, delta=sdelta)
                self.assertAlmostEqual(C[i,j].imag, Cmkl[i,j].imag, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i,j].imag, Cmkl[i,j].imag, delta=sdelta)


    def test_src_C_c64fft1dbackFrom2D_axis_0(self):

        #prepare data
        N1 = 4
        N2 = 3
        A = np.zeros((N1,N2), dtype=np.complex64)
        A[0,0] = 1+1j; A[0,1] = 0.5-1.2j; A[0,2] = 0.8+1j
        A[1,0] = 1-1j; A[1,1] = 2+1j; A[1,2] = 1.5+2j
        A[2,0] = 1.4+1.4j; A[2,1] = 0.9+0.2j; A[2,2] = 1.8-1j
        A[3,0] = 3+1j; A[3,1] = 1.5+0.2j; A[3,2] = -0.1+2.5j
        Cmkl = np.zeros_like(A)
        np.copyto(Cmkl, A)

        # DO FFTs across axis=0
        C = np.fft.ifft(A, axis=0)
        Cscipy = fftpack.ifft(A, axis=0)

        c64fft1dbackFrom2d(Cmkl, N1, N2, 0) #this is done IN-PLACE
        
        #check all values one-by-one
        for i in range(N1):
            for j in range(N2):
                self.assertAlmostEqual(C[i,j].real, Cmkl[i,j].real, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i,j].real, Cmkl[i,j].real, delta=sdelta)
                self.assertAlmostEqual(C[i,j].imag, Cmkl[i,j].imag, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i,j].imag, Cmkl[i,j].imag, delta=sdelta)


    def test_src_C_c64fft1dbackFrom2D_axis_1(self):

        #prepare data
        N1 = 4
        N2 = 3
        A = np.zeros((N1,N2), dtype=np.complex64)
        A[0,0] = 1+1j; A[0,1] = 0.5-1.2j; A[0,2] = 0.8+1j
        A[1,0] = 1-1j; A[1,1] = 2+1j; A[1,2] = 1.5+2j
        A[2,0] = 1.4+1.4j; A[2,1] = 0.9+0.2j; A[2,2] = 1.8-1j
        A[3,0] = 3+1j; A[3,1] = 1.5+0.2j; A[3,2] = -0.1+2.5j
        Cmkl = np.zeros_like(A)
        np.copyto(Cmkl, A)

        # DO FFTs across axis=0
        C = np.fft.ifft(A, axis=1)
        Cscipy = fftpack.ifft(A, axis=1)

        c64fft1dbackFrom2d(Cmkl, N1, N2, 1) #this is done IN-PLACE
        
        #check all values one-by-one
        for i in range(N1):
            for j in range(N2):
                self.assertAlmostEqual(C[i,j].real, Cmkl[i,j].real, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i,j].real, Cmkl[i,j].real, delta=sdelta)
                self.assertAlmostEqual(C[i,j].imag, Cmkl[i,j].imag, delta=sdelta)
                self.assertAlmostEqual(Cscipy[i,j].imag, Cmkl[i,j].imag, delta=sdelta)
