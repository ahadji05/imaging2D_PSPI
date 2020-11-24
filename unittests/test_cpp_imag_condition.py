
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
        imag_cond_cpu(ns, shotSz, nx, nf, forw, back, imageSz, 0, image)

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
        imag_cond_cpu(ns, shotSz, nx, nf, forw, back, imageSz, 5, image)

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

        imag_cond_cpu(ns, shotSz, nx, nf, forw, back, imageSz, depthIdx, image)

        self.assertAlmostEqual(image[sIdx,depthIdx,0], 6.0, delta=sdelta)
        self.assertAlmostEqual(image[sIdx,depthIdx,2], 15.0, delta=sdelta)
        self.assertAlmostEqual(image[sIdx,depthIdx,3], (forw[sIdx,2,3]*np.conj(back[sIdx,2,3])).real, delta=sdelta)

        for i in range(ns):
            for l in range(nz):
                if i != sIdx and l != depthIdx:
                    for ix in range(nx):
                        self.assertAlmostEqual(image[i,l,ix], 0.0, delta=sdelta)
