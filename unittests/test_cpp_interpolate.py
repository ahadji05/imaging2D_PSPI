
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
