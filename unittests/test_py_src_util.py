
import sys
sys.path.append('../py_src/')

import unittest
 
import numpy as np
from math import pi

sdelta = 1e-5

#   ------------------------------------------

from util import makeRickerWavelet as makeShot

#need to run an un-tested example in advance to avoid run-time error by numba!
shot = makeShot([0], 0.0, 4, 3, [1,2,3,4], [1,2,3], 0.1, 500)

class Test_makeRickerWavelet(unittest.TestCase):

    def test_makeRickerWavelet(self):
        
        #test-setup
        isx = -600
        nt = 500
        dt = 0.001
        nx = 151
        dx = 15
        xmin = -800 # meters
        z1 = 0; z2 = 10 #meters
        v = 2600 # meters/s

        #prep hard-coded test case
        testshot = np.zeros((nt,nx), dtype=np.float32)
        for i in range(nx):
            xloc = xmin + i*dx
            r = np.sqrt( (z2-z1)**2 + (isx-xloc)**2 )
            t0 = r/v
            for j in range(nt):        
                t_t0 = j*dt - t0
                term = pi**2*30.0**2*(t_t0)**2
                testshot[j,i] = (1-2*term)*np.exp(-term)/np.sqrt(r)

        #run function
        tj = [j*dt for j in range(nt)]
        xi = [xmin + i*dx for i in range(nx)]
        shot = makeShot([isx], z1, nt, nx, tj, xi, z2, v)

        #test results
        self.assertEqual(shot.shape[0], nt)
        self.assertEqual(shot.shape[1], nx)
        for i in range(nx):
            for j in range(nt):
                self.assertAlmostEqual(shot[j,i], testshot[j,i], delta=sdelta)

# -------------------------------------------------------------

from util import createKappa

class Test_createKappa(unittest.TestCase):

    def test_createKappa_different_N(self):

        minvel = 1000
        wmax = 650
        N = [10, 20, 100, 300, 500, 1000, 2000, 3000]

        for n in N:
            kappa, deltak = createKappa(minvel, wmax, n)
            self.assertAlmostEqual(0.65, kappa[n-1], delta=sdelta)
            self.assertAlmostEqual(0.0, kappa[0], delta=sdelta)
            self.assertAlmostEqual(deltak, kappa[1]-kappa[0], delta=sdelta)

    def test_createKappa_different_wmax(self):

        minvel = 1000
        N = 2000
        WMAX = [50, 250, 600, 930, 2100]

        for wmax in WMAX:
            kappa, deltak = createKappa(minvel, wmax, N)
            self.assertAlmostEqual(kappa[N-1], wmax/minvel, delta=sdelta)
            self.assertAlmostEqual(kappa[0], 0.0, delta=sdelta)
            self.assertEqual(N, len(kappa))
            self.assertAlmostEqual(deltak, kappa[1]-kappa[0], delta=sdelta)

#   ------------------------------------------

from util import find_nearest

find_nearest(np.array([1,2,3],dtype=np.int16),2)

class Test_find_nearest(unittest.TestCase):

    #test for integers
    def test_find_nearest_int16(self):

        Aint16 = np.array([0,4,6,3,2,54,3], dtype=np.int16)

        idx = find_nearest(Aint16, 6)
        self.assertEqual(idx,2)

    #test for floats
    def test_find_nearest_float32(self):
    
        Afloat32 = np.array([0.0,4.5,6.2,3.7,2.4,5.4,3000.0], dtype=np.float32)

        idx = find_nearest(Afloat32, 5.0)
        self.assertEqual(idx,5)

        idx = find_nearest(Afloat32, 4.8)
        self.assertEqual(idx,1)
        