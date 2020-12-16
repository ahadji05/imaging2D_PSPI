
import sys
sys.path.append('../py_src/')

import unittest
 
import numpy as np
from math import pi

sdelta = 1e-5

#   ------------------------------------------

from pyprepOps import forwOperators as makeForwPSoperators
from pyprepOps import backOperators as makeBackPSoperators
from matplotlib import pyplot as plt

class Test_makeForwPSoperators(unittest.TestCase):

    def test_(self):

        kappa = np.array([i*0.05 for i in range(11)])

        kxmax = 0.5; kxmin = -kxmax
        Nkx = 51; dkx = (kxmax - kxmin) / (Nkx-1)
        kx = np.array([kxmin + i*dkx for i in range(Nkx)])
        
        dz = 5.0

        w_op_fk_forw = makeForwPSoperators(kappa, kx, dz)

        self.assertEqual(w_op_fk_forw.shape[0], len(kappa))
        self.assertEqual(w_op_fk_forw.shape[1], len(kx))
        
        print(w_op_fk_forw[0].imag)
        
        idx = 2
        plt.plot(kx, w_op_fk_forw[idx,:].imag, marker='.')
        plt.plot(kx, w_op_fk_forw[idx,:].real, marker='.')
        plt.plot(kx, np.abs(w_op_fk_forw[idx,:]), marker='.')
        plt.yticks(np.arange(-1.0, 1.01, 0.1))
        plt.xticks(np.arange(kxmin, kxmax, 0.1))
        plt.grid()
        plt.show()


# class Test_makeForwPSoperators(unittest.TestCase):

    # def test_makeRickerWavelet(self):
