
import sys
import os
from ctypes import CDLL, c_int, c_void_p, c_float
from numpy.ctypeslib import ndpointer
import numpy as np

interface_path = os.path.abspath('.')

interpolation_gpu_dll      = CDLL(interface_path + '/libs/interpolation_gpu.so')

interpolation_gpu = interpolation_gpu_dll.test_interpolate_cu
interpolation_gpu.restype = c_void_p
interpolation_gpu.argtypes = [c_int, c_int, c_int, c_int,
                              ndpointer( dtype=np.float32, flags=("C","A") ),
                              ndpointer( dtype=np.complex64, flags=("C","A") ),
                              ndpointer( dtype=np.complex64, flags=("C","A") )]

