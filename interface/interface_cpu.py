
import sys
import os
from ctypes import CDLL, c_int, c_void_p, c_float
from numpy.ctypeslib import ndpointer
import numpy as np

interface_path = os.path.abspath('.')

PWM2d_dll = CDLL(interface_path + '/cpp_src/PWM_cpu.so')

PWM2d = PWM2d_dll.extrapAndImag
PWM2d.restype = c_void_p
PWM2d.argtypes = [c_int, c_int, c_int, c_int,
                          c_int, c_int, c_int,
                          c_float, c_int, c_float, c_int,
                          ndpointer( dtype=np.float32, flags=("C","A") ),
                          ndpointer( dtype=np.float32, flags=("C","A") ),
                          ndpointer( dtype=np.float32, flags=("C","A") ),
                          ndpointer( dtype=np.complex64, flags=("C","A") ),
                          ndpointer( dtype=np.complex64, flags=("C","A") ),
                          ndpointer( dtype=np.float32, flags=("C","A") )]
