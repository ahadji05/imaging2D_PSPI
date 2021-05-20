
import sys
import os
from ctypes import CDLL, c_int, c_void_p, c_float
from numpy.ctypeslib import ndpointer
import numpy as np

interface_path = os.path.abspath('.')

PWM2d_AMD_dll = CDLL(interface_path + '/cpp_src/HIP/PWM_hip.so')

PWM2d_AMD = PWM2d_AMD_dll.extrapAndImag_amd_cu
PWM2d_AMD.restype = c_void_p
PWM2d_AMD.argtypes = [c_int, c_int, c_int, c_int,
                        c_int, c_int, c_int,
                        c_float, c_int, c_float, c_int,
                        ndpointer( dtype=np.float32, flags=("C","A") ),
                        ndpointer( dtype=np.float32, flags=("C","A") ),
                        ndpointer( dtype=np.float32, flags=("C","A") ),
                        ndpointer( dtype=np.complex64, flags=("C","A") ),
                        ndpointer( dtype=np.complex64, flags=("C","A") ),
                        ndpointer( dtype=np.float32, flags=("C","A") )]
