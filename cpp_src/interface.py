
import sys
import os
from ctypes import CDLL, POINTER, c_int, c_void_p, c_float, c_wchar
from numpy.ctypeslib import ndpointer
import numpy as np

#------------------------
interface_path = os.path.dirname(__file__)

extrap = CDLL(interface_path + '/extrap.so')
prepOps = CDLL(interface_path + '/prepOps.so')
mkl_fft = CDLL(interface_path + '/mkl_fft.so')
imag_cpu = CDLL(interface_path + '/imag_condition_cpu.so')
interpolate_cpu = CDLL(interface_path + '/interpolate_cpu.so')

#--------------------------
#   extrap
#--------------------------

extrapolation = extrap.extrapAndImag
extrapolation.restype = c_void_p
extrapolation.argtypes = [c_int, c_int, c_int, c_int,
                          c_int, c_int, c_int,
                          c_float, c_int, c_float, c_int,
                          ndpointer( dtype=np.float32, flags=("C","A") ),
                          ndpointer( dtype=np.float32, flags=("C","A") ),
                          ndpointer( dtype=np.float32, flags=("C","A") ),
                          ndpointer( dtype=np.complex64, flags=("C","A") ),
                          ndpointer( dtype=np.complex64, flags=("C","A") ),
                          ndpointer( dtype=np.float32, flags=("C","A") )]

PSforw = extrap.phase_shift_forw
PSforw.restype = c_void_p
PSforw.argtypes = [c_int, c_int, c_int,
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.float32, flags=("C","A") ),
                   ndpointer( dtype=np.float32, flags=("C","A") ),
                   c_float]

PSback = extrap.phase_shift_back
PSback.restype = c_void_p
PSback.argtypes = [c_int, c_int, c_int,
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.float32, flags=("C","A") ),
                   ndpointer( dtype=np.float32, flags=("C","A") ),
                   c_float]


#--------------------------
#   prepOps
#--------------------------

test_class_prepOps = prepOps.testClassWrapperFor_f_kx_operators
test_class_prepOps.restype = c_void_p
test_class_prepOps.argtypes = [ndpointer( dtype=np.float32, flags=("C","A") ),
                               ndpointer( dtype=np.float32, flags=("C","A") ),
                               ndpointer( dtype=np.complex64, flags=("C","A") ),
                               c_int, c_float, c_int, c_float, c_wchar]

chooseOperatorIndex = prepOps.chooseOperatorIndex
chooseOperatorIndex.restype = c_int
chooseOperatorIndex.argtypes = [c_int,
                               ndpointer( dtype=np.float32, flags=("C","A") ),
                               c_float]
                               
#--------------------------
#   mkl_fft
#--------------------------

c64fft1dforw = mkl_fft.c64fft1dforw
c64fft1dforw.restype = c_void_p
c64fft1dforw.argtypes = [ndpointer( dtype=np.complex64, flags=("C","A") ),
                         c_int]

c64fft1dback = mkl_fft.c64fft1dback
c64fft1dback.restype = c_void_p
c64fft1dback.argtypes = [ndpointer( dtype=np.complex64, flags=("C","A") ),
                         c_int]

c64fft1dforwFrom2d = mkl_fft.fft1dforwardFrom2Darray
c64fft1dforwFrom2d.restype = c_void_p
c64fft1dforwFrom2d.argtypes = [ndpointer( dtype=np.complex64, flags=("C","A") ),
                               c_int, c_int, c_int]

c64fft1dbackFrom2d = mkl_fft.fft1dbackwardFrom2Darray
c64fft1dbackFrom2d.restype = c_void_p
c64fft1dbackFrom2d.argtypes = [ndpointer( dtype=np.complex64, flags=("C","A") ),
                               c_int, c_int, c_int]

#--------------------------
#   imaging condition
#--------------------------

imag_cond_cpu = imag_cpu.cross_corr
imag_cond_cpu.restype = c_void_p
imag_cond_cpu.argtypes = [c_int, c_int, c_int, c_int, 
                          ndpointer( dtype=np.complex64, flags=("C","A") ),
                          ndpointer( dtype=np.complex64, flags=("C","A") ),
                          c_int, c_int,
                          ndpointer( dtype=np.float32, flags=("C","A") )]

#--------------------------
#   interpolation
#--------------------------

interpolation = interpolate_cpu.interpolate
interpolation.restype = c_void_p
interpolation.argtypes = [c_int, c_int, c_int,
                          ndpointer( dtype=np.float32, flags=("C","A") ),
                          ndpointer( dtype=np.complex64, flags=("C","A") ),
                          ndpointer( dtype=np.complex64, flags=("C","A") )]

find_coeff = interpolate_cpu.find_coeff
find_coeff.restype = c_void_p
find_coeff.argtypes = [c_int,
                       ndpointer( dtype=np.float32, flags=("C","A") ),
                       c_int,
                       ndpointer( dtype=np.float32, flags=("C","A") ),
                       ndpointer( dtype=np.float32, flags=("C","A") )]
    
norm_coeff = interpolate_cpu.norm_coeff
norm_coeff.restype = c_void_p
norm_coeff.argtypes = [c_int,
                       c_int,
                       ndpointer( dtype=np.float32, flags=("C","A") )]
