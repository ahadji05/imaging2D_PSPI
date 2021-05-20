
import sys
import os
from ctypes import CDLL, c_int, c_void_p, c_float
from numpy.ctypeslib import ndpointer
import numpy as np

interface_path = os.path.abspath('.')

interpolation_dll      = CDLL(interface_path + '/libs/interpolation.so')
imaging_conditions_dll = CDLL(interface_path + '/libs/imaging_conditions.so')
phase_shifts_dll       = CDLL(interface_path + '/libs/phase_shifts.so')


interpolation_cpu = interpolation_dll.interpolate
interpolation_cpu.restype = c_void_p
interpolation_cpu.argtypes = [c_int, c_int, c_int, c_int,
                              ndpointer( dtype=np.float32, flags=("C","A") ),
                              ndpointer( dtype=np.complex64, flags=("C","A") ),
                              ndpointer( dtype=np.complex64, flags=("C","A") )]


imaging_conditions_cpu = imaging_conditions_dll.imaging
imaging_conditions_cpu.restype = c_void_p
imaging_conditions_cpu.argtypes = [c_int, c_int, c_int, c_int,
                                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                                   c_int, c_int,
                                   ndpointer( dtype=np.float32, flags=("C","A") )]


phase_shifts_forw_cpu = phase_shifts_dll.phase_shift_forw
phase_shifts_forw_cpu.restype = c_void_p
phase_shifts_forw_cpu.argtypes = [c_int, c_int, c_int, 
                             ndpointer( dtype=np.complex64, flags=("C","A") ),
                             ndpointer( dtype=np.float32, flags=("C","A") ),
                             ndpointer( dtype=np.float32, flags=("C","A") ),
                             c_float]


phase_shifts_back_cpu = phase_shifts_dll.phase_shift_back
phase_shifts_back_cpu.restype = c_void_p
phase_shifts_back_cpu.argtypes = [c_int, c_int, c_int, 
                             ndpointer( dtype=np.complex64, flags=("C","A") ),
                             ndpointer( dtype=np.float32, flags=("C","A") ),
                             ndpointer( dtype=np.float32, flags=("C","A") ),
                             c_float]


extrap_ref_wavefields_cpu = phase_shifts_dll.extrap_ref_wavefields
extrap_ref_wavefields_cpu.restype = c_void_p
extrap_ref_wavefields_cpu.argtypes = [
                             ndpointer( dtype=np.complex64, flags=("C","A") ),
                             ndpointer( dtype=np.complex64, flags=("C","A") ),
                             ndpointer( dtype=np.complex64, flags=("C","A") ),
                             ndpointer( dtype=np.int32, flags=("C","A") ),
                             c_int, c_int, c_int, c_int]


