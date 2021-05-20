import sys

import numpy as np
import scipy.fftpack as spfft
from scipy import ndimage

import readShotFiles

sys.path.append('py_src/')
import problemConfig as pConfig
import util

sys.path.append('interface/')

import time

time_start = time.time()

#   ------------------------------------------
#   READ COMMAND LINE ARGUMENTS
io_time_start = time.time()

file_vel = sys.argv[1] #filename of velocity model (csv)
file_config = sys.argv[2] #filename of configuration
shots_dir = sys.argv[3] #directory to shots
path_to_output = sys.argv[4] #path to output directory
option = str(sys.argv[5])

if option == "host":
    from interface_cpu import PWM2d as run_PWM2d
elif option == "devAMD":
    from interface_hip import PWM2d_AMD as run_PWM2d
elif option == "devCUDA":
    from interface_cuda import PWM2d_CUDA as run_PWM2d
else:
    print("NO VALID OPTION HAS BEEN SPECIFIED")
    exit()
#   ------------------------------------------
#   READ VELOCITY MODEL

velocity_model = np.genfromtxt(file_vel, delimiter=',')
slownes = 1.0 / velocity_model
slownes = ndimage.gaussian_filter(slownes, sigma=2)
velocity_model = 1.0 / slownes
velocity_model = velocity_model.astype(np.float32)
np.savetxt(path_to_output+"/smoothed_velmod.csv", velocity_model, delimiter=',')
vmin = np.min(velocity_model)
vmax = np.max(velocity_model)

#   ------------------------------------------
#   SETUP EXPERIMENT CONFIGURATION

config = pConfig.problemConfig(filename=file_config, \
    vmin=vmin, vmax=vmax, \
    nz=velocity_model.shape[0],\
    nx=velocity_model.shape[1], ny=1)

io_time_stop = time.time()
io_time_total = round(io_time_stop-io_time_start,2)

#   ------------------------------------------
#   PRINT PROBLEM CONFIGURATION

config.dispInfo()

#   ------------------------------------------
#   READ SEISMOGRAPH FILES AND SOURCES INDICES
print("Preparing shots for all sources ...")

prep_shots_time_start = time.time()

shot_isx, file_shot = readShotFiles.returnShotIndices(shots_dir, "csv", "seis", "_")
ns = len(shot_isx)

print("number of shots:",ns)
print("shot's indices:",shot_isx)

#   ------------------------------------------
#   PREPARE ONE RICKER WAVELETS PER SHOT

v = velocity_model[0,0] #assumes const velocity in first depth-step
tj = [j*config.dt for j in range(config.nt)]
xi = [config.xmin+(i+1)*config.dx for i in range(config.nx)]

pulse_forw_st = np.zeros((ns,config.nt,config.nx), dtype=np.float32)
pulse_forw_fs = np.zeros((ns,config.nt,config.nx), dtype=np.complex64)
pulse_back_st = np.zeros((ns,config.nt,config.nx), dtype=np.float32)
pulse_back_fs = np.zeros((ns,config.nt,config.nx), dtype=np.complex64)
for s in range(ns):
    pulse_forw_st[s,:,:] = util.makeRickerWavelet([config.xmin + shot_isx[s]*config.dx], config.zinit, config.nt, config.nx, tj, xi, config.dz, v)
    pulse_forw_fs[s,:,:] = spfft.fft(pulse_forw_st[s,:,:], axis=0)
    pulse_back_st[s,:,:] = np.genfromtxt(file_shot[s], delimiter=',', dtype=np.float32)
    pulse_back_fs[s,:,:] = spfft.fft(pulse_back_st[s,:,:], axis=0)

prep_shots_time_stop = time.time()
prep_shots_time_total = round(prep_shots_time_stop-prep_shots_time_start,2)

#   -------------------------------------------
#   EXTRAPOLATION AND IMAGING
print("extrapolation and imaging ...")

extrap_time_start = time.time()
#---------------------------------------------------------------------
kmax = config.w[config.nw] / vmin
image = np.zeros((ns,config.nz, config.nx), dtype=np.float32)
omega = config.w.astype(np.float32)
kxx = config.kx.astype(np.float32)
print("wmax:", config.w[config.nw])

run_PWM2d(ns, config.nvel, config.nz, config.nextrap, config.nt, \
    config.nw, config.nx, config.dz, 1000, kmax, config.nx, omega, kxx, \
    velocity_model, pulse_forw_fs, pulse_back_fs, image)

#---------------------------------------------------------------------
extrap_time_stop = time.time()
extrap_time_total = round(extrap_time_stop-extrap_time_start,2)

time_stop = time.time()
total_time = round(time_stop-time_start,2)
#   ---------------------------------

print("-------------------------------")
print("Total program time (s) :",total_time)
print("    I/O-time (s)                   :",io_time_total)
print("    Prep shots (s)                 :",prep_shots_time_total)
print("    Extrapolation and Imaging (s)  :",extrap_time_total)

final_image = np.zeros((config.nz,config.nx), dtype=np.float32)
for s in range(ns):
    np.savetxt(path_to_output+"/image"+str(shot_isx[s])+"_.csv", image[s], delimiter=',')
    final_image += image[s]

#save final image
np.savetxt(path_to_output+"/final_image.csv", final_image, delimiter=',')
