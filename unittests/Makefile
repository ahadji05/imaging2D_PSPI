# NOTE!!
# For nvcc compiler must add -Xcompiler in the CXXFLAGS options, for hipcc this is not important!
#------------------------------------------------------------------------------------------------

# path(s) to include directorie(s)
path_to_MKL_include = /home/floriano/anaconda3/include/
INC_hip       = /opt/rocm/hip/include/hip/
INC_DEV_FFT    = /opt/rocm/hipfft/include/


# path(s) to librarie(s)
LIB_MKL = /home/floriano/anaconda3/lib/

LIB_DEV_FFT    = /opt/rocm/hipfft/lib/ -lhipfft

# set compilers and flag(s)
CXX = hipcc --offload-arch=gfx803 -DPWM_ENABLE_HIP
# CXX = clang++-10
# CXX = g++

CXXFLAGS = -Wall -O3 -std=c++14 -fPIC -fopenmp


# NO NEED TO EDIT BELOW THIS LINE
#################################

MKLLIB=-L${LIB_MKL} -Wl,--no-as-needed \
-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl# Use Intel's OpenMP

INC_MKL = -DMKL_ILP64 -m64 -I$(path_to_MKL_include)

LFLAGS = $(MKLLIB)

INC = ../cpp_src/include/

SRC = ../cpp_src/kernels/

cpu: imaging_conditions.so interpolation.so phase_shifts.so mkl_fft.so

gpu_amd: gpu_common dev_rocm_fft.so

gpu_cuda: gpu_common dev_cuda_fft.so

gpu_common: imaging_conditions_gpu.so interpolation_gpu.so phase_shifts_gpu.so


# 		GENERATE THE DYNAMICALY LINKED LIBRARIES (DLLs)
#		################################################

# 						----- CPU DLLs -----

mkl_fft.so: mkl_fft.o
	$(CXX) -shared $(CXXFLAGS) mkl_fft.o -o mkl_fft.so -L$(LIB_MKL) $(LFLAGS)
	rm mkl_fft.o
	mv mkl_fft.so libs/


phase_shifts.so: phase_shifts.o
	$(CXX) -shared $(CXXFLAGS) phase_shifts.o -o phase_shifts.so
	rm phase_shifts.o
	mv phase_shifts.so libs/


imaging_conditions.so: imaging_conditions.o
	$(CXX) -shared $(CXXFLAGS) imaging_conditions.o -o imaging_conditions.so
	rm imaging_conditions.o
	mv imaging_conditions.so libs/


interpolation.so: interpolation.o
	$(CXX) -shared $(CXXFLAGS) interpolation.o -o interpolation.so
	rm interpolation.o
	mv interpolation.so libs/

mkl_fft.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)cpu/mkl_fft.cpp -I$(INC) -I$(INC_MKL)

interpolation.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)cpu/interpolation.cpp -I$(INC)

phase_shifts.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)cpu/phase_shifts.cpp -I$(INC)

imaging_conditions.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)cpu/imaging_conditions.cpp -I$(INC)



# 						----- GPU_COMMON DLLs -----

interpolation_gpu.so: interpolation_gpu.o
	$(CXX) -shared $(CXXFLAGS) test_interpolation_gpu.o interpolation_gpu.o -o interpolation_gpu.so
	rm interpolation_gpu.o
	rm test_interpolation_gpu.o
	mv interpolation_gpu.so libs/

imaging_conditions_gpu.so: imaging_conditions_gpu.o
	$(CXX) -shared $(CXXFLAGS) imaging_conditions_gpu.o -o imaging_conditions_gpu.so
	rm imaging_conditions_gpu.o
	mv imaging_conditions_gpu.so libs/


phase_shifts_gpu.so: phase_shifts_gpu.o
	$(CXX) -shared $(CXXFLAGS) phase_shifts_gpu.o -o phase_shifts_gpu.so
	rm phase_shifts_gpu.o
	mv phase_shifts_gpu.so libs/

interpolation_gpu.o:
	$(CXX) $(CXXFLAGS) -c gpu_wrappers/test_interpolation_gpu.cpp -o test_interpolation_gpu.o -I$(INC)
	$(CXX) $(CXXFLAGS) -c $(SRC)gpu/interpolation.cu -o interpolation_gpu.o -I$(INC)

imaging_conditions_gpu.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)gpu/imaging_conditions.cu -o imaging_conditions_gpu.o -I$(INC)

phase_shifts_gpu.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)gpu/phase_shifts.cu -o phase_shifts_gpu.o -I$(INC)



#						----- DEV_ROCm_FFTs DLL -----

dev_rocm_fft.so: dev_rocm_fft.o
	$(CXX) -shared $(CXXFLAGS) dev_rocm_fft.o -o dev_rocm_fft.so -L$(LIB_DEV_FFT)
	rm dev_rocm_fft.o
	mv dev_rocm_fft.so libs/

dev_rocm_fft.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)gpu/device_ROCm_fft.cu -o dev_rocm_fft.o -I$(INC) -I$(INC_DEV_FFT)



#						----- DEV_CUDA_FFTs DLL -----

dev_cuda_fft.so: dev_cuda_fft.o
	$(CXX) -shared $(CXXFLAGS) dev_cuda_fft.o -o dev_cuda_fft.so -L$(LIB_DEV_FFT)
	rm dev_cuda_fft.o
	mv dev_cuda_fft.so libs/

dev_cuda_fft.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)gpu/device_CUDA_fft.cu -o dev_cuda_fft.o -I$(INC) -I$(INC_DEV_FFT)




#						----- CLEANING STUFF -----

clean: clean_libs

clean_libs:
	rm libs/*.so

clean_obj:
	rm *.o
