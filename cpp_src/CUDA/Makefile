
# paths to include directories
INC           = ../include/

# path to cufft library : libcufft.so
LIB_cufft    = /path/to/cufft/lib/

# target amd architecture
Tarch = -arch=sm_35

# set compiler and flags
CXX = nvcc
CXXFLAGS = -O3 -std=c++14 -Xcompiler -fPIC $(Tarch) -DPWM_ENABLE_CUDA
INCFLAGS = -I$(INC)


# NO NEED TO EDIT BELOW THIS LINE
#################################

SRC = ../kernels/gpu/

all: 2dPWM

2dPWM: interpolation.o phase_shifts.o imaging_conditions.o device_CUDA_fft.o prep_operators.o
	$(CXX) -shared $(CXXFLAGS) *.o PWM_cuda.cu -o PWM_cuda.so $(INCFLAGS) -L$(LIB_cufft) -lcufft

device_CUDA_fft.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)device_CUDA_fft.cu -I$(INC)

imaging_conditions.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)imaging_conditions.cu -I$(INC)

interpolation.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)interpolation.cu -I$(INC)

phase_shifts.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)phase_shifts.cu -I$(INC)

prep_operators.o:
	$(CXX) $(CXXFLAGS) -c ../kernels/cpu/prep_operators.cpp -I$(INC)

remake: clean all

clean:
	rm *.o PWM_cuda.so
	
