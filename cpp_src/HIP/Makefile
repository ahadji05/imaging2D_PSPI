
# paths to include directories
INC           = ../include/
INC_hip       = /opt/rocm/hip/include/hip/
INC_rocthrust = /opt/rocm/rocthrust/include/
INC_hipfft    = /opt/rocm/hipfft/include/

# path to hipfft library : libhipfft.so
LIB_hipfft    = /opt/rocm/hipfft/lib/

# target amd architecture
Tarch = --offload-arch=gfx803

# set compiler and flags
CXX = hipcc
CXXFLAGS = -O3 -std=c++14 -fPIC $(Tarch) -DPWM_ENABLE_HIP
INCFLAGS = -I$(INC) -I$(INC_hip) -I$(INC_hipfft) -I$(INC_rocthrust)


# NO NEED TO EDIT BELOW THIS LINE
#################################

SRC = ../kernels/gpu/

all: 2dPWM

2dPWM: interpolation.o phase_shifts.o imaging_conditions.o device_ROCm_fft.o prep_operators.o
	$(CXX) -shared $(CXXFLAGS) *.o PWM_hip.cu -o PWM_hip.so $(INCFLAGS) -L$(LIB_hipfft) -lhipfft

device_ROCm_fft.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)device_ROCm_fft.cu -I$(INC) -I$(INC_hipfft)

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
	rm *.o PWM_hip.so
	