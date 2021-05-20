
# path(s) to include directorie(s)
INC = ./include/
path_to_MKL_include = /opt/intel/mkl/include/


# path(s) to librarie(s)
LIB_MKL = /opt/intel/mkl/lib/intel64/


# define MKL libraries
MKLLIB=-L${LIB_MKL} -Wl,--no-as-needed \
-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl# Use Intel's OpenMP


# set compilers and flag(s)
# CXX = hipcc --offload-arch=gfx803 -DPWM_ENABLE_HIP
# CXX = clang++-10
CXX = g++
CXXFLAGS = -Wall -O3 -std=c++14 -fPIC -fopenmp


# NO NEED TO EDIT BELOW THIS LINE
#################################

INC_MKL = -DMKL_ILP64 -m64 -I$(path_to_MKL_include)

LFLAGS = $(MKLLIB)

SRC = ./kernels/cpu/

all: 2dPWM

2dPWM: interpolation.o prep_operators.o phase_shifts.o imaging_conditions.o mkl_fft.o
	$(CXX) -shared $(CXXFLAGS) *.o PWM_cpu.cpp -o PWM_cpu.so -I$(INC) -I$(INC_MKL) -L$(LIB_MKL) $(LFLAGS)

mkl_fft.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)mkl_fft.cpp -I$(INC) -I$(INC_MKL)

imaging_conditions.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)imaging_conditions.cpp -I$(INC)

interpolation.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)interpolation.cpp -I$(INC)

prep_operators.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)prep_operators.cpp -I$(INC)

phase_shifts.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)phase_shifts.cpp -I$(INC)

remake: clean all

clean:
	rm *.o PWM_cpu.so
	
