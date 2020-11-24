
#path to intel directory
INTEL = /opt/intel
# INTEL = /home/ahadji05/intel

#path to directory mkl
MKLROOT = $(INTEL)/mkl

# MKLROOT = $(EBROOTIMKL)/mkl

#define required for the linking MKL libraries
MKLLIB=-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed \
-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

#linking flags
LFLAGS= $(MKLLIB)

#inlcude Seiscmic Unix and MKL header files
INC = -DMKL_ILP64 -m64 -I$(MKLROOT)/include

#select compilers and flags
CXX=g++
CXXFLAGS = -O3 -std=c++11 $(INC)


# ------------------------------------------------
#     no need to edit below that line

all: extrap mkl_fft interpolate_cpu imag_condition_cpu

extrap:
	$(CXX) -shared -fPIC $(CXXFLAGS) -o extrap.so extrap.cpp

mkl_fft:
	$(CXX) -shared -fPIC $(CXXFLAGS) -o mkl_fft.so mkl_fft.cpp

interpolate_cpu:
	$(CXX) -shared -fPIC $(CXXFLAGS) -o interpolate_cpu.so interpolate.cpp

imag_condition_cpu:
	$(CXX) -shared -fPIC $(CXXFLAGS) -o imag_condition_cpu.so imag_condition.cpp

#remove executable and object files
clean:
	rm -f *.so *.o