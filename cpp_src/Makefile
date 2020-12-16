
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
CXXFLAGS = -Wall -shared -fPIC -O3 -std=c++11


# ------------------------------------------------
#     no need to edit below that line

remake: clean all

all: extrap prepOps mkl_fft imag_condition_cpu interpolate_cpu

extrap: interpolate.o imag_condition.o extrap.o prepOps.o mkl_fft.o
	$(CXX) $(CXXFLAGS) -o extrap.so extrap.o prepOps.o mkl_fft.o interpolate.o imag_condition.o $(LFLAGS)

extrap.o:
	$(CXX) -Wall $(CXXFLAGS) $(INC) -c extrap.cpp

prepOps: prepOps.o
	$(CXX) $(CXXFLAGS) -o prepOps.so prepOps.o

prepOps.o:
	$(CXX) $(CXXFLAGS) -c prepOps.cpp

mkl_fft: mkl_fft.o
	$(CXX) $(CXXFLAGS) $(INC) -o mkl_fft.so mkl_fft.o $(LFLAGS)

mkl_fft.o:
	$(CXX) -Wall $(CXXFLAGS) $(INC) -c mkl_fft.cpp

interpolate_cpu: interpolate_cpu.o
	$(CXX) $(CXXFLAGS) -o interpolate_cpu.so interpolate.o

interpolate_cpu.o:
	$(CXX) -Wall $(CXXFLAGS) -c interpolate.cpp

imag_condition_cpu: imag_condition_cpu.o
	$(CXX) $(CXXFLAGS) -o imag_condition_cpu.so imag_condition.o -fopenmp

imag_condition_cpu.o: imag_condition.cpp
	$(CXX) -Wall $(CXXFLAGS) -c imag_condition.cpp -fopenmp

#remove executable and object files
clean:
	rm -f *.so *.o