
#path to intel directory
INTEL = /opt/intel
# INTEL = /home/ahadji05/intel

#path to directory mkl
MKLROOT = $(INTEL)/mkl
# MKLROOT = $(EBROOTIMKL)/mkl

#define required for the linking MKL libraries
# --------------------------------------------
MKLLIB=-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed \
-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl #Intel's OpenMP

# MKLLIB=-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed \
# -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl # NO OpenMP 
#---------------------------------------------

#linking flags
LFLAGS= $(MKLLIB)

#inlcude Seiscmic Unix and MKL header files
INC = -DMKL_ILP64 -m64 -I$(MKLROOT)/include

#select compilers and flags
CXX=g++
CXXFLAGS = -shared -fPIC -O3 -std=c++11


# ------------------------------------------------
#     no need to edit below that line

all: imaging interpolate mkl_fft prepOps extrap

extrap: extrap.o
	$(CXX) $(CXXFLAGS) -o extrap.so extrap.o prepOps.o mkl_fft.o imag_condition.o interpolate.o $(LFLAGS)

extrap.o: extrap.cpp
	$(CXX) -Wall $(CXXFLAGS) $(INC) -c extrap.cpp -fopenmp

prepOps: prepOps.o
	$(CXX) $(CXXFLAGS) -o prepOps.so prepOps.o

prepOps.o: prepOps.cpp
	$(CXX) -Wall $(CXXFLAGS) -c prepOps.cpp

mkl_fft: mkl_fft.o
	$(CXX) $(CXXFLAGS) -o mkl_fft.so mkl_fft.o $(LFLAGS)

mkl_fft.o: mkl_fft.cpp
	$(CXX) -Wall $(CXXFLAGS) $(INC) -c mkl_fft.cpp

interpolate: interpolate.o
	$(CXX) $(CXXFLAGS) -o interpolate_cpu.so interpolate.o -fopenmp

interpolate.o: interpolate.cpp
	$(CXX) -Wall $(CXXFLAGS) -c interpolate.cpp -fopenmp

imaging: imaging.o
	$(CXX) $(CXXFLAGS) -o imag_condition_cpu.so imag_condition.o -fopenmp

imaging.o: imag_condition.cpp
	$(CXX) -Wall $(CXXFLAGS) -c imag_condition.cpp -fopenmp

remake: clean all

#remove executable and object files
clean:
	rm -f *.so *.o