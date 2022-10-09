
#set path to Libraries installation
mklroot=/opt/intel/mkl
fftwroot=/usr/local

# NO NEED TO EDIT BELOW THIS LINE
#################################




# path(s) to include directorie(s)
# --------------------------------
INC = ./include/
INC_MKL=$(mklroot)/include/
INC_FFTW=$(fftwroot)/include/

# path(s) to librarie(s)
# ----------------------
MKLLIB=-L$(mklroot)/lib/intel64/ -Wl,--no-as-needed \
-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl# Use Intel's OpenMP
FFTWLIB = -L${fftwroot}/lib/

# set compilers and flag(s)
# -------------------------
CXX = g++
CXXFLAGS = -Wall -O3 --fast-math -std=c++14 -fPIC -fopenmp

SRC = ./kernels/cpu/

2dPWM_mkl: interpolation.o prep_operators.o phase_shifts.o imaging_conditions.o mkl_fft.o
	$(CXX) -shared $(CXXFLAGS) *.o PWM_cpu.cpp -o PWM_cpu.so -I$(INC) -I$(INC_MKL) $(MKLLIB)

2dPWM_fftw: interpolation.o prep_operators.o phase_shifts.o imaging_conditions.o fftw_fft.o
	$(CXX) -shared $(CXXFLAGS) -DENABLE_FFTW_PWM *.o PWM_cpu.cpp -o PWM_cpu.so -I$(INC) -I$(INC_FFTW) $(FFTWLIB)

mkl_fft.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)mkl_fft.cpp -I$(INC) -I$(INC_MKL)

fftw_fft.o:
	$(CXX) $(CXXFLAGS) -c $(SRC)fftw_fft.cpp -I$(INC) -I$(INC_FFTW)

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
	
