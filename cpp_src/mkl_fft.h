#ifndef MKL_FFT_H
#define MKL_FFT_H

#include "types.h"
#include "mkl.h"

extern "C"

{

void c64fft1dforw(fcomp * data, int N);
void c64fft1dback(fcomp * data, int N);
void fft1dforwardFrom2Darray(fcomp * data, int N1, int N2, int axis);
void fft1dbackwardFrom2Darray(fcomp * data, int N1, int N2, int axis);

} // end extren "C"

#endif