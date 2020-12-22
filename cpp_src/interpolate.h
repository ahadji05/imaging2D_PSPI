
#include <cmath>
#include "types.h"
#include <iostream>

extern "C"

{

void interpolate(int ns, int nf, int nx, int nref, float * coeff, fcomp * refWavefields, fcomp * finalWavefield);
void find_coeff(int nx, float * nxVels, int nref, float * refVels, float * coeff);
void norm_coeff(int nx, int nref, float * coeff);

__global__ void interpolate_cu(int ns, int nf, int nx, int nref, \
    float * coeff, fcomp * ref, fcomp * base);

} // end extern "C"