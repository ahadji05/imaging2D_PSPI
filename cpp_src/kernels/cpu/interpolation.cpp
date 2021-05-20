
#include "interpolation.hpp"

extern "C"

{

// FUNCTION SCOPE: Use the per-position precalculated coefficients to interpolate ref. wavefields
//     and obtain the next depth wavefield.
// INPUT parameters:
//    - ns : number of shot records(sources)
//    - nf : number of frequencies
//    - nx : number of lateral positions
//    - nref : number of ref. velocities
//    - coeff : "2D" array of coefficients for interpolation with the ref. wavefields
//    - refWavefields : "3D" array of reference wavefields
// OUTPUT:
//    - finalWavefield : "2D" array of final (interpolated) wavefield
void interpolate(int ns, int nf, int nx, int nref, float * coeff, \
    fcomp * refWavefields, fcomp * finalWavefield){

    int shotSz = nf*nx;
    #pragma omp parallel for schedule(dynamic,64)
    for(int i=0; i<ns*nf*nx; ++i)
        finalWavefield[i] = fcomp(0.0,0.0);

    #pragma omp parallel for schedule(dynamic,1)
    for(int j=0; j<nf; ++j){
        for(int s=0; s<ns; ++s){
            int sRef = s*nref*nf*nx;
            for(int n=0; n<nref; ++n){
                for(int i=0; i<nx; ++i){
                finalWavefield[s*nf*nx + j*nx + i] += coeff[i*nref + n] * \
                    refWavefields[sRef + n*shotSz + j*nx + i];
                }
            }
        }
    }
}

} //end extern "C"
