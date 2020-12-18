
#include "interpolate.h"

extern "C"

{

// FUNCTION SCOPE: Use the per-position precalculated coefficients to interpolate ref. wavefields
//     and obtain the next depth wavefield.
// INPUT parameters:
//    - nf : number of frequencies
//    - nx : number of lateral positions
//    - nref : number of ref. velocities
//    - coeff : "2D" array of coefficients for interpolation with the ref. wavefields
//    - refWavefields : "3D" array of reference wavefields
// OUTPUT:
//    - finalWavefield : "2D" array of final (interpolated) wavefield
void interpolate(int nf, int nx, int nref, float * coeff, \
    fcomp * refWavefields, fcomp * finalWavefield){

    int shotSz = nf*nx;

    #pragma omp parallel for schedule(dynamic,1)
    for(int j=0; j<nf; ++j){
        for(int i=0; i<nx; ++i){
            finalWavefield[j*nx + i] = fcomp(0.0,0.0);
            for(int n=0; n<nref; ++n){
                finalWavefield[j*nx + i] += coeff[i*nref + n] * refWavefields[n*shotSz + j*nx + i];
            }
        }
    }
}



// FUNCTION SCOPE: Calculate coefficients for all ref. velocities for each lateral position based on 'distance'
//      between exact and reference velocity.
// INPUT parameters:
//    - nx : number of lateral positions
//    - nxVels : array with exact velocities
//    - nref: number of reference velocities
//    - refVels: reference velocities
// OUTPUT:
//    - coeff: coefficients (should be of size: nx * nref * sizeof(float))
void find_coeff(int nx, float * nxVels, int nref, float * refVels, float * coeff){
    
    float e = 0.0001; // necessary to avoid division by 0. small though to have negligible contribution

    for(int i=0; i<nx; ++i){

        float exactVel = nxVels[i];
        
        for(int n=0; n<nref; ++n)
            coeff[i*nref + n] = 1.0 / ( std::abs(exactVel - refVels[n]) + e );

    }

}



// FUNCTION SCOPE: normalize coefficients (in-place)
// INPUT parameters:
//    - nx : number of lateral positions
//    - nref: number of reference velocities
// OUTPUT:
//    - coeff: normalized coefficients for all ref. velocities for each lateral position (should be of size nx * nref * sizeof(float))
void norm_coeff(int nx, int nref, float * coeff){

    for(int i=0; i<nx; ++i){
        
        float sum_ = 0.0;

        for(int n=0; n<nref; ++n)
            sum_ += coeff[i*nref + n]; // sum all coefficients for i'th position
        
        float A = 1.0 / sum_; // find normalization factor A

        for(int n=0; n<nref; ++n)
            coeff[i*nref + n] *= A; // normalize coefficients for i'th position
        
    }

}

} //end extern "C"
