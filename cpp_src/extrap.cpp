
#include "types.h"
#include <iostream>

extern "C"

{

// FUNCTION SCOPE: Forward Phase-Shift in the frequency-space domain for 
//    a set of ns wavefields/shot-records.
// INPUT parameters:
//    - ns : number of shots
//    - nf : number of frequencies
//    - nx : number of lateral positions
// OUTPUT:
//    - wavefields : "3D" array of ns 2D wavefields to-be-shifted in-place.
void phase_shift_forw(int ns, int nf, int nx, fcomp * wavefields,
    float * velSlide, float * omega, float dz){

    int shotSz = nf*nx; // size of wavefields/shot records

    for(int j=0; j<nf; ++j){
        for(int i=0; i<nx; ++i){

            float vel = velSlide[i];
            float k = omega[j] / vel;    
            fcomp term = std::exp( -fcomp(0.0,1.0) * k * dz );

            for(int is=0; is<ns; ++is)
                wavefields[is*shotSz + j*nx + i] *= term;

        }
    }
}



// FUNCTION SCOPE: Backward Phase-Shift in the frequency-space domain for 
//    a set of ns wavefields/shot-records.
// INPUT parameters:
//    - ns : number of shots
//    - nf : number of frequencies
//    - nx : number of lateral positions
// OUTPUT:
//    - wavefields : "3D" array of ns 2D wavefields to-be-shifted in-place.
void phase_shift_back(int ns, int nf, int nx, fcomp * wavefields,
    float * velSlide, float * omega, float dz){

    int shotSz = nf*nx; // size of wavefields/shot records

    for(int j=0; j<nf; ++j){
        for(int i=0; i<nx; ++i){

            float vel = velSlide[i];
            float k = omega[j] / vel;    
            fcomp term = std::exp( +fcomp(0.0,1.0) * k * dz );

            for(int is=0; is<ns; ++is)
                wavefields[is*shotSz + j*nx + i] *= term;

        }
    }
}


} // end extern "C"
