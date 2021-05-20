
#include "phase_shifts.hpp"

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

        #pragma omp parallel for schedule(dynamic,1)
        for(int j=0; j<nf; ++j){
            for(int i=0; i<nx; ++i){

                float vel = velSlide[i];
                float k = omega[j] / vel;    
                fcomp term = pwm::exp( -fcomp(0.0,1.0) * k * dz );

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

        #pragma omp parallel for schedule(dynamic,1)
        for(int j=0; j<nf; ++j){
            for(int i=0; i<nx; ++i){

                float vel = velSlide[i];
                float k = omega[j] / vel;    
                fcomp term = pwm::exp( +fcomp(0.0,1.0) * k * dz );

                for(int is=0; is<ns; ++is)
                    wavefields[is*shotSz + j*nx + i] *= term;

            }
        }
    }


    // FUNCTION SCOPE: take a wavefield(baseWf) of size = nf*nx and propagate it using a set(# = nref) different velicities.
    // INPUT parameters:
    //    - baseWf : this is the base wavefield that is going to be extrapolated to a dif set of ferent wavefields using
    //    a set of different propagation velocities.
    //    - tableOps : look-up table of operators.
    //    - idx :  an array of length = nref*nf integers where each value represents the starting-point of the operator to use
    //    per velocity and frequency from the look-up table of operators.
    //    - nref : number of reference velocities
    //    - nf : number of frequencies (non-contiguous dimension)
    //    - nx : number of spatial points (contiguous dimension)
    // OUTPUT:
    //    - refWfds : resultant extrapolated wavefields
    void extrap_ref_wavefields(fcomp * refWfds, fcomp * baseWf, fcomp * tableOps, \
        int * idx, int ns, int nref, int nf, int nx){

        #pragma omp parallel for schedule(dynamic,1)
        for(int j=0; j<nf; ++j){
                int fIdx = j * nx;

            for(int n=0; n<nref; ++n){
                int velIdx = n * nf * nx;

                int opIdx = idx[j*nref + n] * nx; //operator's index(start-position)

                for(int s=0; s<ns; ++s){
                    int sIdx = s * nref * nf * nx;
                    for(int xIdx=0; xIdx<nx; ++xIdx)
                        refWfds[sIdx + velIdx + fIdx + xIdx] = tableOps[opIdx + xIdx] * \
                            baseWf[s*nf*nx + fIdx + xIdx];
                    
                } // end loop over source locations

            } // end loop over reference velocities
        } //end loop over frequencies
    }

} // end extern "C"