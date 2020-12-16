
#include <iostream>
#include<cstring>
#include <fstream>
#include <cmath>

#include "interpolate.h"
#include "imag_condition.h"
#include "mkl_fft.h"
#include "prepOps.h"
#include "types.h"

extern "C"

{


/* ----------------------- Forward function declarations ----------------------- */
void phase_shift_forw(int , int , int , fcomp * , float * , float * , float );
void phase_shift_back(int , int , int , fcomp * , float * , float * , float );
float find_min_vel(int , float * );
float find_max_vel(int , float * );
void define_ref_velocities(int , float , float , float * );
void extrap_ref_wavefields(fcomp * , fcomp * , fcomp * , int * , int, int , int , int );
int * prep_lookUp_indices(int , float * , int , int , float * , int , int , float * );

bool assertForNanComplex(int , fcomp * , std::string );
bool assertForNanFloat(int , float * , std::string );
/* -------------------------------------------------------------------------------*/



// FUNCTION SCOPE: migration of 2D wavefields using PSPI with a fixed number of
//    propagation velocities as defined according to the min and max velocities
//    in each depth slide.
void extrapAndImag(int ns, int nref, int nz, int nextrap, int nt, int nf, int nx, \
    float dz, int Nk, float kmax, int Nkx, \
    float * omega, float * kx, float * velmod, \
    fcomp * forw_wf, fcomp * back_wf, float * image){

    int imgSize = nz * nx;
    int wfSize = nf * nx;
    bool check; // to check for Nan values

    float * refVels = new float[nref];
    float * coeff = new float[nref*nx];
    fcomp * base_forw = new fcomp[ns*wfSize];
    fcomp * base_back = new fcomp[ns*wfSize];
    fcomp * ref_forw = new fcomp[ns*nref*wfSize];
    fcomp * ref_back = new fcomp[ns*nref*wfSize];
    std::memset(image, 0, ns*nz*nx*sizeof(float));

    // read wavefields to new storages selecting propagating frequencies only (nf < nt)!
    for(int s=0; s<ns; ++s){
        std::memcpy(&base_forw[s*wfSize], &forw_wf[s*nt*nx], wfSize*sizeof(fcomp));
        std::memcpy(&base_back[s*wfSize], &back_wf[s*nt*nx], wfSize*sizeof(fcomp));
    }
    check = assertForNanComplex(ns*nf*nx, base_forw, "nan in copied base_forw");
    if (check == true) exit(1);
    check = assertForNanComplex(ns*nf*nx, base_back, "nan in copied base_back");    
    if (check == true) exit(1);

    // prepare table of operators
    f_kx_operators forwOps(Nk, kmax, Nkx, dz, 'f', &kx[0]); // 'f' : forward-in-time
    f_kx_operators backOps(Nk, kmax, Nkx, dz, 'b', &kx[0]); // 'b' : backward-in-time

    // prepare look-up indices
    int * Idx = prep_lookUp_indices(nf, &omega[0], nextrap, nx, &velmod[0], nref, Nk, forwOps.k.data());

    for(int l=0; l<nextrap; ++l){ // start loop over depths

        std::cout << "depth " << l << "\n";

        // define the coefficients for interpolation
        float vmin = find_min_vel(nx, &velmod[l*nx]);
        float vmax = find_max_vel(nx, &velmod[l*nx]);
        define_ref_velocities(nref, vmin, vmax, &refVels[0]);
        find_coeff(nx, &velmod[l*nx], nref, &refVels[0], &coeff[0]);
        norm_coeff(nx, nref, &coeff[0]);

        // phase-shifts in the f-x domain.
        phase_shift_forw(ns, nf, nx, &base_forw[0], &velmod[l*nx], &omega[0], dz);
        phase_shift_back(ns, nf, nx, &base_back[0], &velmod[l*nx], &omega[0], dz);

        // do FFTs : f-x -> f-kx
        for(int s=0; s<ns; ++s){
            fft1dforwardFrom2Darray(&base_forw[s*wfSize], nf, nx, 1);
            fft1dforwardFrom2Darray(&base_back[s*wfSize], nf, nx, 1);
        }

        // propagate the base wavefields to reference wavefields
        extrap_ref_wavefields(&ref_forw[0], &base_forw[0], forwOps.values.data(), \
            &Idx[l*nf*nref], ns, nref, nf, nx);
        extrap_ref_wavefields(&ref_back[0], &base_back[0], backOps.values.data(), \
            &Idx[l*nf*nref], ns, nref, nf, nx);

        // do IFFTs : f-kx -> f-x
        for(int s=0; s<ns; ++s)
            for(int n=0; n<nref; ++n){
                fft1dbackwardFrom2Darray(&ref_forw[s*nref*wfSize + n*wfSize], nf, nx, 1);
                fft1dbackwardFrom2Darray(&ref_back[s*nref*wfSize + n*wfSize], nf, nx, 1);
            }
        
        // interpolation : from ref. wavefields to base wavefields
        for(int s=0; s<ns; ++s){
            interpolate(nf, nx, nref, &coeff[0], &ref_forw[s*nf*nref*nx], &base_forw[s*nf*nx]);
            interpolate(nf, nx, nref, &coeff[0], &ref_back[s*nf*nref*nx], &base_back[s*nf*nx]);
        }
        check = assertForNanComplex(ns*nf*nx, base_forw, "extrap base_forw: nan appeared\n");
        if (check == true) exit(1);
        check = assertForNanComplex(ns*nf*nx, base_back, "extrap base_back: nan appeared\n");
        if (check == true) exit(1);

        // image depth slide
        cross_corr(ns, wfSize, nx, nf, base_forw, base_back, imgSize, l, image);
        check = assertForNanFloat(ns*nz*nx, image, "image: nan appeared\n");
        if (check == true) exit(1);

    } // end loop over depths

    delete [] Idx;
    delete [] base_forw;
    delete [] base_back;
    delete [] ref_forw;
    delete [] ref_back;
    delete [] coeff;
    delete [] refVels;
}

bool assertForNanComplex(int N, fcomp * array, std::string text){
    for(int i=0; i<N; ++i){
        float re = reinterpret_cast<float*>(array)[2*N];
        float im = reinterpret_cast<float*>(array)[2*N+1];
        if(std::isnan(re) || std::isnan(im)){
            std::cout << text << "\n";
            return true;
        }
    }
    return false;
}

bool assertForNanFloat(int N, float * array, std::string text){
    for(int i=0; i<N; ++i){
        if(std::isnan(array[i])){
            std::cout << text << "\n";
            return true;
        }
    }
    return false;
}

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

    for(int s=0; s<ns; ++s){
        int sIdx = s * nref * nf * nx;
        for(int n=0; n<nref; ++n){
            int velIdx = n * nf * nx;
            for(int j=0; j<nf; ++j){
                int fIdx = j * nx;
                int opIdx = idx[j*nref + n] * nx; //operator's index(start-position)
                for(int xIdx=0; xIdx<nx; ++xIdx){

                    refWfds[sIdx + velIdx + fIdx + xIdx] = tableOps[opIdx + xIdx] * \
                        baseWf[s*nf*nx + fIdx + xIdx];
                }
            }
        }
    }
}



float find_min_vel(int N, float * refVels){
    
    float min_vel = 1000000.0; // some very large value

    for(int i=0; i<N; ++i){
        if (refVels[i] < min_vel)
            min_vel = refVels[i];
    }

    return min_vel;
}



float find_max_vel(int N, float * refVels){
    
    float max_vel = 0.0; // some very small value

    for(int i=0; i<N; ++i){
        if (refVels[i] > max_vel)
            max_vel = refVels[i];
    }

    return max_vel;
}



void define_ref_velocities(int nref, float min_vel, float max_vel, float * refVels){

    int N = nref-1;
    float dvel = (max_vel - min_vel) / N;

    for(int i=0; i<nref; ++i)
        refVels[i] = min_vel + i*dvel;

}



int * prep_lookUp_indices(int Nf, float * omega, int Nextrap, int Nx, float * velmod, \
    int Nref, int Nk, float * k){

    int tableOfIndicesSize = Nextrap * Nf * Nref;

    int * table = new int[tableOfIndicesSize];
    float * refVels = new float[Nref];

    for(int l=0; l<Nextrap; ++l){

        //define ref. velocities for current depth!
        float minVel = find_min_vel(Nx, &velmod[l*Nx]);
        float maxVel = find_max_vel(Nx, &velmod[l*Nx]);
        define_ref_velocities(Nref, minVel, maxVel, &refVels[0]);
        
        for(int j=0; j<Nf; ++j)
            for(int n=0; n<Nref; ++n){

                float kref = omega[j] / refVels[n];

                table[l*Nf*Nref + j*Nref + n] = chooseOperatorIndex(Nk, &k[0], kref);
            
            }
    }
    
    delete [] refVels;

/*check for Nan values*/
    for(int i=0; i<tableOfIndicesSize; ++i){
        if( std::isnan(table[i])){
            std::cout << "Nan value identified\n";
            exit(1);
        }
    }

/*check for invalid values*/
    for(int i=0; i<tableOfIndicesSize; ++i){
        if( table[i] >= Nk ){
            std::cout << "index cannot be >= Nk\n";
            exit(1);
        }
    }

    return table;
}

} // end extern "C"
