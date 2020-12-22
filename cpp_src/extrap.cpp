
#include <iostream>
#include <cstring>
#include <cmath>

#include "imag_condition.h"
#include "interpolate.h"
#include "mkl_fft.h"
#include "prepOps.h"
#include "types.h"
#include "timer.h"

extern "C"

{

timer tall("total time");
timer t0("ref. wavefields extrap.");
timer t1("imaging");
timer t2("interpolation");
timer t3("prepOps");
timer t4("FFTs");
timer t5("phase-shift");

/* ----------------------- Forward function declarations ----------------------- */
void phase_shift_forw(int , int , int , fcomp * , float * , float * , float );
void phase_shift_back(int , int , int , fcomp * , float * , float * , float );
float find_min_vel(int , float * );
float find_max_vel(int , float * );
void define_ref_velocities(int , float , float , float * );
void extrap_ref_wavefields(fcomp * , fcomp * , fcomp * , int * , int, int , int , int );
int * prep_lookUp_indices(int , float * , int , int , float * , int , int , float * );
float * prep_interpolation_coeff(float * , int , int , int );
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

    tall.start();

    int imgSize = nz * nx; // images size
    int wfSize = nf * nx; // wavefields size

    fcomp * base_forw = new fcomp[ns*wfSize];
    fcomp * base_back = new fcomp[ns*wfSize];
    fcomp * ref_forw = new fcomp[ns*nref*wfSize];
    fcomp * ref_back = new fcomp[ns*nref*wfSize];
    std::memset(image, 0, ns*nz*nx*sizeof(float));

    // read wavefields to new storages selecting propagating frequencies only (nf < nt)!
    for(int s=0; s<ns; ++s){
        for(int j=0; j<nf; ++j){
            for(int i=0; i<nx; ++i){
                base_forw[s*wfSize + j*nx + i] = forw_wf[s*nt*nx + j*nx + i];
                base_back[s*wfSize + j*nx + i] = back_wf[s*nt*nx + j*nx + i];
            }
        }
    }

    // prepare table of operators
    t3.start();
    f_kx_operators forwOps(Nk, kmax, Nkx, dz, 'f', kx); // 'f' : forward-in-time
    f_kx_operators backOps(Nk, kmax, Nkx, dz, 'b', kx); // 'b' : backward-in-time
    
    // prepare look-up indices & interpolation coefficients
    int * Idx = prep_lookUp_indices(nf, omega, nextrap, nx, velmod, nref, Nk, forwOps.k.data());
    float * coeff = prep_interpolation_coeff(velmod, nextrap, nref, nx);
    t3.stop();

    for(int l=0; l<nextrap; ++l){ // start loop over depths

        std::cout << "Depth " << l << "\n";

        // phase-shifts in the f-x domain.
        t5.start();
        phase_shift_forw(ns, nf, nx, base_forw, &velmod[l*nx], omega, dz);
        phase_shift_back(ns, nf, nx, base_back, &velmod[l*nx], omega, dz);
        t5.stop();

        // do FFTs : f-x -> f-kx
        t4.start();
        fft1dforwardFrom2Darray(base_forw, ns*nf, nx, 1);
        fft1dforwardFrom2Darray(base_back, ns*nf, nx, 1);
        t4.stop();

        // propagate the base wavefields to reference wavefields
        t0.start();
        extrap_ref_wavefields(ref_forw, base_forw, forwOps.values.data(), \
            &Idx[l*nf*nref], ns, nref, nf, nx);
        extrap_ref_wavefields(ref_back, base_back, backOps.values.data(), \
            &Idx[l*nf*nref], ns, nref, nf, nx);
        t0.stop();

        // do IFFTs : f-kx -> f-x
        t4.start();
        fft1dbackwardFrom2Darray(ref_forw, ns*nref*nf, nx, 1);
        fft1dbackwardFrom2Darray(ref_back, ns*nref*nf, nx, 1);            
        t4.stop();
        
        // interpolation : from ref. wavefields to base wavefields
        t2.start();
        interpolate(ns, nf, nx, nref, &coeff[l*nref*nx], ref_forw, base_forw);
        interpolate(ns, nf, nx, nref, &coeff[l*nref*nx], ref_back, base_back);
        t2.stop();

        // image depth slide
        t1.start();
        cross_corr(ns, wfSize, nx, nf, base_forw, base_back, imgSize, l, image);
        t1.stop();

    } // end loop over depths

    delete [] coeff;
    delete [] Idx;
    delete [] base_forw;
    delete [] base_back;
    delete [] ref_forw;
    delete [] ref_back;
    tall.stop();

    tall.dispInfo();
    t0.dispInfo();
    t1.dispInfo();
    t2.dispInfo();
    t3.dispInfo();
    t4.dispInfo();
    t5.dispInfo();
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

    #pragma omp parallel for schedule(dynamic,1)
    for(int j=0; j<nf; ++j){
        for(int i=0; i<nx; ++i){

            float vel = velSlide[i];
            float k = omega[j] / vel;    
            fcomp term = thrust::exp( -fcomp(0.0,1.0) * k * dz );

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
            fcomp term = thrust::exp( +fcomp(0.0,1.0) * k * dz );

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

float * prep_interpolation_coeff(float * velmod, int nextrap, int nref, int nx){

    float * coeff = new float[nextrap*nref*nx];
    float * refVels = new float[nref];

    for(int l=0; l<nextrap; ++l){
        float vmin = find_min_vel(nx, &velmod[l*nx]);
        float vmax = find_max_vel(nx, &velmod[l*nx]);
        define_ref_velocities(nref, vmin, vmax, refVels);
        find_coeff(nx, &velmod[l*nx], nref, refVels, &coeff[l*nref*nx]);
        norm_coeff(nx, nref, &coeff[l*nref*nx]);
    }

    delete [] refVels;
    return coeff;
}

} // end extern "C"
