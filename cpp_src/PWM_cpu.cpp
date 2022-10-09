
#include <iostream>
#include <cstring>
#include <cmath>

#include "interpolation.hpp"
#include "imaging_conditions.hpp"
#include "phase_shifts.hpp"
#include "prep_operators.hpp"
#include "timer.hpp"
#include "utils.hpp"

#ifdef ENABLE_FFTW_PWM
#include "fftw_fft.hpp"
#else
#include "mkl_fft.hpp"
#endif

extern "C"

{

timer tall("total time");
timer t0("ref. wavefields extrap.");
timer t1("imaging");
timer t2("interpolation");
timer t3("prepOps");
timer t4("FFTs");
timer t5("phase-shift");


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
#ifdef ENABLE_FFTW_PWM
	int fftw_forward = -1;
	batched1dffts(base_forw, ns*nf, nx, fftw_forward);
	batched1dffts(base_back, ns*nf, nx, fftw_forward);
#else
        fft1dforwardFrom2Darray(base_forw, ns*nf, nx, 1);
        fft1dforwardFrom2Darray(base_back, ns*nf, nx, 1);
#endif
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
#ifdef ENABLE_FFTW_PWM
	int fft_inverse = +1;
	batched1dffts(ref_forw, ns*nref*nf, nx, fft_inverse);
	batched1dffts(ref_back, ns*nref*nf, nx, fft_inverse);
#else
        fft1dbackwardFrom2Darray(ref_forw, ns*nref*nf, nx, 1);
        fft1dbackwardFrom2Darray(ref_back, ns*nref*nf, nx, 1);            
#endif
	t4.stop();
        
        // interpolation : from ref. wavefields to base wavefields
        t2.start();
        interpolate(ns, nf, nx, nref, &coeff[l*nref*nx], ref_forw, base_forw);
        interpolate(ns, nf, nx, nref, &coeff[l*nref*nx], ref_back, base_back);
        t2.stop();

        // image depth slide
        t1.start();
        imaging(ns, wfSize, nx, nf, base_forw, base_back, imgSize, l, image);
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

} // end extern "C"
