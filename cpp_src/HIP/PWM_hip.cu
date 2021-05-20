
#include <iostream>
#include <cstring>
#include <cmath>

#include "imaging_conditions.hpp"
#include "device_ROCm_fft.hpp"
#include "prep_operators.hpp"
#include "interpolation.hpp"
#include "phase_shifts.hpp"
#include "types.hpp"
#include "timer.hpp"
#include "utils.hpp"

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
void extrapAndImag_amd_cu(int ns, int nref, int nz, int nextrap, int nt, int nf, int nx, \
    float dz, int Nk, float kmax, int Nkx, float * omega, float * kx, float * velmod, \
    fcomp * forw_wf, fcomp * back_wf, float * image){

    tall.start();

    int imgSize = nz * nx; // images size
    int wfSize  = nf * nx; // wavefields size

    fcomp * base_forw = new fcomp[ns*wfSize];
    fcomp * base_back = new fcomp[ns*wfSize];
    fcomp * ref_forw  = new fcomp[ns*nref*wfSize];
    fcomp * ref_back  = new fcomp[ns*nref*wfSize];
    fcomp * h_image   = new fcomp[ns*nz*nx];
    fcomp * d_base_forw, * d_base_back, * d_ref_forw, * d_ref_back, \
          * d_ops_forw,  * d_ops_back,  * d_image;
    float * d_coeff, * d_velmod, * d_omega;
    int * d_opIndices;
    hipMalloc(&d_base_forw, ns*wfSize*sizeof(fcomp));
    hipMalloc(&d_base_back, ns*wfSize*sizeof(fcomp));
    hipMalloc(&d_ops_forw,  Nk*Nkx*sizeof(fcomp));
    hipMalloc(&d_ops_back,  Nk*Nkx*sizeof(fcomp));
    hipMalloc(&d_opIndices, nextrap*nref*nf*sizeof(int));
    hipMalloc(&d_coeff,     nextrap*nref*nx*sizeof(float));
    hipMalloc(&d_velmod,    nz*nx*sizeof(float));
    hipMalloc(&d_omega,     Nk*sizeof(float));
    hipMalloc(&d_ref_forw,  ns*nref*wfSize*sizeof(fcomp));
    hipMalloc(&d_ref_back,  ns*nref*wfSize*sizeof(fcomp));
    hipMalloc(&d_image,     ns*nz*nx*sizeof(fcomp));

    // read wavefields to new storages selecting propagating frequencies only (nf < nt)!
    for(int s=0; s<ns; ++s){
        std::memcpy(&base_forw[s*wfSize], &forw_wf[s*nt*nx], wfSize*sizeof(fcomp));
        std::memcpy(&base_back[s*wfSize], &back_wf[s*nt*nx], wfSize*sizeof(fcomp));
    }

    // prepare table of operators
    t3.start();
    f_kx_operators forwOps(Nk, kmax, Nkx, dz, 'f', kx); // 'f' : forward-in-time
    f_kx_operators backOps(Nk, kmax, Nkx, dz, 'b', kx); // 'b' : backward-in-time
    
    // prepare look-up indices & interpolation coefficients
    int   * Idx   = prep_lookUp_indices(nf, omega, nextrap, nx, velmod, nref, Nk, forwOps.k.data());
    float * coeff = prep_interpolation_coeff(velmod, nextrap, nref, nx);
    t3.stop();

    hipMemcpy(d_base_forw, base_forw,             ns*wfSize*sizeof(fcomp),       hipMemcpyHostToDevice);
    hipMemcpy(d_base_back, base_back,             ns*wfSize*sizeof(fcomp),       hipMemcpyHostToDevice);
    hipMemcpy(d_ops_forw,  forwOps.values.data(), Nk*Nkx*sizeof(fcomp),          hipMemcpyHostToDevice);
    hipMemcpy(d_ops_back,  backOps.values.data(), Nk*Nkx*sizeof(fcomp),          hipMemcpyHostToDevice);
    hipMemcpy(d_opIndices, Idx,                   nextrap*nref*nf*sizeof(int),   hipMemcpyHostToDevice);
    hipMemcpy(d_coeff,     coeff,                 nextrap*nref*nx*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_velmod,    velmod,                nz*nx*sizeof(float),           hipMemcpyHostToDevice);
    hipMemcpy(d_omega,     omega,                 Nk*sizeof(float),              hipMemcpyHostToDevice);
    hipMemset(d_image,     0,                     2*ns*nz*nx*sizeof(float));

    dim3 nThreads(64, 1, 1);
    size_t nBlocks_x = nx % nThreads.x == 0 ? size_t(nx/nThreads.x) : size_t(1 + nx/nThreads.x);
    dim3 nBlocks(nBlocks_x, nf, ns);
    dim3 nBlocks_ps(nBlocks_x, nf, 1);
    dim3 nBlocks_img(nBlocks_x, ns, 1);

    //create plans for hipFFTs
    hipfftHandle planBase = make_hipFFTplan_Batched1dSignals(ns*nf, nx);
    hipfftHandle planRef  = make_hipFFTplan_Batched1dSignals(ns*nref*nf, nx);

    for( int l = 0; l < nextrap; ++l ){ // start loop over depths

        std::cout << "Depth " << l << "\n";

        // phase-shifts in the f-x domain.
        t5.start();
        hipLaunchKernelGGL( phase_shift_forw_cu, nBlocks_ps, nThreads, 0, 0,\
            ns, nf, nx, &d_velmod[l*nx], d_omega, dz, d_base_forw );
        hipLaunchKernelGGL( phase_shift_forw_cu, nBlocks_ps, nThreads, 0, 0,\
            ns, nf, nx, &d_velmod[l*nx], d_omega, dz, d_base_back );
        hipDeviceSynchronize();
        t5.stop();

        // do FFTs : f-x -> f-kx
        t4.start();
        hipfftFORW_Batched1dSignals( d_base_forw, &planBase );
        hipfftFORW_Batched1dSignals( d_base_back, &planBase );
        hipDeviceSynchronize();
        t4.stop();

        // propagate the base wavefields to reference wavefields
        t0.start();
        hipLaunchKernelGGL( extrap_ref_wavefields_cu, nBlocks, nThreads, 0, 0,\
            ns, nf, nx, nref, d_ref_forw, d_base_forw, &d_opIndices[l*nf*nref], d_ops_forw );
        hipLaunchKernelGGL( extrap_ref_wavefields_cu, nBlocks, nThreads, 0, 0,\
            ns, nf, nx, nref, d_ref_back, d_base_back, &d_opIndices[l*nf*nref], d_ops_back );
        hipDeviceSynchronize();
        t0.stop();

        // do IFFTs : f-kx -> f-x
        t4.start();
        hipfftBACK_Batched1dSignals(d_ref_forw, ns*nref*nf, nx, &planRef);
        hipfftBACK_Batched1dSignals(d_ref_back, ns*nref*nf, nx, &planRef);
        hipDeviceSynchronize();
        t4.stop();

        // interpolation : from ref. wavefields to base wavefields
        t2.start();
        hipLaunchKernelGGL( interpolate_cu, nBlocks, nThreads, 0, 0,\
            ns, nf, nx, nref, &d_coeff[l*nref*nx], d_ref_forw, d_base_forw );
        hipLaunchKernelGGL( interpolate_cu, nBlocks, nThreads, 0, 0,\
            ns, nf, nx, nref, &d_coeff[l*nref*nx], d_ref_back, d_base_back );
        hipDeviceSynchronize();
        t2.stop();
        
        // image depth slide
        t1.start();
        hipLaunchKernelGGL( imaging_cu, nBlocks_img, nThreads, 0, 0, d_image, d_base_forw, d_base_back, ns, nf, nx, l, imgSize);
        hipDeviceSynchronize();
        t1.stop();

    } // end loop over threads

    hipMemcpy(h_image, d_image, ns*nz*nx*sizeof(fcomp), hipMemcpyDeviceToHost);
    for(int s=0; s<ns; ++s)
        for(int l=0; l<nextrap; ++l)
            for(int i=0; i<nx; ++i)
                image[s*nz*nx + l*nx + i] = reinterpret_cast<float*>(h_image)[2*(s*nz*nx + l*nx + i)];
    
    hipfftDestroy(planBase);
    hipfftDestroy(planRef);
    hipFree(d_base_forw);
    hipFree(d_base_back);
    hipFree(d_ref_forw);
    hipFree(d_ref_back);
    hipFree(d_image);
    hipFree(d_ops_forw);
    hipFree(d_ops_back);
    hipFree(d_opIndices);
    hipFree(d_coeff);
    hipFree(d_velmod);
    hipFree(d_omega);
    delete [] h_image;
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
} // end extrapAndImag_amd_cu

} // end extern "C"
