
#include <iostream>
#include <cstring>
#include <cmath>

#include "deviceCUDA_FFTs.h"
#include "imag_condition.h"
#include "interpolate.h"
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
__global__ void phase_shift_forw_cu(int ns, int nf, int nx, float * velSlide, \
    float * omega, float dz, fcomp * wavefield);
__global__ void phase_shift_back_cu(int ns, int nf, int nx, float * velSlide, \
    float * omega, float dz, fcomp * wavefield);
__global__ void extrap_ref_wavefields_cu(int ns, int nf, int nx, int nref, \
    fcomp * ref, fcomp * base,
    int * opIndices, fcomp * table);

float find_min_vel(int , float * );
float find_max_vel(int , float * );
void define_ref_velocities(int , float , float , float * );
int * prep_lookUp_indices(int , float * , int , int , float * , int , int , float * );
float * prep_interpolation_coeff(float * , int , int , int );
/* -------------------------------------------------------------------------------*/



// FUNCTION SCOPE: migration of 2D wavefields using PSPI with a fixed number of
//    propagation velocities as defined according to the min and max velocities
//    in each depth slide.
void extrapAndImag_cu(int ns, int nref, int nz, int nextrap, int nt, int nf, int nx, \
    float dz, int Nk, float kmax, int Nkx, float * omega, float * kx, float * velmod, \
    fcomp * forw_wf, fcomp * back_wf, float * image){

    tall.start();

    int imgSize = nz * nx; // images size
    int wfSize = nf * nx; // wavefields size

    fcomp * base_forw = new fcomp[ns*wfSize];
    fcomp * base_back = new fcomp[ns*wfSize];
    fcomp * ref_forw = new fcomp[ns*nref*wfSize];
    fcomp * ref_back = new fcomp[ns*nref*wfSize];
    fcomp * h_image = new fcomp[ns*nz*nx];
    fcomp * d_base_forw, * d_base_back, * d_ref_forw, * d_ref_back, \
        * d_ops_forw, * d_ops_back, * d_image;
    float * d_coeff, * d_velmod, * d_omega;
    int * d_opIndices;
    cudaMalloc(&d_base_forw, ns*wfSize*sizeof(fcomp));
    cudaMalloc(&d_base_back, ns*wfSize*sizeof(fcomp));
    cudaMalloc(&d_ref_forw, ns*nref*wfSize*sizeof(fcomp));
    cudaMalloc(&d_ref_back, ns*nref*wfSize*sizeof(fcomp));
    cudaMalloc(&d_ops_forw, Nk*Nkx*sizeof(fcomp));
    cudaMalloc(&d_ops_back, Nk*Nkx*sizeof(fcomp));
    cudaMalloc(&d_coeff, nextrap*nref*nx*sizeof(float));
    cudaMalloc(&d_image, ns*nz*nx*sizeof(fcomp));
    cudaMalloc(&d_opIndices, nextrap*nref*nf*sizeof(int));
    cudaMalloc(&d_velmod, nz*nx*sizeof(float));
    cudaMalloc(&d_omega, Nk*sizeof(float));

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
    int * Idx = prep_lookUp_indices(nf, omega, nextrap, nx, velmod, nref, Nk, forwOps.k.data());
    float * coeff = prep_interpolation_coeff(velmod, nextrap, nref, nx);
    t3.stop();

    cudaMemcpy(d_base_forw, base_forw, ns*wfSize*sizeof(fcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base_back, base_back, ns*wfSize*sizeof(fcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ops_forw, forwOps.values.data(), Nk*Nkx*sizeof(fcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ops_back, backOps.values.data(), Nk*Nkx*sizeof(fcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_opIndices, Idx, nextrap*nref*nf*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeff, coeff, nextrap*nref*nx*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velmod, velmod, nz*nx*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega, omega, Nk*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_image, 0, ns*nz*nx*sizeof(float));

    dim3 nThreads(32, 1, 1);
    size_t nBlocks_x = nx % nThreads.x == 0 ? size_t(nx/nThreads.x) : size_t(1 + nx/nThreads.x);
    dim3 nBlocks(nBlocks_x, nf, ns);
    dim3 nBlocks_ps(nBlocks_x, nf, 1);
    dim3 nBlocks_img(nBlocks_x, ns, 1);

    //create plans for cuFFTs
    cufftHandle planBase = make_cuFFTplan_Batched1dSignals(ns*nf, nx);
    cufftHandle planRef = make_cuFFTplan_Batched1dSignals(ns*nref*nf, nx);

    for(int l=0; l<nextrap; ++l){ // start loop over depths

        std::cout << "Depth " << l << "\n";

        // phase-shifts in the f-x domain.
        t5.start();
        phase_shift_forw_cu<<<nBlocks_ps, nThreads>>>(ns, nf, nx, &d_velmod[l*nx], \
            d_omega, dz, d_base_forw);
        phase_shift_back_cu<<<nBlocks_ps, nThreads>>>(ns, nf, nx, &d_velmod[l*nx], \
            d_omega, dz, d_base_back);
        cudaDeviceSynchronize();
        t5.stop();

        // do FFTs : f-x -> f-kx
        t4.start();
        cufftFORW_Batched1dSignals(d_base_forw, &planBase);
        cufftFORW_Batched1dSignals(d_base_back, &planBase);
        cudaDeviceSynchronize();
        t4.stop();

        // propagate the base wavefields to reference wavefields
        t0.start();
        extrap_ref_wavefields_cu<<<nBlocks, nThreads>>>(ns, nf, nx, nref, \
            d_ref_forw, d_base_forw, &d_opIndices[l*nf*nref], d_ops_forw);
        extrap_ref_wavefields_cu<<<nBlocks, nThreads>>>(ns, nf, nx, nref, \
            d_ref_back, d_base_back, &d_opIndices[l*nf*nref], d_ops_back);
        cudaDeviceSynchronize();
        t0.stop();

        // do IFFTs : f-kx -> f-x
        t4.start();
        cufftBACK_Batched1dSignals(d_ref_forw, ns*nref*nf, nx, &planRef);
        cufftBACK_Batched1dSignals(d_ref_back, ns*nref*nf, nx, &planRef);
        cudaDeviceSynchronize();
        t4.stop();
        
        // interpolation : from ref. wavefields to base wavefields
        t2.start();
        interpolate_cu<<<nBlocks, nThreads>>>(ns, nf, nx, nref, &d_coeff[l*nref*nx], \
            d_ref_forw, d_base_forw);
        interpolate_cu<<<nBlocks, nThreads>>>(ns, nf, nx, nref, &d_coeff[l*nref*nx], \
            d_ref_back, d_base_back);
        cudaDeviceSynchronize();
        t2.stop();
        
        // image depth slide
        t1.start();
        imaging<<<nBlocks_img, nThreads>>>(d_image, d_base_forw, d_base_back, ns, nf, nx, l, imgSize);
        cudaDeviceSynchronize();
        t1.stop();

    } // end loop over depths

    cudaMemcpy(h_image, d_image, ns*nz*nx*sizeof(fcomp), cudaMemcpyDeviceToHost);
    for(int s=0; s<ns; ++s)
        for(int l=0; l<nextrap; ++l)
            for(int i=0; i<nx; ++i)
                image[s*nz*nx + l*nx + i] = reinterpret_cast<float*>(h_image)[2*(s*nz*nx + l*nx + i)];

    cufftDestroy(planBase);
    cufftDestroy(planRef);
    cudaFree(d_base_forw);
    cudaFree(d_base_back);
    cudaFree(d_ref_forw);
    cudaFree(d_ref_back);
    cudaFree(d_image);
    cudaFree(d_ops_forw);
    cudaFree(d_ops_back);
    cudaFree(d_opIndices);
    cudaFree(d_coeff);
    cudaFree(d_velmod);
    cudaFree(d_omega);
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
}


__global__ void phase_shift_forw_cu(int ns, int nf, int nx, float * velSlide, \
    float * omega, float dz, fcomp * wavefield){

    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_f = blockDim.y * blockIdx.y + threadIdx.y;
    int shotSz = nf*nx;

    if(idx_x < nx && idx_f < nf){
        float k = omega[idx_f] / velSlide[idx_x];
        fcomp term = -fcomp(0.0,1.0) * k * dz;
        term = thrust::exp(term);

        for(int s=0; s<ns; ++s)
            wavefield[s*shotSz + idx_f*nx + idx_x] *= term;    
    }
}



__global__ void phase_shift_back_cu(int ns, int nf, int nx, float * velSlide, \
    float * omega, float dz, fcomp * wavefield){

    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_f = blockDim.y * blockIdx.y + threadIdx.y;
    int shotSz = nf*nx;

    if(idx_x < nx && idx_f < nf){
        float k = omega[idx_f] / velSlide[idx_x];
        fcomp term = +fcomp(0.0,1.0) * k * dz;
        term = thrust::exp(term);

        for(int s=0; s<ns; ++s)
            wavefield[s*shotSz + idx_f*nx + idx_x] *= term;    
    }
}



__global__ void extrap_ref_wavefields_cu(int ns, int nf, int nx, int nref, \
    fcomp * ref, fcomp * base, int * opIndices, fcomp * table){

    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_f = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_s = blockDim.z * blockIdx.z + threadIdx.z;
    int baseIdx = idx_s*nf*nx + idx_f*nx + idx_x;
    fcomp basePoint;
    
    int fRef = idx_f*nx;
    int sRef = idx_s*nref*nf*nx;

    if(baseIdx < ns*nf*nx){
        basePoint = base[baseIdx]; //keep value from the base wavefield in registers

        for(int n=0; n<nref; ++n){
            int vRef = n*nf*nx;
            int opIdx = opIndices[idx_f*nref + n] * nx;// find operator's index based on frequency and reference velocity
            ref[sRef + vRef + fRef + idx_x] = table[opIdx + idx_x] * basePoint;
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


