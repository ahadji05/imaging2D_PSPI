
#include "phase_shifts.hpp"

extern "C"
{
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


} // end extern "C"
