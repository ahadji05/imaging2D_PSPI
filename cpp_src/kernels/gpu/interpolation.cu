
#include "interpolation.hpp"

extern "C"

{

__global__ void interpolate_cu(int ns, int nf, int nx, int nref, \
    float * coeff, fcomp * ref, fcomp * base){

    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_f = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_s = blockDim.z * blockIdx.z + threadIdx.z;
    int baseIdx = idx_s*nf*nx + idx_f*nx + idx_x;

    fcomp tmp = fcomp(0.0,0.0);
    
    if(baseIdx < ns*nf*nx){

        int sRef = idx_s*nref*nf*nx;
        
        for(int n=0; n<nref; ++n)
            tmp += coeff[idx_x*nref + n] * \
                ref[sRef + n*nf*nx + idx_f*nx + idx_x];
        
        base[baseIdx] = tmp;
    }
}

} // end extern "C"
