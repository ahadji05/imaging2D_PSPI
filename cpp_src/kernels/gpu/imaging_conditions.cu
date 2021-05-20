
#include "imaging_conditions.hpp"

extern "C"

{

__global__ void imaging_cu(fcomp * image, fcomp * forw_pulse, fcomp * back_pulse, \
    int ns, int nf, int nx, int depthIdx, int imgSize){

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_s = blockIdx.y * blockDim.y + threadIdx.y;

    fcomp conv = fcomp(0.0,0.0);

    for(int j=0; j<nf; j++){
        int idx = idx_s*nf*nx + j * nx + idx_x;
        conv += forw_pulse[idx] * thrust::conj(back_pulse[idx]);
    }

    image[idx_s*imgSize + depthIdx*nx + idx_x] = conv;
}

} // end extern "C"
