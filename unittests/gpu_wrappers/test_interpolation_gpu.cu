
#include "interpolation.hpp"

extern "C"
{

void test_interpolate_cu(int ns, int nf, int nx, int nref, \
    float * coeff, fcomp * ref, fcomp * base){

#ifdef PWM_ENABLE_HIP

    float * d_coeff;
    fcomp * d_ref, * d_base;
    hipMalloc( &d_coeff,   nref*nx*sizeof(float) );
    hipMalloc( &d_ref,     ns*nref*nf*nx*sizeof(fcomp) );
    hipMalloc( &d_base,    ns*nf*nx*sizeof(fcomp) );

    hipMemcpy( d_coeff, coeff, nref*nx*sizeof(float),       hipMemcpyHostToDevice );
    hipMemcpy( d_ref,   ref,   ns*nref*nf*nx*sizeof(fcomp), hipMemcpyHostToDevice );
    
    dim3 nThreads(64, 1, 1);
    size_t nBlocks_x = nx % nThreads.x == 0 ? size_t(nx/nThreads.x) : size_t(1 + nx/nThreads.x);
    dim3 nBlocks( nBlocks_x, nf, ns );

    hipLaunchKernelGGL(interpolate_cu, nBlocks, nThreads , 0, 0, \
        ns, nf, nx, nref, d_coeff, d_ref, d_base);

    hipMemcpy( base, d_base,   ns*nf*nx*sizeof(fcomp),      hipMemcpyDeviceToHost );

    hipFree( d_base );
    hipFree( d_ref );
    hipFree( d_coeff );

#else

    float * d_coeff;
    fcomp * d_ref, * d_base;
    cudaMalloc( &d_coeff,   nref*nx*sizeof(float) );
    cudaMalloc( &d_ref,     ns*nref*nf*nx*sizeof(fcomp) );
    cudaMalloc( &d_base,    ns*nf*nx*sizeof(fcomp) );

    cudaMemcpy( d_coeff, coeff, nref*nx*sizeof(float),       cudaMemcpyHostToDevice );
    cudaMemcpy( d_ref,   ref,   ns*nref*nf*nx*sizeof(fcomp), cudaMemcpyHostToDevice );
    
    dim3 nThreads(64, 1, 1);
    size_t nBlocks_x = nx % nThreads.x == 0 ? size_t(nx/nThreads.x) : size_t(1 + nx/nThreads.x);
    dim3 nBlocks( nBlocks_x, nf, ns );

    interpolate_cu<<<nBlocks, nThreads>>> \
        ( ns, nf, nx, nref, d_coeff, d_ref, d_base );

    cudaMemcpy( base, d_base,   ns*nf*nx*sizeof(fcomp),      cudaMemcpyDeviceToHost );

    cudaFree( d_base );
    cudaFree( d_ref );
    cudaFree( d_coeff );

#endif

    }

} // end extern "C"