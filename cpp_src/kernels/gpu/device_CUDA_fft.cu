
#include "device_CUDA_fft.hpp"

extern "C"
{

cufftHandle make_cuFFTplan_Batched1dSignals(int Nbatch, int NX){

	int dimN[NRANK] = {NX}; // signal's size (number of spatial points)
	
	int inembed[NRANK] = {NX}; // storage is same as dimN - no-padding!
	int onembed[NRANK] = {NX};
	
	int inputStride = 1; // dist. between successive input elements
	int outputStride = inputStride;
	
	int inputDist = NX; // dist. between 1st elem. in successive input signals
	int outputDist = inputDist;
	
// make cuFFT plan
	cufftHandle plan;
	cufftPlanMany(&plan, NRANK, dimN, \
				inembed, inputStride, inputDist, \
				onembed, outputStride, outputDist, \
				CUFFT_C2C, Nbatch);
	
	return plan;
}


void cufftFORW_Batched1dSignals(thrust::complex<float> * d_signals, cufftHandle * plan){
	
	//execute in-place batched FFTs
	cufftExecC2C(*plan, (cufftComplex *)d_signals, (cufftComplex *)d_signals, CUFFT_FORWARD);
}


__global__ void cu_scaleData(fcomp * data, int N, float value)
{
	int pixelIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( pixelIdx_x < N )
		data[pixelIdx_x] *= value;
}


void cufftBACK_Batched1dSignals(fcomp* d_signals, int Nbatch, int NX, cufftHandle * plan){

	//execute in-place batched IFFTs
	cufftExecC2C(*plan, (cufftComplex *)d_signals, (cufftComplex *)d_signals, CUFFT_INVERSE);
	
	cudaDeviceSynchronize();
	
/* for backward FFTs need to scale the data
  -------------------------------------------*/
	size_t SIZE = Nbatch * NX;
	dim3 nThreads(256, 1, 1);
	
	size_t nBlocks_x = SIZE % nThreads.x == 0 ? size_t(SIZE/nThreads.x) : size_t(1 + SIZE/nThreads.x);
	dim3 nBlocks(nBlocks_x, 1, 1);
	
	float scaleValue = 1.0 / (float)(NX);//scale by the signal's length
	cu_scaleData<<<nBlocks, nThreads>>>(d_signals, SIZE, scaleValue);
/*-------------------------------------------*/
}

} // end extern "C"
