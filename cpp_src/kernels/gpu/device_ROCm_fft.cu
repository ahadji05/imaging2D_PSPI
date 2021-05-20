
#include "device_ROCm_fft.hpp"

extern "C"
{

hipfftHandle make_hipFFTplan_Batched1dSignals(int Nbatch, int NX){

	int dimN[NRANK] = {NX}; // signal's size (number of spatial points)
	
	int inembed[NRANK] = {NX}; // storage is same as dimN - no-padding!
	int onembed[NRANK] = {NX};
	
	int inputStride = 1; // dist. between successive input elements
	int outputStride = inputStride;
	
	int inputDist = NX; // dist. between 1st elem. in successive input signals
	int outputDist = inputDist;
	
	// make hipFFT plan
	hipfftHandle plan;
	hipfftPlanMany(&plan, NRANK, dimN, \
				inembed, inputStride, inputDist, \
				onembed, outputStride, outputDist, \
				HIPFFT_C2C, Nbatch);
	
	return plan;
}


void hipfftFORW_Batched1dSignals(fcomp * d_signals, hipfftHandle * plan){
	
	//execute in-place batched FFTs
	hipfftExecC2C(*plan, (hipfftComplex *)d_signals, (hipfftComplex *)d_signals, HIPFFT_FORWARD);
}


__global__ void hip_scaleData(fcomp * data, int N, float value)
{
	int pixelIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( pixelIdx_x < N )
		data[pixelIdx_x] *= value;
}


void hipfftBACK_Batched1dSignals(fcomp * d_signals, int Nbatch, int NX, hipfftHandle * plan){

    //execute in-place batched IFFTs
    hipfftExecC2C(*plan, (hipfftComplex *)d_signals, (hipfftComplex *)d_signals, HIPFFT_BACKWARD);
	
    hipDeviceSynchronize();
	
/* for backward FFTs need to scale the data
  -------------------------------------------*/
    size_t SIZE = Nbatch * NX;
	dim3 nThreads(256, 1, 1);
	
    size_t nBlocks_x = SIZE % nThreads.x == 0 ? size_t(SIZE/nThreads.x) : size_t(1 + SIZE/nThreads.x);
    dim3 nBlocks(nBlocks_x, 1, 1);
    
    float scaleValue = 1.0 / (float)(NX);//scale by the signal's length
    hipLaunchKernelGGL( hip_scaleData, nBlocks, nThreads, 0, 0, d_signals, SIZE, scaleValue );
/*-------------------------------------------*/
}

} // end extern "C"
