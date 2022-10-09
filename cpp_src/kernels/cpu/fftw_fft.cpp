
#include "fftw_fft.hpp"
#include <cassert>

extern "C"

{

void batched1dffts(fcomp * data, int N1, int N2, int sign){
    
    int dim[] = {N2};
    int rank=1;
    int nBatch=N1;
    int stride=1;
    int distance=N2;

    bool invalid_sign = false;
    if(sign != 1 && sign != -1)
        assert(invalid_sign);

    fftwf_plan p = fftwf_plan_many_dft(rank, 
		             dim,
			     nBatch,
                             (fftwf_complex*)data, 
			     dim,
                             stride, 
			     distance,
                             (fftwf_complex*)data, 
			     dim,
                             stride, 
			     distance,
                             sign, 
			     FFTW_ESTIMATE);

    fftwf_execute(p);

    // Scale the data for inverse FFT based on signal lenght(N2)
    if(sign == 1){
        float scale_value = 1.0 / N2;
        for( int i = 0; i < N1*N2; ++i)
	    data[i] *= scale_value;
    }

    fftwf_destroy_plan(p);
}

} //end extern C
