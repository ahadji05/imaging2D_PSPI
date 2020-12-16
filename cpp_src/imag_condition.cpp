
#include "imag_condition.h"

extern "C"

{

void cross_corr(int ns, int shotSize, int nx, int nf, fcomp * forw, fcomp * back, int sizeImage, int depthIdx, float * image){

    fcomp * conv = new fcomp[ns*nx];

    for (int is=0; is<ns; ++is)
        for (int i=0; i<nx; i++)
            conv[is*nx + i] = fcomp(0.0,0.0);
    
    for (int is=0; is<ns; ++is)
        for (int j=0; j<nf; j++)
            for (int i=0; i<nx; i++)
                conv[is*nx + i] += forw[is*shotSize + j*nx + i] * std::conj(back[is*shotSize + j*nx + i]);

    for (int is=0; is<ns; ++is)
        for (int i=0; i<nx; i++){
            image[is*sizeImage + depthIdx*nx + i] = reinterpret_cast<float*>(&conv[is*nx])[2*i];
        }

    delete [] conv;

}

} // end extern "C"
