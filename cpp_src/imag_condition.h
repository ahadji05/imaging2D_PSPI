
#include "types.h"
#include <iostream>

extern "C"

{

void cross_corr(int ns, int shotSize, int nx, int nf, fcomp * forw, fcomp * back, int sizeImage, int depthIdx, float * image);

} // end extern "C"