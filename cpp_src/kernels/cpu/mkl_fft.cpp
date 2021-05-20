
#include "mkl_fft.hpp"

extern "C"

{

// FUNCTION SCOPE: perform 1d forward FFT
// INPUT parameters:
//    - data : "1D" array of data to be transformed in-place
//    - N : number of points
void c64fft1dforw(fcomp * data, int N){
    
    MKL_LONG length = N;

    DFTI_DESCRIPTOR_HANDLE dfti_handle;

    DftiCreateDescriptor(&dfti_handle, DFTI_SINGLE, DFTI_COMPLEX, 1, length);

    DftiCommitDescriptor(dfti_handle);

    DftiComputeForward(dfti_handle, data);

    DftiFreeDescriptor(&dfti_handle);
}



// FUNCTION SCOPE: perform 1d backward FFT
// INPUT parameters:
//    - data : "1D" array of data to be transformed in-place
//    - N : number of points
void c64fft1dback(fcomp * data, int N){

    MKL_LONG length = N;

    DFTI_DESCRIPTOR_HANDLE dfti_handle;

    DftiCreateDescriptor(&dfti_handle, DFTI_SINGLE, DFTI_COMPLEX, 1, length);

    DftiSetValue(dfti_handle, DFTI_BACKWARD_SCALE, 1.0/(float)(N));
    
    DftiCommitDescriptor(dfti_handle);
    
    DftiComputeBackward(dfti_handle, data);
    
    DftiFreeDescriptor(&dfti_handle);
}



// FUNCTION SCOPE: perform multiple 1D forward FFTs using data stored in a "2D" array
// INPUT parameters:
//    - data : "2D" array of data to be transformed in-place
//    - N1 : number of rows
//    - N2 : number of columns
//    - axis : choice of axis of transformations (axis=0 -> N1 transforms, axis=1 -> N2 transforms)
void fft1dforwardFrom2Darray(fcomp * data, int N1, int N2, int axis){
    
    DFTI_DESCRIPTOR_HANDLE dfti_handle;

    if(axis == 1){
        DftiCreateDescriptor(&dfti_handle, DFTI_SINGLE, DFTI_COMPLEX, 1, N2);
        
        DftiSetValue(dfti_handle, DFTI_NUMBER_OF_TRANSFORMS, N1);
        DftiSetValue(dfti_handle, DFTI_INPUT_DISTANCE, N2);
        DftiSetValue(dfti_handle, DFTI_OUTPUT_DISTANCE, N2);
        
        DftiCommitDescriptor(dfti_handle);

        DftiComputeForward(dfti_handle, data);
    }

    MKL_LONG strides[2];
    if(axis == 0){
        strides[0] = 0; strides[1] = N2;
        DftiCreateDescriptor(&dfti_handle, DFTI_SINGLE, DFTI_COMPLEX, 1, N1);

        DftiSetValue(dfti_handle, DFTI_NUMBER_OF_TRANSFORMS, N2);
        DftiSetValue(dfti_handle, DFTI_INPUT_DISTANCE, 1);
        DftiSetValue(dfti_handle, DFTI_OUTPUT_DISTANCE, 1);
        DftiSetValue(dfti_handle, DFTI_INPUT_STRIDES, strides);
        DftiSetValue(dfti_handle, DFTI_OUTPUT_STRIDES, strides);

        DftiCommitDescriptor(dfti_handle);

        DftiComputeForward(dfti_handle, data);
    }
    
    DftiFreeDescriptor(&dfti_handle);
}



// FUNCTION SCOPE: perform multiple 1D backward FFTs using data stored in a "2D" array
// INPUT parameters:
//    - data : "2D" array of data to be transformed in-place
//    - N1 : number of rows
//    - N2 : number of columns
//    - axis : choice of axis of transformations (axis=0 -> N1 transforms, axis=1 -> N2 transforms)
void fft1dbackwardFrom2Darray(fcomp * data, int N1, int N2, int axis){
    
    DFTI_DESCRIPTOR_HANDLE dfti_handle;

    if(axis == 1){
        DftiCreateDescriptor(&dfti_handle, DFTI_SINGLE, DFTI_COMPLEX, 1, N2);
        
        DftiSetValue(dfti_handle, DFTI_NUMBER_OF_TRANSFORMS, N1);
        DftiSetValue(dfti_handle, DFTI_INPUT_DISTANCE, N2);
        DftiSetValue(dfti_handle, DFTI_OUTPUT_DISTANCE, N2);
        DftiSetValue(dfti_handle, DFTI_BACKWARD_SCALE, 1.0/(float)(N2));
        
        DftiCommitDescriptor(dfti_handle);

        DftiComputeBackward(dfti_handle, data);
    }

    MKL_LONG strides[2];
    if(axis == 0){
        strides[0] = 0; strides[1] = N2;
        DftiCreateDescriptor(&dfti_handle, DFTI_SINGLE, DFTI_COMPLEX, 1, N1);

        DftiSetValue(dfti_handle, DFTI_NUMBER_OF_TRANSFORMS, N2);
        DftiSetValue(dfti_handle, DFTI_INPUT_DISTANCE, 1);
        DftiSetValue(dfti_handle, DFTI_OUTPUT_DISTANCE, 1);
        DftiSetValue(dfti_handle, DFTI_INPUT_STRIDES, strides);
        DftiSetValue(dfti_handle, DFTI_OUTPUT_STRIDES, strides);
        DftiSetValue(dfti_handle, DFTI_BACKWARD_SCALE, 1.0/(float)(N1));

        DftiCommitDescriptor(dfti_handle);

        DftiComputeBackward(dfti_handle, data);
    }
    
    DftiFreeDescriptor(&dfti_handle);
}

} //end extern C
