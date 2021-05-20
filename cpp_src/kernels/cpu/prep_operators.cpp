
#include "prep_operators.hpp"

extern "C"

{


// FUNCTION SCOPE: Define N kappa(k) values in an array, in the range [0-kmax].
// INPUT parameters:
//    - kmax : max kappa value in the final array ( k[N-1] = kmax )
//    - N : number of kappa values
// OUTPUT:
//    - k : array with the final values (result)
void create_k(float * k, float kmax, int N){

    float dk = kmax / (N-1);

    for(int i=0; i<N; ++i)
        k[i] = i*dk;
}



// FUNCTION SCOPE:

void read_kx(float * kx, int N, float * read_from){

    for(int i=0; i<N; ++i)
        kx[i] = read_from[i];
}



// FUNCTION SCOPE: Prepare a 2D table of 1D operators for forward-in-time
//    propagation. Operators are designed based on a set of k's given in array
//    k, and given a set of wavenumbers (kx).
// INPUT parameters:
//    - nk : number of k values
//    - k : array with desired k values (must: kappa[0] = 0.0, kappa[nk-1]
// be kmax, no negative values)
//    - nkx : number of kx values (wavenumbers)
//    - kx : array with wavenumbers (represent the positive Fourier 
// coefficients)
//    - dz : depth step (dz > 0)
// OUTPUT:
//    - psOp : "2D" array of ns 1D operators
void forwOperators(fcomp * psOp, int nk, float * k, int nkx, float * kx, float dz){

    fcomp kz;

    for(int j=0; j<nk; ++j){

        for(int i=0; i<nkx; ++i){
        
            if ( std::abs(k[j]) >= std::abs(kx[i]) ){
        
                kz = std::sqrt( std::pow(k[j], 2) - std::pow(kx[i], 2) );
        
            } else {
            
                float term = std::sqrt( std::pow(kx[i],2) - std::pow(k[j], 2) );
                kz = -fcomp(0.0,1.0) * term;
            }
            psOp[j*nkx + i] = pwm::exp( -fcomp(0.0,1.0) * (kz - k[j]) * dz );
        }
    }
}



// FUNCTION SCOPE: Prepare a 2D table of 1D operators for backward-in-time
//    propagation. Operators are designed based on a set of k's given in array
//    k, and given a set of wavenumbers (kx).
// INPUT parameters:
//    - nk : number of k values
//    - k : array with desired k values (must: kappa[0] = 0.0, kappa[nk-1]
// be kmax, no negative values)
//    - nkx : number of kx values (wavenumbers)
//    - kx : array with wavenumbers (represent the positive Fourier 
// coefficients)
//    - dz : depth step (dz > 0)
// OUTPUT:
//    - psOp : "2D" array of ns 1D operators
void backOperators(fcomp * psOp, int nk, float * k, int nkx, float * kx, float dz){

    fcomp kz;

    for(int j=0; j<nk; ++j){

        for(int i=0; i<nkx; ++i){
        
            if ( std::abs(k[j]) >= std::abs(kx[i]) ){
        
                kz = std::sqrt( std::pow(k[j], 2) - std::pow(kx[i], 2) );
        
            } else {
        
                float term = std::sqrt( std::pow(kx[i],2) - std::pow(k[j], 2) );
                kz = +fcomp(0.0,1.0) * term;
            }
            psOp[j*nkx + i] = pwm::exp( +fcomp(0.0,1.0) * (kz - k[j]) * dz );
        }
    }
}



// FUNCTION SCOPE: select based on indexing the most suitable operator for
//    propagation from the table of operators (class f_kx_operators.)
//    assume data in array are equally separated, starting ftom 0.0
int chooseOperatorIndex(int N, float * k, float kref){

    if(N < 2){
        std::cout << "too small vector\n";
        exit(1);
    }
    float deltak = k[1] - k[0];

    if (deltak == 0.0 || deltak < 0){
        std::cout << "The provided kappa array is wrong!\n";
        exit(1);
    }

    if(kref < 0.0){
        std::cout << "kref cannot be negative!\n";
        exit(1);
    }

    if(kref > k[N-1]+deltak){
        std::cout << "kref is larger than the largest k!\n";
        exit(1);
    }

    int idx = 0;
    if (kref == 0)
        return idx;

    idx = (int)(std::floor( kref / deltak ));

    if (idx > N-2){
        idx = N-1; //without this, potential for invalid access in next steps!
        return idx;
    }

    float diffDown = std::abs(kref - k[idx]);
    float diffUp = std::abs(kref - k[idx+1]);

    if (diffUp < diffDown)
        idx += 1; //if closer to upper index choose this.

    return idx;
}



// FUNCTION SCOPE: Is to test the class : f_kx_operators
// This wrapper function exposes the class's constructor in a C-style fashion
// for testing through interfacing with python (ctypes)!
void testClassWrapperFor_f_kx_operators(float * k, float * kx, fcomp * values, \
    int Nk, float kmax, int Nkx, float dz, const char type){
    
    //construct class : here we test the constructor
    f_kx_operators ops(Nk, kmax, Nkx, dz, type, kx);

    //now just copy data from class members to equivalent arrays for testing
    //---------------------------------------
    for(int i=0; i<Nk; ++i)
        k[i] = ops.k[i];
    
    for(int i=0; i<Nkx; ++i)
        kx[i] = ops.kx[i];
    
    for(int i=0; i<Nk; ++i)
        for(int j=0; j<Nkx; ++j)
            values[i*Nkx + j] = ops.values[i*Nkx + j];
    //---------------------------------------
    std::cout << std::endl;
}

} // end extern "C"
