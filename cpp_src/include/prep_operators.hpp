#ifndef PREP_OPERATORS_HPP
#define PREP_OPERATORS_HPP


#include <iostream>
#include <vector>
#include "types.hpp"

extern "C"

{

// Functions declarations : these functions are used inside the class: f_kx_operators.
// The reason for NOT implementing them as member functions is for testing 
// purposes, since the methodology relies on interfacing with python (ctypes)! 
void create_k(float * , float , int );
void read_kx(float * , int , float * );
void forwOperators(fcomp * , int , float * , int , float * , float );
void backOperators(fcomp * , int , float * , int , float * , float );
int chooseOperatorIndex(int , float * , float );

// CLASS SCOPE: prepare a 2D array of propagation operators based on a range of
// kappa's (k=omega/velocity) and wavenumbers (kx=1/x).
class f_kx_operators {

    public:
    
        int Nk;
        float kmax;
        int Nkx;
        float dz;
        const char type; // should be f (for forward) or b (for backward),
        // to specify the direction of propagation

        std::vector<float> k;
        std::vector<float> kx;
        std::vector<fcomp> values;

        // constructor
        f_kx_operators(int Nk, float kmax, int Nkx, \
            float dz, const char type, float * kx_from)
                : Nk(Nk), kmax(kmax), Nkx(Nkx), dz(dz), type(type) {
            
            k.reserve( Nk );
            create_k(k.data(), kmax, Nk);

            kx.reserve( Nkx );
            read_kx(kx.data(), Nkx, &kx_from[0]);

            values.reserve( Nk * Nkx );
            if (type == 'f'){
                forwOperators(values.data(), Nk, k.data(), Nkx, kx.data(), dz);
            } else 
            if (type == 'b') {
                backOperators(values.data(), Nk, k.data(), Nkx, kx.data(), dz);
            } else {
                
                //set all values to zero (needed for testing purposes)
                for(int i=0; i<Nk; ++i)
                    for(int j=0; j<Nkx; ++j)
                        values[i*Nkx + j] = fcomp(0.0,0.0);

                std::cout << "Wrong operator type!\n";
            }
        }
};

} // end extern "C"

#endif