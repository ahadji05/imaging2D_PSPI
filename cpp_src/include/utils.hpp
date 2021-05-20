#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include "types.hpp"

extern "C"
{

/* ----------------------- Forward function declarations ----------------------- */
void find_coeff(int nx, float * nxVels, int nref, float * refVels, float * coeff);
void norm_coeff(int nx, int nref, float * coeff);
float find_min_vel(int N, float * refVels);
float find_max_vel(int N, float * refVels);
void define_ref_velocities(int nref, float min_vel, float max_vel, float * refVels);
int * prep_lookUp_indices(int Nf, float * omega, int Nextrap, int Nx, float * velmod, int Nref, int Nk, float * k);
float * prep_interpolation_coeff(float * velmod, int nextrap, int nref, int nx);
/* -------------------------------------------------------------------------------*/



// FUNCTION SCOPE: Calculate coefficients for all ref. velocities for each lateral position based on 'distance'
//      between exact and reference velocity.
// INPUT parameters:
//    - nx : number of lateral positions
//    - nxVels : array with exact velocities
//    - nref: number of reference velocities
//    - refVels: reference velocities
// OUTPUT:
//    - coeff: coefficients (should be of size: nx * nref * sizeof(float))
void find_coeff(int nx, float * nxVels, int nref, float * refVels, float * coeff){
    
    float e = 0.0001; // necessary to avoid division by 0. small though to have negligible contribution

    for(int i=0; i<nx; ++i){

        float exactVel = nxVels[i];
        
        for(int n=0; n<nref; ++n)
            coeff[i*nref + n] = 1.0 / ( std::abs(exactVel - refVels[n]) + e );

    }

}


// FUNCTION SCOPE: normalize coefficients (in-place)
// INPUT parameters:
//    - nx : number of lateral positions
//    - nref: number of reference velocities
// OUTPUT:
//    - coeff: normalized coefficients for all ref. velocities for each lateral position (should be of size nx * nref * sizeof(float))
void norm_coeff(int nx, int nref, float * coeff){

    for(int i=0; i<nx; ++i){
        
        float sum_ = 0.0;

        for(int n=0; n<nref; ++n)
            sum_ += coeff[i*nref + n]; // sum all coefficients for i'th position
        
        float A = 1.0 / sum_; // find normalization factor A

        for(int n=0; n<nref; ++n)
            coeff[i*nref + n] *= A; // normalize coefficients for i'th position
        
    }

}


float find_min_vel(int N, float * refVels){
    
    float min_vel = 1000000.0; // some very large value

    for(int i=0; i<N; ++i){
        if (refVels[i] < min_vel)
            min_vel = refVels[i];
    }

    return min_vel;
}


float find_max_vel(int N, float * refVels){
    
    float max_vel = 0.0; // some very small value

    for(int i=0; i<N; ++i){
        if (refVels[i] > max_vel)
            max_vel = refVels[i];
    }

    return max_vel;
}


void define_ref_velocities(int nref, float min_vel, float max_vel, float * refVels){

    int N = nref-1;
    float dvel = (max_vel - min_vel) / N;

    for(int i=0; i<nref; ++i)
        refVels[i] = min_vel + i*dvel;

}

int * prep_lookUp_indices(int Nf, float * omega, int Nextrap, int Nx, float * velmod, \
    int Nref, int Nk, float * k){

    int tableOfIndicesSize = Nextrap * Nf * Nref;

    int * table = new int[tableOfIndicesSize];
    float * refVels = new float[Nref];

    for(int l=0; l<Nextrap; ++l){

        //define ref. velocities for current depth!
        float minVel = find_min_vel(Nx, &velmod[l*Nx]);
        float maxVel = find_max_vel(Nx, &velmod[l*Nx]);
        define_ref_velocities(Nref, minVel, maxVel, &refVels[0]);
        
        for(int j=0; j<Nf; ++j)
            for(int n=0; n<Nref; ++n){

                float kref = omega[j] / refVels[n];

                table[l*Nf*Nref + j*Nref + n] = chooseOperatorIndex(Nk, &k[0], kref);
            
            }
    }
    
    delete [] refVels;

/*check for Nan values*/
    for(int i=0; i<tableOfIndicesSize; ++i){
        if( std::isnan(table[i])){
            std::cout << "Nan value identified\n";
            exit(1);
        }
    }

/*check for invalid values*/
    for(int i=0; i<tableOfIndicesSize; ++i){
        if( table[i] >= Nk ){
            std::cout << "index cannot be >= Nk\n";
            exit(1);
        }
    }

    return table;
}

float * prep_interpolation_coeff(float * velmod, int nextrap, int nref, int nx){

    float * coeff = new float[nextrap*nref*nx];
    float * refVels = new float[nref];

    for(int l=0; l<nextrap; ++l){
        float vmin = find_min_vel(nx, &velmod[l*nx]);
        float vmax = find_max_vel(nx, &velmod[l*nx]);
        define_ref_velocities(nref, vmin, vmax, refVels);
        find_coeff(nx, &velmod[l*nx], nref, refVels, &coeff[l*nref*nx]);
        norm_coeff(nx, nref, &coeff[l*nref*nx]);
    }

    delete [] refVels;
    return coeff;
}

} // end extern "C"
#endif