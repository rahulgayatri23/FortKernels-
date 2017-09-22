#ifndef __GPUCOMPLEX
#define __GPUCOMPLEX

#include <iostream>
#include <omp.h>
#include <cmath>
#include "/sw/summitdev/cuda/8.0.54/include/vector_types.h"

class GPUComplex : public float4 {

    private : 
    double re;
    double im;

public:
//#pragma omp declare target
explicit GPUComplex () {
    re = 0.00;
    im = 0.00;
}


//#pragma omp declare target
explicit GPUComplex(const double& x, const double& y) {
    re = x;
    im = y;
}

GPUComplex(const GPUComplex& src) {
    re = src.re;
    im = src.im;
}

//#pragma omp declare target
    GPUComplex& operator =(const GPUComplex& src) {
    re = src.re;
    im = src.im;

    return *this;
}

    GPUComplex& operator +=(const GPUComplex& src) {
    re = src.re + this->re;
    im = src.im + this->im;

    return *this;
}
void print() {
    printf("re,im : %f, %f\n", this->re, this->im);
}

//#pragma omp declare target
    double abs(const GPUComplex& src) {

    double re_this = src.x * src.re;
    double im_this = src.y * src.im;

    double result = sqrt(re_this+im_this);
    return result;

    }

    double get_real() const
    {
        return this->re;
    }

    double get_imag() const
    {
        return this->im;
    }

    void set_real(double val)
    {
        this->re = val;
    }

    void set_imag(double val) 
    {
        this->im = val;
    }

    friend GPUComplex GPUComplex_square(GPUComplex& src) ;
    friend GPUComplex GPUComplex_conj(const GPUComplex& src) ;
    friend GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
    friend double GPUComplex_abs(const GPUComplex& src) ;
    friend void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    friend void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    friend void GPUComplex_mult(GPUComplex& a, double b, double c) ;
        
};
    GPUComplex GPUComplex_square(GPUComplex& src) ;
    GPUComplex GPUComplex_conj(const GPUComplex& src) ;
    GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
    double GPUComplex_abs(const GPUComplex& src) ;
    void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    void GPUComplex_mult(GPUComplex& a, double b, double c) ;


#endif
