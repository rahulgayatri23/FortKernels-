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
#pragma omp declare target
explicit GPUComplex () {
    re = 0.00;
    im = 0.00;
}


explicit GPUComplex(const double& x, const double& y) {
    re = x;
    im = y;
}

GPUComplex(const GPUComplex& src) {
    re = src.re;
    im = src.im;
}

    GPUComplex& operator =(const GPUComplex& src) {
    re = src.re;
    im = src.im;

    return *this;
}

    GPUComplex& operator =(const double &src) {
    re = src;
    im = 0.00;

    return *this;
}

    GPUComplex& operator +=(const GPUComplex& src) {
    re = src.re + this->re;
    im = src.im + this->im;

    return *this;
}

    GPUComplex& operator -() {
    re = -this->re;
    im = -this->im;

    return *this;
}

void print() const {
    printf("\( %f, %f) ", this->re, this->im);
    printf("\n");
}

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


    friend const GPUComplex GPUComplex_square(GPUComplex& src) ;
    friend const GPUComplex GPUComplex_conj(const GPUComplex& src) ;
    friend const GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
    friend const double GPUComplex_abs(const GPUComplex& src) ;
    friend const GPUComplex  GPUComplex_mult(GPUComplex& a, double b, double c) ;
    friend const GPUComplex GPUComplex_mult(const GPUComplex& a, double b) ;
    friend const void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    friend const void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    friend const GPUComplex doubleMinusGPUComplex(double &a, GPUComplex& src) ;
    friend const GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) ;
    friend double GPUComplex_real( const GPUComplex& src) ;
    friend double GPUComplex_imag( const GPUComplex& src) ;
#pragma omp end declare target
        
};
#pragma omp declare target
    const GPUComplex GPUComplex_square(GPUComplex& src) ;
    const GPUComplex GPUComplex_conj(const GPUComplex& src) ;
    const GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
    const double GPUComplex_abs(const GPUComplex& src) ;
    const GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) ;
    const GPUComplex GPUComplex_mult(const GPUComplex& a, double b) ;
    const void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    const void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;
    const GPUComplex doubleMinusGPUComplex(double &a, GPUComplex& src) ;
    const GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) ;
    double GPUComplex_real( const GPUComplex& src) ;
    double GPUComplex_imag( const GPUComplex& src) ;
#pragma omp end declare target


#endif
