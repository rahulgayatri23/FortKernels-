#ifndef __GPUCOMPLEX
#define __GPUCOMPLEX

#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
#include <chrono>

#include <vector_types.h>
//#include "/sw/summitdev/cuda/8.0.54/include/vector_types.h"

class GPUComplex : public double2{

    private : 

public:
    double re;
    double im;

explicit GPUComplex () {
    re = 0.00;
    im = 0.00;
}


KOKKOS_INLINE_FUNCTION
explicit GPUComplex(const double& x, const double& y) {
    re = x;
    im = y;
}

KOKKOS_INLINE_FUNCTION
GPUComplex(const GPUComplex& src) {
    re = src.re;
    im = src.im;
}

KOKKOS_INLINE_FUNCTION
GPUComplex& operator =(const GPUComplex& src) {
    re = src.re;
    im = src.im;

    return *this;
}

KOKKOS_INLINE_FUNCTION
GPUComplex& operator +=(const GPUComplex& src) {
    re = src.re + this->re;
    im = src.im + this->im;

    return *this;
}

KOKKOS_INLINE_FUNCTION
GPUComplex& operator -=(const GPUComplex& src) {
    re = src.re - this->re;
    im = src.im - this->im;

    return *this;
}

KOKKOS_INLINE_FUNCTION
GPUComplex& operator -() {
    re = -this->re;
    im = -this->im;

    return *this;
}

KOKKOS_INLINE_FUNCTION
GPUComplex& operator ~() {
    return *this;
}

KOKKOS_INLINE_FUNCTION
void print() const {
    printf("( %f, %f) ", this->re, this->im);
    printf("\n");
}

KOKKOS_INLINE_FUNCTION
double abs(const GPUComplex& src) {

    double re_this = src.re * src.re;
    double im_this = src.im * src.im;

    double result = (re_this+im_this);
    return result;
}

KOKKOS_INLINE_FUNCTION
double get_real() const
{
    return this->re;
}

KOKKOS_INLINE_FUNCTION
double get_imag() const
{
    return this->im;
}

KOKKOS_INLINE_FUNCTION
void set_real(double val)
{
    this->re = val;
}

KOKKOS_INLINE_FUNCTION
void set_imag(double val) 
{
    this->im = val;
}
};
    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_square(GPUComplex& src) ;

    KOKKOS_INLINE_FUNCTION
    void GPUComplex_equals(GPUComplex& a, GPUComplex& b) ;

    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_minus(const GPUComplex& a, const GPUComplex& b) ;

    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_plus(const GPUComplex& a, const GPUComplex& b) ;

    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_conj(const GPUComplex& src) ;

    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;

    KOKKOS_INLINE_FUNCTION
    double GPUComplex_abs(const GPUComplex& src) ;

    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) ;

    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_divide(const GPUComplex& a, const GPUComplex& b) ;

    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_divide(const GPUComplex& a, double b) ;

    KOKKOS_INLINE_FUNCTION
    GPUComplex GPUComplex_mult(const GPUComplex& a, double b) ;

    KOKKOS_INLINE_FUNCTION
    void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;

    KOKKOS_INLINE_FUNCTION
    void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) ;

    KOKKOS_INLINE_FUNCTION
    GPUComplex doubleMinusGPUComplex(const double &a, GPUComplex& src) ;

    KOKKOS_INLINE_FUNCTION
    GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) ;

    KOKKOS_INLINE_FUNCTION
    double GPUComplex_real( const GPUComplex& src) ;

    KOKKOS_INLINE_FUNCTION
    double GPUComplex_imag( const GPUComplex& src) ;

/*
 * Return the square of a complex number 
 */
GPUComplex GPUComplex_square(GPUComplex& src) {
    double re_this = src.re ;
    double im_this = src.im ;

    GPUComplex result(re_this*re_this - im_this*im_this, 2*re_this*im_this);

    return result;
}

/*
 * GPUComplex a = b
 */
void GPUComplex_equals(GPUComplex& a, GPUComplex& b) {
    a.re = b.re;
    a.im = b.im;
}

GPUComplex GPUComplex_minus(const GPUComplex& a, const GPUComplex& b) {

    GPUComplex result(a.re - b.re, a.im - b.im);
    return result;
}

GPUComplex GPUComplex_plus(const GPUComplex& a, const GPUComplex& b) {

    GPUComplex result(a.re + b.re, a.im + b.im);
    return result;
}

/*
 * Return the conjugate of a complex number 
 */
GPUComplex GPUComplex_conj(const GPUComplex& src) {

double re_this = src.re;
double im_this = -1 * src.im;

GPUComplex result(re_this, im_this);
return result;

}

/*
 * Return the product of 2 complex numbers 
 */
GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) {

    double re_this = a.re * b.re - a.im*b.im ;
    double im_this = a.re * b.im + a.im*b.re ;

    GPUComplex result(re_this, im_this);
    return result;
}

/*
 * Return the absolute of a complex number 
 */
double GPUComplex_abs(const GPUComplex& src) {
    double re_this = src.re * src.re;
    double im_this = src.im * src.im;

    double result = (re_this+im_this);
//    double result = sqrt(re_this+im_this);
    return result;
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) {

    GPUComplex result(a.re * b * c, a.im * b * c);
    return result;

}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
GPUComplex GPUComplex_mult(const GPUComplex& a, double b) {

   GPUComplex result(a.re*b, a.im*b);
   return result;

}

/*
 * Return the division of 2 complex numbers 
 */
GPUComplex GPUComplex_divide(const GPUComplex& a, const GPUComplex& b) {

    GPUComplex b_conj = GPUComplex_conj(b);
    GPUComplex numerator = GPUComplex_product(a, b_conj);
    GPUComplex denominator = GPUComplex_product(b, b_conj);

    double re_this = numerator.re / denominator.re;
    double im_this = numerator.im / denominator.re;

    GPUComplex result(re_this, im_this);
    return result;
}

/*
 * Return the complex number c = a / b (a is complex, b is double) 
 */
GPUComplex GPUComplex_divide(const GPUComplex& a, double b) {

   GPUComplex result(a.re/b, a.im/b);
   return result;

}


/*
 * Return the complex number a += b * c  
 */
void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.re * c.re - b.im*c.im ;
    double im_this = b.re * c.im + b.im*c.re ;

    GPUComplex mult_result(re_this, im_this);

    a.re += mult_result.re;
    a.im += mult_result.im;
}

/*
 * Return the complex number a -= b * c  
 */
void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.re * c.re - b.im*c.im ;
    double im_this = b.re * c.im + b.im*c.re ;

    GPUComplex mult_result(re_this, im_this);

    a.re -= mult_result.re;
    a.im -= mult_result.im;
}


GPUComplex doubleMinusGPUComplex(const double &a, GPUComplex& src) {
    GPUComplex result(a - src.re, 0 - src.im);
    return result;
}

GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) {
    GPUComplex result(a + src.re, 0 + src.im);
    return result;
}

double GPUComplex_real( const GPUComplex& src) {
    return src.re;
}

double GPUComplex_imag( const GPUComplex& src) {
    return src.im;
}

#endif
