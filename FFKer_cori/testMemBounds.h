#include <iostream>
#include <omp.h>

#define CUDAVER 0

using namespace std;

#define N 10000 
#define M 30000 

class GPUComplex {

    private : 
    double x;
    double y;

public:
#pragma omp declare target
explicit GPUComplex () {
    x = 0.00;
    y = 0.00;
}

explicit GPUComplex(const double& a, const double& b) {
    x = a;
    y = b;
}

GPUComplex& operator =(const GPUComplex& src) {
    x = src.x;
    y = src.y;

    return *this;
}

GPUComplex& operator +=(const GPUComplex& src) {
    x = src.x + this->x;
    y = src.y + this->y;

    return *this;
}

GPUComplex& operator ~() {
    return *this;
}

void print() const {
    printf("( %f, %f) ", this->x, this->y);
    printf("\n");
}

double get_real() const
{
    return this->x;
}

double get_imag() const
{
    return this->y;
}

friend GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
friend const inline GPUComplex d_GPUComplex_product(const GPUComplex& a, const GPUComplex& b) ;
friend inline double d_GPUComplex_real( const GPUComplex& src) ;
friend inline double d_GPUComplex_imag( const GPUComplex& src) ;
friend void inline d_GPUComplex_Equals( GPUComplex& a, const GPUComplex & b) ;

#pragma omp end declare target
};


/*
 * Return the product of 2 complex numbers 
 */
#if !CUDAVER
GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) 
{

    double re_this = a.x * b.x - a.y*b.y ;
    double im_this = a.x * b.y + a.y*b.x ;

    GPUComplex result(re_this, im_this);
    return result;
}
#endif

void testMemBounds_cuKernel(GPUComplex &achsDtemp, GPUComplex *aqsmtemp, GPUComplex *aqsntemp);
