#include "Complex.h" 

GPUComplex GPUComplex_square(GPUComplex& src) {
    double re_this = src.re ;
    double im_this = src.im ;

    GPUComplex result(re_this*re_this - im_this*im_this, 2*re_this*im_this);

    return result;
}

GPUComplex GPUComplex_conj(const GPUComplex& src) {

double re_this = src.re;
double im_this = -1 * src.im;

GPUComplex result(re_this, im_this);
return result;

}

GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) {

    double re_this = a.re * b.im - a.im*b.im ;
    double im_this = a.re * b.im + a.im*b.re ;

    GPUComplex result(re_this, im_this);

    return result;
}


double GPUComplex_abs(const GPUComplex& src) {
    double re_this = src.re * src.re;
    double im_this = src.im * src.im;

    double result = sqrt(re_this+im_this);
    return result;

}

void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.re * c.im - b.im*c.im ;
    double im_this = b.re * c.im + b.im*c.re ;

    GPUComplex mult_result(re_this, im_this);

    a.re += mult_result.re;
    a.im += mult_result.im;
}

void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.re * c.im - b.im*c.im ;
    double im_this = b.re * c.im + b.im*c.re ;

    GPUComplex mult_result(re_this, im_this);

    a.re -= mult_result.re;
    a.im -= mult_result.im;
}

void GPUComplex_mult(GPUComplex& a, double b, double c) {

    a.re = a.re * b * c;
    a.im = a.im * b *c;

}
