#include "Complex.h" 


/*
 * Return the square of a complex number 
 */
const GPUComplex GPUComplex_square(GPUComplex& src) {
    double re_this = src.re ;
    double im_this = src.im ;

    GPUComplex result(re_this*re_this - im_this*im_this, 2*re_this*im_this);

    return result;
}

/*
 * Return the conjugate of a complex number 
 */
const GPUComplex GPUComplex_conj(const GPUComplex& src) {

double re_this = src.re;
double im_this = -1 * src.im;

GPUComplex result(re_this, im_this);
return result;

}


/*
 * Return the product of 2 complex numbers 
 */
const GPUComplex GPUComplex_product(const GPUComplex& a, const GPUComplex& b) {

    double re_this = a.re * b.re - a.im*b.im ;
    double im_this = a.re * b.im + a.im*b.re ;

    GPUComplex result(re_this, im_this);
    return result;
}


/*
 * Return the absolute of a complex number 
 */
const double GPUComplex_abs(const GPUComplex& src) {
    double re_this = src.re * src.re;
    double im_this = src.im * src.im;

    double result = sqrt(re_this+im_this);
    return result;

}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
const GPUComplex GPUComplex_mult(GPUComplex& a, double b, double c) {

    GPUComplex result(a.re * b * c, a.im * b * c);
    return result;

}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
const GPUComplex GPUComplex_mult(const GPUComplex& a, double b) {

   GPUComplex result(a.re*b, a.im*b);
   return result;

}



/*
 * Return the complex number a += b * c  
 */
const void GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.re * c.im - b.im*c.im ;
    double im_this = b.re * c.im + b.im*c.re ;

    GPUComplex mult_result(re_this, im_this);

    a.re += mult_result.re;
    a.im += mult_result.im;
}


/*
 * Return the complex number a -= b * c  
 */
const void GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.re * c.im - b.im*c.im ;
    double im_this = b.re * c.im + b.im*c.re ;

    GPUComplex mult_result(re_this, im_this);

    a.re -= mult_result.re;
    a.im -= mult_result.im;
}


const GPUComplex doubleMinusGPUComplex(double &a, GPUComplex& src) {
    GPUComplex result(a - src.re, 0 - src.im);
    return result;
}

const GPUComplex doublePlusGPUComplex(double a, GPUComplex& src) {
    GPUComplex result(a + src.re, 0 + src.im);
    return result;
}

double GPUComplex_real( const GPUComplex& src) {
    return src.re;
}

double GPUComplex_imag( const GPUComplex& src) {
    return src.im;
}
