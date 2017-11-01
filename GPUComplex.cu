#include "GPUComplex.h"

#if CudaKernel
/*
 * Return the square of a complex number 
 */
__device__ const GPUComplex d_GPUComplex_square(GPUComplex& src) {
    double re_this = src.re ;
    double im_this = src.im ;

    GPUComplex result(re_this*re_this - im_this*im_this, 2*re_this*im_this);

    return result;
}

/*
 * Return the conjugate of a complex number 
 */
__device__ const GPUComplex d_GPUComplex_conj(const GPUComplex& src) {

double re_this = src.re;
double im_this = -1 * src.im;

GPUComplex result(re_this, im_this);
return result;

}


/*
 * Return the product of 2 complex numbers 
 */
__device__ const GPUComplex d_GPUComplex_product(const GPUComplex& a, const GPUComplex& b) {

    double re_this = a.re * b.re - a.im*b.im ;
    double im_this = a.re * b.im + a.im*b.re ;

    GPUComplex result(re_this, im_this);
    return result;
}

/*
 * Return the absolute of a complex number 
 */
__device__ double d_GPUComplex_abs(const GPUComplex& src) {
    double re_this = src.re * src.re;
    double im_this = src.im * src.im;

 //   double result = (re_this+im_this);
    double result = sqrt(re_this+im_this);
    return result;
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
__device__ const GPUComplex d_GPUComplex_mult(GPUComplex& a, double b, double c) {

    GPUComplex result(a.re * b * c, a.im * b * c);
    return result;

}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
__device__ const GPUComplex d_GPUComplex_mult(const GPUComplex& a, double b) {

   GPUComplex result(a.re*b, a.im*b);
   return result;

}

/*
 * Return the complex number a += b * c  
 */
__device__ void d_GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.re * c.re - b.im*c.im ;
    double im_this = b.re * c.im + b.im*c.re ;

    GPUComplex mult_result(re_this, im_this);

    a.re += mult_result.re;
    a.im += mult_result.im;
}

/*
 * Return the complex number a -= b * c  
 */
__device__ void d_GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    double re_this = b.re * c.re - b.im*c.im ;
    double im_this = b.re * c.im + b.im*c.re ;

    GPUComplex mult_result(re_this, im_this);

    a.re -= mult_result.re;
    a.im -= mult_result.im;
}


__device__ GPUComplex d_doubleMinusGPUComplex(const double &a, GPUComplex& src) {
    GPUComplex result(a - src.re, 0 - src.im);
    return result;
}

__device__ const GPUComplex d_doublePlusGPUComplex(double a, GPUComplex& src) {
    GPUComplex result(a + src.re, 0 + src.im);
    return result;
}

__device__ double d_GPUComplex_real( const GPUComplex& src) {
    return src.re;
}

__device__ double d_GPUComplex_imag( const GPUComplex& src) {
    return src.im;
}

__device__ void d_GPUComplex_plusEquals( GPUComplex& a, const GPUComplex & b) {
    a.re = a.re + b.re;
    a.im = a.im + b.im;
}

__device__ void d_GPUComplex_Equals( GPUComplex& a, const GPUComplex & b) {
    a.re = b.re;
    a.im = b.im;
}

__device__ void d_print( const GPUComplex& a) {
    printf("( %f, %f) ", a.re, a.im);
    printf("\n");
}

__global__  void cudaBGWKernel( GPUComplex *wtilde_array, GPUComplex *aqsntemp, GPUComplex *I_eps_array, int ncouls, double wxt, double& achtemp_re_iw, double& achtemp_im_iw, int my_igp, GPUComplex mygpvar1, int n1, double vcoul_igp)
{
    int ig = threadIdx.x;

    GPUComplex sch_array_iw(0.00, 0.00);
    GPUComplex expr(0.50, 0.50);

    if(ig < ncouls)
    {
        GPUComplex wdiff = d_doubleMinusGPUComplex(wxt , wtilde_array[my_igp*ncouls+ig]);
        double rden = d_GPUComplex_real(d_GPUComplex_product(wdiff, d_GPUComplex_conj(wdiff)));
        rden = 1/rden;

        d_GPUComplex_Equals(sch_array_iw, d_GPUComplex_mult(d_GPUComplex_product(d_GPUComplex_product(mygpvar1 , aqsntemp[n1*ncouls+ig]), d_GPUComplex_product(d_GPUComplex_mult(d_GPUComplex_product(wtilde_array[my_igp*ncouls+ig] , d_GPUComplex_conj(wdiff)), rden), I_eps_array[my_igp*ncouls+ig])), 0.5));

        atomicAdd(&achtemp_re_iw , d_GPUComplex_real( d_GPUComplex_mult(sch_array_iw , vcoul_igp)));
        atomicAdd(&achtemp_im_iw , d_GPUComplex_imag( d_GPUComplex_mult(sch_array_iw , vcoul_igp)));

//        achtemp_re_iw = d_GPUComplex_real( d_GPUComplex_mult(sch_array_iw , vcoul_igp));
//        achtemp_im_iw = d_GPUComplex_imag( d_GPUComplex_mult(sch_array_iw , vcoul_igp));
    }
}

void gppKernelGPU( GPUComplex *wtilde_array, GPUComplex *aqsntemp, GPUComplex *I_eps_array, int ncouls, double wxt, double &achtemp_re_iw, double &achtemp_im_iw, int my_igp, GPUComplex mygpvar1, int n1, double vcoul_igp)
{
    cudaBGWKernel <<< 1, ncouls>>> ( wtilde_array, aqsntemp, I_eps_array, ncouls, wxt, achtemp_re_iw, achtemp_im_iw, my_igp, mygpvar1, n1, vcoul_igp);
}

#endif
