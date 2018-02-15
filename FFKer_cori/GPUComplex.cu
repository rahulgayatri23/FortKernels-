#include "GPUComplex.h"

/*
 * Return the square of a complex number 
 */
__device__ const inline GPUComplex d_GPUComplex_square(GPUComplex& src) {
    return GPUComplex(src.re*src.re - src.im*src.im, 2*src.re*src.im);
}

/*
 * Return the conjugate of a complex number 
 */
__device__ const inline GPUComplex d_GPUComplex_conj(const GPUComplex& src) {
return GPUComplex(src.re, -src.im);
}


/*
 * Return the product of 2 complex numbers 
 */
__device__ const inline GPUComplex d_GPUComplex_product(const GPUComplex& a, const GPUComplex& b) {
    return GPUComplex(a.re * b.re - a.im*b.im, a.re * b.im + a.im*b.re);
}


/*
 * Return the absolute of a complex number 
 */
__device__ inline double d_GPUComplex_abs(const GPUComplex& src) {
    return sqrt(src.re * src.re + src.im * src.im);
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
__device__ const inline GPUComplex d_GPUComplex_mult(GPUComplex& a, double b, double c) {
    return GPUComplex(a.re * b * c, a.im * b * c);
}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
__device__ const inline GPUComplex d_GPUComplex_mult(const GPUComplex& a, double b) {
   return GPUComplex(a.re*b, a.im*b);

}

/*
 * Return the complex number a += b * c  
 */
__device__ inline void d_GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    a.re += b.re * c.re - b.im*c.im ;
    a.im += b.re * c.im + b.im*c.re ;
}

/*
 * Return the complex number a -= b * c  
 */
__device__ inline void d_GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    a.re -= b.re * c.re - b.im*c.im ;
    a.im -= b.re * c.im + b.im*c.re ;
}


__device__ inline GPUComplex d_doubleMinusGPUComplex(const double &a, GPUComplex& src) {
    return GPUComplex(a-src.re, -src.im);
}

__device__ inline GPUComplex d_GPUComplex_minus(const GPUComplex &a, const GPUComplex& b) {
    return GPUComplex(a.re-b.re, a.im-b.im);
}

__device__ const inline GPUComplex d_doublePlusGPUComplex(double a, GPUComplex& src) {
    return GPUComplex(a+src.re, src.im);
}

__device__ inline double d_GPUComplex_real( const GPUComplex& src) {
    return src.re;
}

__device__ inline double d_GPUComplex_imag( const GPUComplex& src) {
    return src.im;
}

__device__ inline void d_GPUComplex_plusEquals( GPUComplex& a, const GPUComplex & b) {
    a.re += b.re;
    a.im += b.im;
}

__device__ inline void d_GPUComplex_minusEquals( GPUComplex& a, const GPUComplex & b) {
    a.re -= b.re;
    a.im -= b.im;
}

__device__ void inline d_GPUComplex_Equals( GPUComplex& a, const GPUComplex & b) {
    a.re = b.re;
    a.im = b.im;
}

__device__ void d_print( const GPUComplex& a) {
    printf("( %f, %f) ", a.re, a.im);
    printf("\n");
}

__device__ void schsDtemp_kernel(GPUComplex &schsDtemp, GPUComplex* aqsntemp, GPUComplex *aqsmtemp, GPUComplex * I_epsR_array, int igp, int ncouls, int n1, int my_igp, int ngpown)
{
    for(int ig = 0; ig < ncouls; ++ig)
        d_GPUComplex_Equals(schsDtemp , d_GPUComplex_minus(schsDtemp , d_GPUComplex_product(d_GPUComplex_product(aqsntemp[n1*ncouls + ig] , d_GPUComplex_conj(aqsmtemp[n1*ncouls + igp])) , I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig])));
}

__global__ void d_achsDtemp_solver(GPUComplex *achsDtemp, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, double *inv_igp_index, double *indinv, int number_bands, int ncouls, int ngpown, int numThreadsPerBlock)
{
    int n1 = blockIdx.x;

    if(n1 < number_bands) 
    {
        int loopOverngpown = 1, leftOverngpown = 0;

        if(ngpown > numThreadsPerBlock)
        {
            loopOverngpown = ngpown / numThreadsPerBlock;
            leftOverngpown = ngpown % numThreadsPerBlock;
        }
        GPUComplex schsDtemp(0.00, 0.00);

        for(int x = 0; (x < loopOverngpown && threadIdx.x < numThreadsPerBlock); ++x)
        {
            int my_igp = x*numThreadsPerBlock + threadIdx.x;
            if(my_igp < ngpown)
            {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                schsDtemp_kernel(schsDtemp, aqsntemp, aqsmtemp, I_epsR_array, igp, ncouls, n1, my_igp, ngpown);
            }
        }
        if(leftOverngpown)
        {
            int my_igp = loopOverngpown * numThreadsPerBlock + threadIdx.x;
            if(my_igp < ngpown)
            {
                int indigp = inv_igp_index[my_igp];
                schsDtemp_kernel(schsDtemp, aqsntemp, aqsmtemp, I_epsR_array, indinv[indigp], ncouls, n1, my_igp, ngpown);
            }
        }

        atomicAdd(&achsDtemp->re, schsDtemp.re);
        atomicAdd(&achsDtemp->im, schsDtemp.im);
    }
}

void achsDtemp_kernel(GPUComplex *achsDtemp, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, double *inv_igp_index, double *indinv, int number_bands, int ncouls, int ngpown)
{
    int numBlocks = number_bands;
    int numThreadsPerBlock = ngpown;

    d_achsDtemp_solver<<<numBlocks, ngpown>>> (achsDtemp, aqsmtemp, aqsntemp, I_epsR_array, inv_igp_index, indinv, number_bands, ncouls, ngpown, numThreadsPerBlock);
}





