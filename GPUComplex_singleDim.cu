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
__device__ void ncoulsKernel(GPUComplex& mygpvar1, GPUComplex& wdiff, GPUComplex& aqsntemp_index, GPUComplex& wtilde_array_index, GPUComplex& I_eps_array_index, double vcoul_igp, double& achtemp_re_loc, double& achtemp_im_loc)
{
    double rden = d_GPUComplex_real(d_GPUComplex_product(wdiff, d_GPUComplex_conj(wdiff)));
    rden = 1/rden;
    
    achtemp_re_loc += d_GPUComplex_real(\
        d_GPUComplex_mult(\
        d_GPUComplex_mult(\
        d_GPUComplex_product(d_GPUComplex_product(mygpvar1, aqsntemp_index),\
        d_GPUComplex_product(d_GPUComplex_mult(d_GPUComplex_product(wtilde_array_index, d_GPUComplex_conj(wdiff)), rden), I_eps_array_index)),\
        0.5), \ // mult = 0.5
        vcoul_igp)); // mult = vcoul[igp]
    
    achtemp_im_loc += d_GPUComplex_imag(\
        d_GPUComplex_mult(\
        d_GPUComplex_mult(\
        d_GPUComplex_product(d_GPUComplex_product(mygpvar1, aqsntemp_index),\
        d_GPUComplex_product(d_GPUComplex_mult(d_GPUComplex_product(wtilde_array_index, d_GPUComplex_conj(wdiff)), rden), I_eps_array_index)),\
        0.5), \ // mult = 0.5
        vcoul_igp)); // mult = vcoul[igp]

}

__global__  void cudaBGWKernel( GPUComplex *wtilde_array, GPUComplex *aqsntemp, GPUComplex* aqsmtemp, GPUComplex *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double* achtemp_re, double* achtemp_im, double* vcoul, int nstart, int nend, int* indinv, int* inv_igp_index, int numBlocks, int numThreadsPerBlock)
{
    int n1 = blockIdx.x ;

    if(n1 < number_bands)
    {
        int loopOverngpown = 1, leftOverngpown = 0, \
            loopCounter = 1024;

        if(ngpown > loopCounter)
        {
            loopOverngpown = ngpown / loopCounter;
            leftOverngpown = ngpown % loopCounter;
        }

        for( int x = 0; x < loopOverngpown && threadIdx.x < loopCounter; ++x)
        {
            int my_igp = x*loopCounter + threadIdx.x;
        
            if(my_igp < ngpown)
            {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                if(indigp == ncouls)
                    igp = ncouls-1;

                for(int iw = nstart; iw < nend; ++iw)
                {
                    double achtemp_re_loc = 0.00, achtemp_im_loc = 0.00;

                    for(int ig = 0; ig < ncouls; ++ig) 
                    { 
                        GPUComplex mygpvar1 = d_GPUComplex_conj(aqsmtemp[n1*ncouls+igp]);
                        GPUComplex wdiff = d_doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]);
                        ncoulsKernel(mygpvar1, wdiff, aqsntemp[n1*ncouls+ig], wtilde_array[my_igp*ncouls+ig], I_eps_array[my_igp*ncouls+ig], vcoul[igp], achtemp_re_loc, achtemp_im_loc);

                    } //ncouls

                    atomicAdd(&achtemp_re[iw] , achtemp_re_loc);
                    atomicAdd(&achtemp_im[iw] , achtemp_im_loc );
                } // iw
            } // ngpown
        }

        if(leftOverngpown)
        {
            int my_igp = loopOverngpown*loopCounter + threadIdx.x;
            if(my_igp < ngpown)
            {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                if(indigp == ncouls)
                    igp = ncouls-1;

                for(int iw = nstart; iw < nend; ++iw)
                {
                    double achtemp_re_loc = 0.00, achtemp_im_loc = 0.00;

                    for(int ig = 0; ig < ncouls; ++ig) 
                    { 
                        GPUComplex mygpvar1 = d_GPUComplex_conj(aqsmtemp[n1*ncouls+igp]);
                        GPUComplex wdiff = d_doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]);
                        ncoulsKernel(mygpvar1, wdiff, aqsntemp[n1*ncouls+ig], wtilde_array[my_igp*ncouls+ig], I_eps_array[my_igp*ncouls+ig], vcoul[igp], achtemp_re_loc, achtemp_im_loc);

                    } //ncouls

                        atomicAdd(&achtemp_re[iw] , achtemp_re_loc);
                        atomicAdd(&achtemp_im[iw] , achtemp_im_loc );
                } // iw
            } // ngpown
        }
    }
}

void gppKernelGPU( GPUComplex *wtilde_array, GPUComplex *aqsntemp, GPUComplex* aqsmtemp, GPUComplex *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double *achtemp_re, double *achtemp_im, double *vcoul, int numBlocks, int numThreadsPerBlock, int nstart, int nend, int* indinv, int* inv_igp_index)
{
    cudaBGWKernel <<< numBlocks, numThreadsPerBlock>>> ( wtilde_array, aqsntemp, aqsmtemp, I_eps_array, ncouls, ngpown, number_bands, wx_array, achtemp_re, achtemp_im, vcoul, nstart, nend, indinv, inv_igp_index, numBlocks, numThreadsPerBlock);
}

#endif
