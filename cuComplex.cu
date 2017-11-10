#include "cuComplex.h"
//#include "cuDoubleComplex_cuComplex.h"
//
//#if CudaKernel
/*
 * Return the square of a complex number 
 */
__device__ const cuDoubleComplex d_cuDoubleComplex_square(cuDoubleComplex& src) {
    return make_cuDoubleComplex(src.x*src.x - src.y*src.y, 2*src.x*src.y);
}
//
///*
// * Return the conjugate of a complex number 
// */
__device__ inline cuDoubleComplex d_cuDoubleComplex_conj(cuDoubleComplex& src) {
return make_cuDoubleComplex(src.x, -src.y);
}


/*
 * Return the product of 2 complex numbers 
 */
__device__ const inline cuDoubleComplex d_cuDoubleComplex_product(const cuDoubleComplex& a, const cuDoubleComplex& b) {
    return make_cuDoubleComplex(a.x * b.x - a.y*b.y, a.x * b.y + a.y*b.x);
}


/*
 * Return the absolute of a complex number 
 */
__device__ inline double d_cuDoubleComplex_abs(const cuDoubleComplex& src) {
    return sqrt(src.x * src.x + src.y * src.y);
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
__device__ const inline cuDoubleComplex d_cuDoubleComplex_mult(cuDoubleComplex& a, double b, double c) {
    return make_cuDoubleComplex(a.x * b * c, a.y * b * c);
}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
__device__ const inline cuDoubleComplex d_cuDoubleComplex_mult(const cuDoubleComplex& a, double b) {
   return make_cuDoubleComplex(a.x*b, a.y*b);

}

/*
 * Return the complex number a += b * c  
 */
__device__ inline void d_cuDoubleComplex_fma(cuDoubleComplex& a, const cuDoubleComplex& b, const cuDoubleComplex& c) {
    a.x += b.x * c.x - b.y*c.y ;
    a.y += b.x * c.y + b.y*c.x ;
}

/*
 * Return the complex number a -= b * c  
 */
__device__ inline void d_cuDoubleComplex(cuDoubleComplex& a, const cuDoubleComplex& b, const cuDoubleComplex& c) {
    a.x -= b.x * c.x - b.y*c.y ;
    a.x -= b.x * c.y + b.y*c.x ;
}


__device__ inline cuDoubleComplex d_doubleMinuscuDoubleComplex(const double &a, cuDoubleComplex& src) {
    return make_cuDoubleComplex(a-src.x, -src.y);
}

__device__ inline cuDoubleComplex d_doubleMinuscuComplex(const double &a, cuDoubleComplex& src) {
    return make_cuDoubleComplex(a - src.x, -src.y);
}

__device__ inline cuDoubleComplex d_doublePluscuComplex(const double &a, cuDoubleComplex& src) {
    return make_cuDoubleComplex(a + src.x, src.y);
}


__device__ inline double d_cuDoubleComplex_real( const cuDoubleComplex& src) {
    return src.x;
}

__device__ inline double d_cuDoubleComplex_imag( const cuDoubleComplex& src) {
    return src.y;
}

__device__ inline void d_cuDoubleComplex_plusEquals( cuDoubleComplex& a, const cuDoubleComplex & b) {
    a.y += b.x;
    a.x += b.y;
}

__device__ void inline d_cuDoubleComplex_Equals( cuDoubleComplex& a, const cuDoubleComplex & b) {
    a.x = b.x;
    a.y = b.y;
}

__device__ __host__ void d_print( const cuDoubleComplex& a) {
    printf("( %f, %f) ", a.x, a.y);
    printf("\n");
}



__device__ void ncoulsKernel(cuDoubleComplex& mygpvar1, cuDoubleComplex& wdiff, cuDoubleComplex& aqsntemp_index, cuDoubleComplex& wtilde_array_index, cuDoubleComplex& I_eps_array_index, double vcoul_igp, double& achtemp_re_loc, double& achtemp_im_loc)
{
    double rden = 1/(wdiff.x*wdiff.x + wdiff.y*wdiff.y);

    achtemp_re_loc += d_cuDoubleComplex_product(d_cuDoubleComplex_product(mygpvar1, aqsntemp_index),\
        d_cuDoubleComplex_product(d_cuDoubleComplex_mult(d_cuDoubleComplex_product(wtilde_array_index, d_cuDoubleComplex_conj(wdiff)), rden), I_eps_array_index)).x * 0.5 * vcoul_igp;
    achtemp_im_loc += d_cuDoubleComplex_product(d_cuDoubleComplex_product(mygpvar1, aqsntemp_index),\
        d_cuDoubleComplex_product(d_cuDoubleComplex_mult(d_cuDoubleComplex_product(wtilde_array_index, d_cuDoubleComplex_conj(wdiff)), rden), I_eps_array_index)).y * 0.5 * vcoul_igp;
}

__global__  void cudaBGWKernel_ncouls_ngpown( cuDoubleComplex *wtilde_array, cuDoubleComplex *aqsntemp, cuDoubleComplex* aqsmtemp, cuDoubleComplex *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double* achtemp_re, double* achtemp_im, double* vcoul, int nstart, int nend, int* indinv, int* inv_igp_index, int numThreadsPerBlock)
{
    int n1 = blockIdx.x ;
    int my_igp = blockIdx.y;

    if(n1 < number_bands && my_igp < ngpown)
    {
        int loopOverncouls = 1, leftOverncouls = 0, \
            loopCounter = 1024;

        if(ncouls > loopCounter)
        {
            loopOverncouls = ncouls / loopCounter;
            leftOverncouls = ncouls % loopCounter;
        }

        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];

        for(int iw = nstart; iw < nend; ++iw)
        {
            double achtemp_re_loc = 0.00, achtemp_im_loc = 0.00;

            for( int x = 0; x < loopOverncouls && threadIdx.x < loopCounter ; ++x)
            {
                int ig = x*loopCounter + threadIdx.x;

                if(ig < ncouls)
                { 
                    cuDoubleComplex mygpvar1 = d_cuDoubleComplex_conj(aqsmtemp[n1*ncouls +igp]);
                    cuDoubleComplex wdiff = d_doubleMinuscuComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]);
                    ncoulsKernel(mygpvar1, wdiff, aqsntemp[n1*ncouls+ig], wtilde_array[my_igp*ncouls+ig], I_eps_array[my_igp*ncouls+ig], vcoul[igp], achtemp_re_loc, achtemp_im_loc);
                } //ncouls
            }

            if(leftOverncouls)
            {
                int ig = loopOverncouls*loopCounter + threadIdx.x;
                if(ig < ncouls)
                {
                    cuDoubleComplex mygpvar1 = d_cuDoubleComplex_conj(aqsmtemp[n1*ncouls +igp]);
                    cuDoubleComplex wdiff = d_doubleMinuscuComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]);
                    ncoulsKernel(mygpvar1, wdiff, aqsntemp[n1*ncouls+ig], wtilde_array[my_igp*ncouls+ig], I_eps_array[my_igp*ncouls+ig], vcoul[igp], achtemp_re_loc, achtemp_im_loc);
                } //ncouls
            }


//        if(n1 == 0 && my_igp == 0)
        {
            atomicAdd(&achtemp_re[iw] , achtemp_re_loc);
            atomicAdd(&achtemp_im[iw] , achtemp_im_loc );
        }

        } // iw
    }
}

void gppKernelGPU( cuDoubleComplex *wtilde_array, cuDoubleComplex *aqsntemp, cuDoubleComplex* aqsmtemp, cuDoubleComplex *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double *achtemp_re, double *achtemp_im, double *vcoul, int nstart, int nend, int* indinv, int* inv_igp_index)
{
    printf("gppKernelGPU for cuComplex class\n");
    dim3 numBlocks(number_bands, ngpown);
    int numThreadsPerBlock = ncouls;
    numThreadsPerBlock > 1024 ? numThreadsPerBlock = 1024 : numThreadsPerBlock = ncouls;
    printf("launching 2 dimension grid with (number_bands, ngpown) dime and then calling ncouls loop by threads inside ");
    cudaBGWKernel_ncouls_ngpown <<< numBlocks, numThreadsPerBlock>>> ( wtilde_array, aqsntemp, aqsmtemp, I_eps_array, ncouls, ngpown, number_bands, wx_array, achtemp_re, achtemp_im, vcoul, nstart, nend, indinv, inv_igp_index, numThreadsPerBlock);
}
