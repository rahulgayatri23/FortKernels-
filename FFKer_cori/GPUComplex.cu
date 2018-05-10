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

__device__ inline GPUComplex d_GPUComplex_plus(const GPUComplex& a, const GPUComplex& b) {
    return GPUComplex(a.re+b.re, a.im+b.im);
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

__device__ void ssxDitttSolver(GPUComplex *I_epsR_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, int ifreq, int my_igp, double fact1, double fact2, int igp, int ncouls, int ngpown, int n1, GPUComplex& ssxDitt)
{
    for(int ig = 0; ig < ncouls ; ++ig)
    {
        GPUComplex ssxDit = d_GPUComplex_plus(d_GPUComplex_mult(I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] , fact1 ) , \
            d_GPUComplex_mult(I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] , fact2));

        d_GPUComplex_plusEquals(ssxDitt , d_GPUComplex_product(aqsntemp[n1*ncouls + ig] , d_GPUComplex_product(d_GPUComplex_conj(aqsmtemp[n1*ncouls + igp]) , ssxDit)));
    }
}

__device__ void d_compute_ssxDitt(GPUComplex *I_eps_array, double fact1, double fact2, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, int ifreq, int ngpown, int ncouls, int my_igp, int ig, int n1, GPUComplex &ssxDitt, int igp)
{
    GPUComplex ssxDit(0.00, 0.00);
    d_GPUComplex_Equals(ssxDit, d_GPUComplex_plus(d_GPUComplex_mult(I_eps_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] , fact1 ) , \
                                 d_GPUComplex_mult(I_eps_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] , fact2)));
    d_GPUComplex_plusEquals(ssxDitt, d_GPUComplex_product(aqsntemp[n1*ncouls + ig] , d_GPUComplex_product(d_GPUComplex_conj(aqsmtemp[n1*ncouls + igp]) , ssxDit)));
}

__global__ void d_asxDtemp_solver(GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, int ncouls, int ngpown, int nfreqeval, double *vcoul, int *inv_igp_index, int *indinv, double fact1, double fact2, double wx, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *asxDtemp, int numThreadsPerBlock, int ifreq, int n1, double occ, GPUComplex *ssxDi)
{
    int iw = blockIdx.x;
    if(iw < nfreqeval)
    {
        __shared__ GPUComplex ssxDittt;
//        GPUComplex *ssxDittt_threadArr = new GPUComplex[numThreadsPerBlock];;
        double *ssxDittt_rethreadArr = new double[numThreadsPerBlock];;
        double *ssxDittt_imthreadArr = new double[numThreadsPerBlock];;

        int loopOverngpown = 1, leftOverngpown = 0;
        if(ngpown > numThreadsPerBlock)
        {
            loopOverngpown = ngpown / numThreadsPerBlock;
            leftOverngpown = ngpown % numThreadsPerBlock;
        }
        int loopOverncouls = 1, leftOverncouls = 0;
        if(ncouls > numThreadsPerBlock)
        {
            loopOverncouls = ncouls / numThreadsPerBlock;
            leftOverncouls = ncouls % numThreadsPerBlock;
        }

//        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        for(int x = 0; (x < loopOverngpown && threadIdx.x < numThreadsPerBlock); ++x)
        {
            int my_igp = x*loopOverngpown + threadIdx.x;
            if(my_igp < ngpown)
            {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                GPUComplex ssxDitt(0.00, 0.00);

                for(int ig = 0; ig < ncouls; ++ig)
                {
                    if(wx > 0.00)
                        d_compute_ssxDitt(I_epsR_array, fact1, fact2, aqsntemp, aqsmtemp, ifreq, ngpown, ncouls, my_igp, ig, n1, ssxDitt, igp);
                    else
                        d_compute_ssxDitt(I_epsA_array, fact1, fact2, aqsntemp, aqsmtemp, ifreq, ngpown, ncouls, my_igp, ig, n1, ssxDitt, igp);
                }
                d_GPUComplex_plusEquals(ssxDittt, d_GPUComplex_mult(ssxDitt, vcoul[igp]));
//                ssxDittt_rethreadArr[threadIdx.x] += (d_GPUComplex_mult(ssxDitt, vcoul[igp])).re;
//                ssxDittt_imthreadArr[threadIdx.x] += (d_GPUComplex_mult(ssxDitt, vcoul[igp])).im;
            }
        }
        if(leftOverngpown)
        {
            int my_igp = numThreadsPerBlock*loopOverngpown + threadIdx.x;
            if(my_igp < ngpown)
            {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                GPUComplex ssxDitt(0.00, 0.00);

                for(int ig = 0; ig < ncouls; ++ig)
                {
                    if(wx > 0.00)
                        d_compute_ssxDitt(I_epsR_array, fact1, fact2, aqsntemp, aqsmtemp, ifreq, ngpown, ncouls, my_igp, ig, n1, ssxDitt, igp);
                    else
                        d_compute_ssxDitt(I_epsA_array, fact1, fact2, aqsntemp, aqsmtemp, ifreq, ngpown, ncouls, my_igp, ig, n1, ssxDitt, igp);
                }
                d_GPUComplex_plusEquals(ssxDittt, d_GPUComplex_mult(ssxDitt, vcoul[igp]));
//                ssxDittt_rethreadArr[threadIdx.x] += (d_GPUComplex_mult(ssxDitt, vcoul[igp])).re;
//                ssxDittt_imthreadArr[threadIdx.x] += (d_GPUComplex_mult(ssxDitt, vcoul[igp])).im;
            }
        }
        for(int i = 0; i < numThreadsPerBlock; ++i)
        {
            GPUComplex tmp(ssxDittt_rethreadArr[i], ssxDittt_imthreadArr[i]);
            d_GPUComplex_plusEquals(ssxDittt, tmp);
        }

        d_GPUComplex_plusEquals(ssxDi[iw] , ssxDittt);
        d_GPUComplex_plusEquals(asxDtemp[iw] , d_GPUComplex_mult(ssxDittt, occ));
    }
}

__global__ void d_asxDtemp_solver_negwx(GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, int ncouls, int ngpown, int nfreqeval, double *vcoul, int *inv_igp_index, int *indinv, double fact1, double fact2, double wx, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *asxDtemp, int numThreadsPerBlock, int ifreq, int n1)
{
    int iw = blockIdx.x;
    if(iw < nfreqeval)
    {
        int loopOverngpown = 1, leftOverngpown = 0;
        if(ngpown > numThreadsPerBlock)
        {
            loopOverngpown = ngpown / numThreadsPerBlock;
            leftOverngpown = ngpown % numThreadsPerBlock;
        }

        GPUComplex ssxDittt(0.00, 0.00);

        for(int x = 0; (x < loopOverngpown && threadIdx.x < numThreadsPerBlock); ++x)
        {
            int my_igp = x*numThreadsPerBlock + threadIdx.x;
            if(my_igp < ngpown)
            {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                GPUComplex ssxDitt(0.00, 0.00);

                ssxDitttSolver(I_epsA_array, aqsmtemp, aqsntemp, ifreq, my_igp, fact1, fact2, igp, ncouls, ngpown, n1, ssxDitt);

                d_GPUComplex_plusEquals(ssxDittt, d_GPUComplex_mult(ssxDitt, vcoul[igp]));
            }
        }
        d_GPUComplex_plusEquals(asxDtemp[iw] , ssxDittt);
    }
}

__global__ void d_achsDtemp_solver(GPUComplex *achsDtemp, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, int *inv_igp_index, int *indinv, double* vcoul, int number_bands, int ncouls, int ngpown, int numThreadsPerBlock)
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
        GPUComplex achsDtemp_loc(0.00, 0.00);

        for(int x = 0; (x < loopOverngpown && threadIdx.x < numThreadsPerBlock); ++x)
        {
            int my_igp = x*numThreadsPerBlock + threadIdx.x;
            if(my_igp < ngpown)
            {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                schsDtemp_kernel(schsDtemp, aqsntemp, aqsmtemp, I_epsR_array, igp, ncouls, n1, my_igp, ngpown);
                d_GPUComplex_Equals(achsDtemp_loc, d_GPUComplex_mult(schsDtemp, vcoul[igp]*0.5));
            }
        }
        if(leftOverngpown)
        {
            int my_igp = loopOverngpown * numThreadsPerBlock + threadIdx.x;
            if(my_igp < ngpown)
            {
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                schsDtemp_kernel(schsDtemp, aqsntemp, aqsmtemp, I_epsR_array, igp, ncouls, n1, my_igp, ngpown);
                d_GPUComplex_Equals(achsDtemp_loc, d_GPUComplex_mult(schsDtemp, vcoul[igp]*0.5));
            }
        }

        atomicAdd(&achsDtemp->re, achsDtemp_loc.re);
        atomicAdd(&achsDtemp->im, achsDtemp_loc.im);
    }
}

void achsDtemp_kernel(GPUComplex *achsDtemp, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, int *inv_igp_index, int *indinv, double* vcoul, int number_bands, int ncouls, int ngpown)
{
    int numBlocks = number_bands;
    int numThreadsPerBlock = ngpown;

    d_achsDtemp_solver<<<numBlocks, ngpown>>> (achsDtemp, aqsmtemp, aqsntemp, I_epsR_array, inv_igp_index, indinv, vcoul, number_bands, ncouls, ngpown, numThreadsPerBlock);
    
}

void asxDtemp_kernel(GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, int ncouls, int ngpown, int nfreqeval, double *vcoul, int *inv_igp_index, int *indinv, double fact1, double fact2, double wx, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *asxDtemp, int ifreq, int n1, GPUComplex *ssxDi, double occ)
{
    int numBlocks = nfreqeval;
    int numThreadsPerBlock = 32;
    d_asxDtemp_solver<<<numBlocks, numThreadsPerBlock>>>(I_epsR_array, I_epsA_array, ncouls, ngpown, nfreqeval, vcoul, inv_igp_index, indinv, fact1, fact2, wx, aqsmtemp, aqsntemp, asxDtemp, numThreadsPerBlock, ifreq, n1, occ, ssxDi);
}

