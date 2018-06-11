#include "testMemBounds.h"

/*
 * Return the product of 2 complex numbers 
 */
__device__ const inline GPUComplex d_GPUComplex_product(const GPUComplex& a, const GPUComplex& b) {
    return GPUComplex(a.x * b.x - a.y*b.y, a.x * b.y + a.y*b.x);
}

__device__ void inline d_GPUComplex_Equals( GPUComplex& a, const GPUComplex & b) {
    a.x = b.x;
    a.y = b.y;
}

__device__ inline double d_GPUComplex_real( const GPUComplex& src) {
    return src.x;
}

__device__ inline double d_GPUComplex_imag( const GPUComplex& src) {
    return src.y;
}

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}

__global__ void d_testKernel(double *achsDtemp_re, double*achsDtemp_im, GPUComplex *aqsmtemp, GPUComplex *aqsntemp)
{
    int n1 = blockIdx.x;


    if(n1 == 0)
        printf("From testKernel N = %d\t M = %d\t \n", N, M);

    if(n1 < N)
    {
        double achsDtemp_re_loc = 0.00, achsDtemp_im_loc = 0.00;
        int threadId = threadIdx.x;
        if(threadId == 0)
        {
            for(int ig = 0; ig < M; ++ig)
            {
                achsDtemp_re_loc += d_GPUComplex_real(d_GPUComplex_product(aqsmtemp[n1*M + ig] , aqsntemp[n1*M + ig]));
                achsDtemp_im_loc += d_GPUComplex_imag(d_GPUComplex_product(aqsmtemp[n1*M + ig] , aqsntemp[n1*M + ig]));
            }
            atomicAdd(achsDtemp_re, achsDtemp_re_loc);
            atomicAdd(achsDtemp_im, achsDtemp_im_loc);
        }
    }
}

void testMemBounds_cuKernel(GPUComplex &achsDtemp, GPUComplex *aqsmtemp, GPUComplex *aqsntemp)
{
    GPUComplex *d_aqsmtemp, *d_aqsntemp;
    double achsDtemp_re = 0.00, achsDtemp_im = 0.00;
    double *d_achsDtemp_re , *d_achsDtemp_im;

    CudaSafeCall(cudaMallocManaged((void**) &d_aqsmtemp, N * M *sizeof(GPUComplex)));
    CudaSafeCall(cudaMallocManaged((void**) &d_aqsntemp, N * M *sizeof(GPUComplex)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp_re, sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp_im, sizeof(double)));

    CudaSafeCall(cudaMemcpy(d_aqsmtemp, aqsmtemp, N*M*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsntemp, aqsntemp, N*M*sizeof(GPUComplex), cudaMemcpyHostToDevice));


    d_testKernel<<<N, 1>>> (d_achsDtemp_re, d_achsDtemp_im, d_aqsmtemp, d_aqsntemp);

    CudaSafeCall(cudaMemcpy(&achsDtemp_re, d_achsDtemp_re, sizeof(double), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(&achsDtemp_im, d_achsDtemp_im, sizeof(double), cudaMemcpyDeviceToHost));

    GPUComplex tmp(achsDtemp_re, achsDtemp_im);
    achsDtemp = tmp;

    cudaFree(d_aqsmtemp);
    cudaFree(d_aqsntemp);
    cudaFree(d_achsDtemp_re);
    cudaFree(d_achsDtemp_im);
}
