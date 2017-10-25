#include <iostream>
#include <cstdlib>
#include <memory>
#include <algorithm>

#include <cmath>
#include <chrono>

#define N 3

#define c(x,y,z) c[x + ny* (y+nz*z)]
#define a(x,y,z) a[x + ny* (y+nz*z)]
#define b(x,y,z) b[x + ny* (y+nz*z)]

void print_results(double* a)
{
    size_t nx = N, ny = N, nz = N;
    for(size_t z = 0; z < nz ; z++)
    {
        for(size_t y = 0; y < ny ; y++)
        {
            for(size_t x = 0; x < nx ; x++)
            {
                std::cout << a(x,y,z) ;
            }
            std::cout << "\t" ;
        }
        std::cout << "\n" ;
    }

}

void vecAddCPU(double* a, double* b, double* c, size_t x_, size_t y, size_t z, size_t blockSize)
{
    size_t ny=N, nz=N;
    for(size_t x = x_; x < x_+blockSize ; x++)
        c(x,y,z) = a(x,y,z) + b(x,y,z);
}

__global__ void vecAddGPU(double* a, double* b, double* c)
{
    int i = threadIdx.x;
    if(i < N*N*N)
        c[i] = a[i] + b[i];
//    size_t ny=N, nz=N;
//    for(size_t x = x_; x < x_+blockSize ; x++)
//        c(x,y,z) = a(x,y,z) + b(x,y,z);
}

int main()
{
    size_t nx=N, ny=N, nz=N;
    size_t blockSize = 1;

    double *a, *b, *c;

    a = (double *) malloc(nx*ny*nx*sizeof(double));
    b = (double *) malloc(nx*ny*nx*sizeof(double));
    c = (double *) malloc(nx*ny*nx*sizeof(double));

    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, nx*ny*nz*sizeof(double));
    cudaMalloc(&d_b, nx*ny*nz*sizeof(double));
    cudaMalloc(&d_c, nx*ny*nz*sizeof(double));

    for(size_t z = 0; z < nz ; z++)
    {
        for(size_t y = 0; y < ny ; y++)
        {
            for(size_t x = 0; x < nx ; x++)
            {
                a(x,y,z) = x;
                b(x,y,z) = y;
            }
        }
    }

    cudaMemcpy(d_a, a, nx*ny*nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, nx*ny*nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, nx*ny*nz*sizeof(double), cudaMemcpyHostToDevice);

    auto start_chrono1 = std::chrono::high_resolution_clock::now();

    for(size_t z = 0; z < nz ; z++)
        for(size_t y = 0; y < ny ; y++)
            for(size_t x = 0; x < nx ; x+=blockSize)
            {
                int blockEnd = std::min(nx, x+blockSize);
                blockEnd = nx ? blockSize = nx-x : blockSize = blockSize;
//                vecAddCPU(a, b, c, x, y, z, blockSize);
                vecAddGPU<<< 1, nx*ny*nz>>> (d_a, d_b, d_c);
            }

    cudaMemcpy(a, d_a, nx*ny*nz*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, nx*ny*nz*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_c, nx*ny*nz*sizeof(double), cudaMemcpyDeviceToHost);

    std::chrono::duration<double> elapsed_chrono1 = std::chrono::high_resolution_clock::now() - start_chrono1;

//PRINT RESULTS
    print_results(c);

    std::cout << "********** OPENMP3.0 Chrono Time Taken **********= " << elapsed_chrono1.count() << " secs" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
