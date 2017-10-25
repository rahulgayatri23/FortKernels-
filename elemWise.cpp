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
    size_t nx=N, ny=N, nz=N;
    for(size_t x = x_; x < x_+blockSize ; x++)
        c(x,y,z) = a(x,y,z) + b(x,y,z);
}

int main()
{
    size_t nx=N, ny=N, nz=N;
    size_t blockSize = 2;

    double* a = (double *) malloc(nx*ny*nx*sizeof(double));
    double* b = (double *) malloc(nx*ny*nx*sizeof(double));
    double* c = (double *) malloc(nx*ny*nx*sizeof(double));

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

    auto start_chrono1 = std::chrono::high_resolution_clock::now();

    for(size_t z = 0; z < nz ; z++)
        for(size_t y = 0; y < ny ; y++)
            for(size_t x = 0; x < nx ; x+=blockSize)
            {
                int blockEnd = std::min(nx, x+blockSize);
                blockEnd = nx ? blockSize = nx-x : blockSize = blockSize;
                vecAddCPU(a, b, c, x, y, z, blockSize);
 //               c(x,y,z) = a(x,y,z) + b(x,y,z);
            }

    std::chrono::duration<double> elapsed_chrono1 = std::chrono::high_resolution_clock::now() - start_chrono1;

//PRINT RESULTS
    print_results(c);

    std::cout << "********** OPENMP3.0 Chrono Time Taken **********= " << elapsed_chrono1.count() << " secs" << std::endl;

    free(a);
    free(b);
    free(c);

    return 0;
}
