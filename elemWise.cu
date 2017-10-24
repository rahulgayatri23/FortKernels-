#include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
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

void(double* a, double* b, double* c, size_t blockSize)
{
}

int main()
{
    size_t nx=N, ny=N, nz=N;
    size_t blockSize = 1;

    double *a = new double[nx * ny * nz];
    double *b = new double[nx * ny * nz];
    double *c = new double[nx * ny * nz];

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
                vecAdd(&a(x,y,z), &b(x,y,z), &c(x,y,z), blockSize);
                c(x,y,z) = a(x,y,z) + b(x,y,z);
            }

    std::chrono::duration<double> elapsed_chrono1 = std::chrono::high_resolution_clock::now() - start_chrono1;

//PRINT RESULTS
    print_results(c);
//    for(size_t z = 0; z < nz ; z++)
//    {
//        for(size_t y = 0; y < ny ; y++)
//        {
//            for(size_t x = 0; x < nx ; x++)
//            {
//                std::cout << c(x,y,z) ;
//            }
//            std::cout << "\t" ;
//        }
//        std::cout << "\n" ;
//    }


    std::cout << "********** OPENMP3.0 Chrono Time Taken **********= " << elapsed_chrono1.count() << " secs" << std::endl;

    return 0;
}
