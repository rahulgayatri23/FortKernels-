#include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <complex>
#include <openacc.h>
#include <ctime>
#include <chrono>

#define N 800

#define c(x,y,z) c[x + ny* (y+nz*z)]
#define a(x,y,z) a[x + ny* (y+nz*z)]
#define b(x,y,z) b[x + ny* (y+nz*z)]

#define vl 32

int main()
{

    size_t nx=N, ny=N, nz=N;
    double *a = new double[nx * ny * nz];
    double *b = new double[nx * ny * nz];
    double *c = new double[nx * ny * nz];

#pragma acc parallel loop vector_length(vl)
    for(int i = 0; i < N; i++)
        a[i] = i;

/*
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

    auto start_chrono = std::chrono::high_resolution_clock::now();

#pragma acc parallel loop vector_length(vl)
    for(size_t z = 0; z < nz ; z++)
    {
        for(size_t y = 0; y < ny ; y++)
        {
            for(size_t x = 0; x < nx ; x++)
            {
                c(x,y,z) = a(x,y,z) * b(x,y,z);
            }
        }
    }

    std::chrono::duration<double> elapsed_chrono = std::chrono::high_resolution_clock::now() - start_chrono;

////PRINT RESULTS
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
//

    std::cout << "********** Chrono 4.5 Time Taken **********= " << elapsed_chrono.count() << " secs" << std::endl;
    */

    return 0;
}
