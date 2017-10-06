#include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
#include <chrono>

#define N 800

#define c(x,y,z) c[x + ny* (y+nz*z)]
#define a(x,y,z) a[x + ny* (y+nz*z)]
#define b(x,y,z) b[x + ny* (y+nz*z)]

int main()
{
    size_t nx=N, ny=N, nz=N;
    int tid, numThreads, numTeams;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << std::endl;

#pragma omp target enter data map(alloc: numTeams, numThreads)
#pragma omp target teams map(tofrom: numTeams, numThreads) shared(numTeams) private(tid)
    {
        tid = omp_get_team_num();
        if(tid == 0)
        {
            numTeams = omp_get_num_teams();
#pragma omp parallel 
            {
                int ttid = omp_get_thread_num();
                if(ttid == 0)
                    numThreads = omp_get_num_threads();
            }
        }
    }
#pragma omp target exit data map(delete: numTeams, numThreads)
    std::cout << "Number of OpenMP Teams = " << numTeams << std::endl;
    std::cout << "Number of OpenMP DEVICE Threads = " << numThreads << std::endl;

    double *a = new double[nx * ny * nz];
    double *b = new double[nx * ny * nz];
    double *c = new double[nx * ny * nz];
#pragma omp target enter data map(alloc: a[0:nx*ny*nz] , b[0:nx*ny*nz], c[0:nx*ny*nz])

#pragma omp parallel for simd collapse(3)
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
#pragma omp parallel for simd collapse(3)
    for(size_t z = 0; z < nz ; z++)
        for(size_t y = 0; y < ny ; y++)
            for(size_t x = 0; x < nx ; x++)
                c(x,y,z) = a(x,y,z) * b(x,y,z);

    std::chrono::duration<double> elapsed_chrono1 = std::chrono::high_resolution_clock::now() - start_chrono1;

#pragma omp target update to(a[0:nx*ny*nz] , b[0:nx*ny*nz], c[0:nx*ny*nz])

    auto start_chrono = std::chrono::high_resolution_clock::now();

#pragma omp target map(to:a[0:nx*ny*nz] , b[0:nx*ny*nz]) map(tofrom:c[0:nx*ny*nz])
{
#pragma omp teams distribute //shared(a , b, c)
    for(size_t z = 0; z < nz ; z++)
    {
#pragma omp parallel for 
        for(size_t y = 0; y < ny ; y++)
        {
#pragma omp simd 
            for(size_t x = 0; x < nx ; x++)
            {
                c(x,y,z) = a(x,y,z) * b(x,y,z);
            }
        }
    }
} // TARGET

    std::chrono::duration<double> elapsed_chrono = std::chrono::high_resolution_clock::now() - start_chrono;
#pragma omp target update from(c[0:nx*ny*nz])
#pragma omp target exit data map(delete: a[0:nx*ny*nz] , b[0:nx*ny*nz], c[0:nx*ny*nz])

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
    std::cout << "********** OPENMP3.0 Chrono Time Taken **********= " << elapsed_chrono1.count() << " secs" << std::endl;

    return 0;
}
