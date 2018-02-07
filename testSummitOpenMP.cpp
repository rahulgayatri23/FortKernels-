#include <iostream>
#include <omp.h>
#include <ctime>
#include <chrono>
using namespace std;

#pragma omp declare target
class GPUComplex {

    private : 
        double re;
        double im;

public:

explicit GPUComplex () {
    re = 0.00;
    im = 0.00;
}


explicit GPUComplex(const double& x, const double& y) {
    re = x;
    im = y;
}

GPUComplex(const GPUComplex& src) {
    re = src.re;
    im = src.im;
}

GPUComplex& operator =(const GPUComplex& src) {
    re = src.re;
    im = src.im;

    return *this;
}

GPUComplex& operator +=(const GPUComplex& src) {
    re = src.re + this->re;
    im = src.im + this->im;

    return *this;
}

GPUComplex& operator -=(const GPUComplex& src) {
    re = src.re - this->re;
    im = src.im - this->im;

    return *this;
}

GPUComplex& operator -() {
    re = -this->re;
    im = -this->im;

    return *this;
}

GPUComplex& operator ~() {
    return *this;
}
};
#pragma omp end declare target


int main(int argc, char** argv)
{

    auto start_totalTime = std::chrono::high_resolution_clock::now();

    int number_bands = 512;

    //OpenMP Printing of threads on Host and Device
    int tid, numThreads, numTeams;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;

#pragma omp target enter data map(alloc: numTeams, numThreads)
#pragma omp target map(tofrom: numTeams, numThreads)
#pragma omp teams shared(numTeams) private(tid)
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

    GPUComplex *wtilde_array = new GPUComplex[number_bands];
    GPUComplex expr(0.5, 0.5);

   for(int i=0; i<number_bands; i++)
           wtilde_array[i] = expr;

    auto start_chrono = std::chrono::high_resolution_clock::now();

#pragma omp target enter data map(alloc: wtilde_array[0:number_bands])
#pragma omp target update to(wtilde_array[0:number_bands])
#pragma omp target teams distribute parallel for map(to:wtilde_array[0:number_bands])
    for(int n1 = 0; n1<number_bands; ++n1) 
        GPUComplex wdiff =  wtilde_array[0];

#pragma omp target exit data map(delete: wtilde_array[:0])


    std::chrono::duration<double> elapsed_totalTime = std::chrono::high_resolution_clock::now() - start_totalTime;
    cout << "********** Time Taken **********= " << elapsed_totalTime.count() << " secs" << endl;

    free(wtilde_array);

    return 0;
}
