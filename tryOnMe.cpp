#include <iostream>
#include <omp.h>
#include <chrono>
#include <complex>
#include <likwid.h>

using namespace std;
#define N 100

int main(int argc, char** argv)
{

    LIKWID_MARKER_INIT;
    std::complex<double> *a = new std::complex<double> [N];
    std::complex<double> *b = new std::complex<double> [N];
    std::complex<double> *c = new std::complex<double> [N];

    std::complex<double> expr(0.5, 0.5);

    for(int i = 0; i < N ; ++i)
    {
        a[i] = expr;
        b[i] = expr;
        c[i] = expr;
    }


    auto start_timer = std::chrono::high_resolution_clock::now();
    //Vector Add
   LIKWID_MARKER_START("vecadd"); 
#pragma omp parallel for
    for(int i = 0; i < N ; ++i)
        a[i] = b[i] + c[i];

   LIKWID_MARKER_STOP("vecadd"); 

    std::chrono::duration<double> elapsed_time = std::chrono::high_resolution_clock::now() - start_timer;
    cout << "********** Time Taken **********= " << elapsed_time.count() << " secs" << endl;

    LIKWID_MARKER_CLOSE;

    free(a);
    free(b);
    free(c);

    return 0;
}
