#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <chrono>

using namespace std;
#define N 15000

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

GPUComplex operator =(const GPUComplex src) {
    re = src.re;
    im = src.im;

    return *this;
}

};

int main(int argc, char** argv)
{
    GPUComplex expr( 0.5 , 0.5);

    long double mem_alloc = 0.00;


    GPUComplex *arr= new GPUComplex[N * N];
    mem_alloc += (N * N * sizeof(GPUComplex));

    cout << "Memory Used = " << mem_alloc/(1024 * 1024 * 1024) << " GB" << endl;

    for(int i=0; i<N; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            arr[i*N+j] = expr;
        }

    }

//Free the allocated memory 
    free(arr);

    return 0;
}
