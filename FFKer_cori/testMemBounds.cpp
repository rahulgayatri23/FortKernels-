#include "testMemBounds.h"

int main(int argc, char** argv)
{
//Start to allocate the data structures;
    long double mem_alloc = 0.00;

    GPUComplex *aqsntemp = new GPUComplex[N * M];
    mem_alloc += (N * M * sizeof(GPUComplex));

    GPUComplex *aqsmtemp= new GPUComplex[N * M];
    mem_alloc += (N * M * sizeof(GPUComplex));


    //Initialize the data structures
    GPUComplex expr( 0.5 , 0.5);
    for(int i=0; i<N; ++i)
    {
        for(int j=0; j<M; ++j)
        {
            aqsmtemp[i*M+j] = expr;
            aqsntemp[i*M+j] = expr;
        }

    }

    cout << "*************************************************************************" << endl;
    cout << "\nN = " << N << \
        "\nM = " << M << endl;
    cout << "Memory Used = " << mem_alloc/(1024 * 1024 * 1024) << " GB" << endl;

#if !CUDAVER    
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
    //OpenMP Printing of threads on Host and Device
    std::cout << "Number of OpenMP Teams = " << numTeams << std::endl;
    std::cout << "Number of OpenMP DEVICE Threads = " << numThreads << std::endl;



    cout << "starting Kernels" << endl;
    double achsDtemp_re = 0.00, achsDtemp_im = 0.00;

#pragma omp target teams distribute \
    map(to: aqsmtemp[0:N*M], aqsntemp[0:N*M]) \
    map(tofrom: achsDtemp_re, achsDtemp_im) \
    reduction(+: achsDtemp_re, achsDtemp_im) \
    num_teams(N)
    for(int n1 = 0; n1 < N; ++n1)
    {
        for(int ig = 0; ig < M; ++ig)
        {
            achsDtemp_re += (GPUComplex_product(aqsmtemp[n1*M + ig] , aqsntemp[n1*M + ig])).get_real();
            achsDtemp_im += (GPUComplex_product(aqsmtemp[n1*M + ig] , aqsntemp[n1*M + ig])).get_imag();
        }
    } //n1
    GPUComplex achsDtemp(achsDtemp_re, achsDtemp_im);
    cout << "achsDtemp = " ;
    achsDtemp.print();

#else
    GPUComplex achsDtemp(0.00, 0.00);
    testMemBounds_cuKernel(achsDtemp, aqsmtemp, aqsntemp);
    cout << "achsDtemp = " ;
    achsDtemp.print();
#endif


//Free the allocated memory 
    free(aqsntemp);
    free(aqsmtemp);

    return 0;
}
