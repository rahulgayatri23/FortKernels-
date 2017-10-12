#include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
#include <chrono>

#include "Complex.h"

using namespace std;
int debug = 0;

#pragma omp declare target
inline void flagOCC_solver(double wxt, GPUComplex *wtilde_array, int my_igp, int n1, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, GPUComplex &ssxt, GPUComplex &scht,int ncouls, int igp, int number_bands, int ngpown);
inline void reduce_achstemp(int n1, int number_bands, int* inv_igp_index, int ncouls, GPUComplex  *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, GPUComplex achstemp,  int* indinv, int ngpown, double* vcoul, int numThreads);
#pragma omp end declare target

int main(int argc, char** argv)
{

    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> " << endl;
        exit (0);
    }
    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int nodes_per_group = atoi(argv[4]);

    int npes = 1; //Represents the number of ranks per node
    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

    double e_lk = 10;
    double dw = 1;
    int nstart = 0, nend = 3;

    int inv_igp_index[ngpown];
    int indinv[ncouls];

    //OpenMP variables
    int tid, numThreads;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;

    double to1 = 1e-6, \
    gamma = 0.5, \
    sexcut = 4.0;
    double limitone = 1.0/(to1*4.0), \
    limittwo = pow(0.5,2);

    double e_n1kq= 6.0; //This in the fortran code is derived through the double dimenrsion array ekq whose 2nd dimension is 1 and all the elements in the array have the same value

    //Printing out the params passed.
    std::cout << "number_bands = " << number_bands \
        << "\t nvband = " << nvband \
        << "\t ncouls = " << ncouls \
        << "\t nodes_per_group  = " << nodes_per_group \
        << "\t ngpown = " << ngpown \
        << "\t nend = " << nend \
        << "\t nstart = " << nstart \
        << "\t gamma = " << gamma \
        << "\t sexcut = " << sexcut \
        << "\t limitone = " << limitone \
        << "\t limittwo = " << limittwo << endl;


    //ALLOCATE statements from fortran gppkernel.
    
   
    GPUComplex expr0(0.00, 0.00);
    GPUComplex expr(0.5, 0.5);

    GPUComplex *acht_n1_loc = new GPUComplex[number_bands];
    GPUComplex *achtemp = new GPUComplex[nend-nstart];
    GPUComplex *asxtemp = new GPUComplex[nend-nstart];
    GPUComplex *aqsmtemp = new GPUComplex[number_bands*ncouls];
    GPUComplex *aqsntemp = new GPUComplex[number_bands*ncouls];
    GPUComplex *I_eps_array = new GPUComplex[ngpown*ncouls];
    GPUComplex *wtilde_array = new GPUComplex[ngpown*ncouls];
    GPUComplex *ssx_array = new GPUComplex[3];
    GPUComplex *sch_array = new GPUComplex[3];
    GPUComplex *scha = new GPUComplex[ncouls];
    GPUComplex *ssxa = new GPUComplex[ncouls];
    GPUComplex achstemp;

    double *achtemp_re = new double[3];
    double *achtemp_im = new double[3];
                        
    double *vcoul = new double[ncouls];
    double wx_array[3];
    double occ=1.0;
    bool flag_occ;
    double achstemp_real = 0.00, achstemp_imag = 0.00;
    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;

//#pragma omp target enter data map(alloc:aqsmtemp[0:number_bands*ncouls],aqsntemp[0:number_bands*ncouls], asxtemp[0:(nend-nstart)], sch_array[0:3], scha[0:ncouls], achtemp_re[0:3], achtemp_im[0:3], I_eps_array[0:ngpown*ncouls], wx_array[0:3], wtilde_array[0:ngpown*ncouls])
//#pragma omp target map (to:expr, ngpown, ncouls, number_bands)
//{
//#pragma omp teams distribute parallel for simd collapse(2)
   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsmtemp[i*ncouls+j] = expr;
           aqsntemp[i*ncouls+j] = expr;
       }

//#pragma omp teams distribute parallel for simd collapse(2)
   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i*ncouls+j] = expr;
           wtilde_array[i*ncouls+j] = expr;
       }

//#pragma omp teams distribute parallel for simd 
    for(int ig=0; ig < ngpown; ++ig)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
//#pragma omp teams distribute parallel for simd 
    for(int ig=0; ig<ncouls; ++ig)
        indinv[ig] = ig;

    auto start_chrono = std::chrono::high_resolution_clock::now();


#pragma omp target enter data map(alloc:aqsmtemp[0:number_bands*ncouls],aqsntemp[0:number_bands*ncouls], asxtemp[0:(nend-nstart)], sch_array[0:3], scha[0:ncouls], achtemp_re[0:3], achtemp_im[0:3], I_eps_array[0:ngpown*ncouls], wx_array[0:3], wtilde_array[0:ngpown*ncouls])
#pragma omp target update to(aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wx_array[0:3], wtilde_array[0:ngpown*ncouls])

#pragma omp target
{
//#pragma omp teams distribute parallel for
        for(int iw=nstart; iw<nend; ++iw)
        {
           sch_array[iw] = expr0;
           achtemp_re[iw] = 0.00;
           achtemp_im[iw] = 0.00;
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(wx_array[iw] < to1) wx_array[iw] = to1;
        }
#pragma omp teams distribute shared(wtilde_array, aqsntemp, aqsmtemp, I_eps_array, wx_array) firstprivate(sch_array) reduction(+: achtemp_re[0:3], achtemp_im[0:3])
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            int igp = ncouls-1;
            GPUComplex mygpvar1;
            mygpvar1 = GPUComplex_conj(aqsmtemp[ncouls+igp]);
            int iw;

#pragma omp parallel for firstprivate(mygpvar1) //simd schedule(static) //nowait
           for(int ig = 0; ig<ncouls; ++ig)
           {
                GPUComplex scht = expr0; ;
                iw = 0; 
                sch_array[iw] += GPUComplex_product(mygpvar1 , aqsntemp[ncouls+ig]);
                
//                iw = 1;
//                sch_array[iw] += GPUComplex_product(mygpvar1 , aqsntemp[ncouls+ig]);
//                
//                iw = 2;
//                sch_array[iw] += GPUComplex_product(mygpvar1 , aqsntemp[ncouls+ig]);
        }

        for(int iw=nstart; iw<nend; ++iw)
        {
            achtemp_re[iw] += GPUComplex_real( sch_array[iw]);
            achtemp_im[iw] += GPUComplex_imag( sch_array[iw]);
        }

        } //ngpown
} //TARGET
#pragma omp target update from (acht_n1_loc[0:number_bands], asxtemp[0:(nend-nstart)], achtemp_re[0:(nend-nstart)], achtemp_im[0:(nend-nstart)] )

#pragma omp target exit data map(delete: acht_n1_loc[:0], aqsmtemp[:0],aqsntemp[:0], scha[:0], asxtemp[:0])


    printf(" \n Final achstemp\n");
    achstemp.print();
    std::chrono::duration<double> elapsed_chrono = std::chrono::high_resolution_clock::now() - start_chrono;

    printf("\n Final achtemp\n");
    for(int iw=nstart; iw<nend; ++iw)
    {
        GPUComplex tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
        achtemp[iw].print();
    }

//    cout << "********** Chrono Time Taken **********= " << elapsed_chrono.count() << " secs" << endl;

    free(acht_n1_loc);
    free(achtemp);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(wtilde_array);
    free(asxtemp);
    free(vcoul);
    free(ssx_array);

    return 0;
}
//#pragma omp parallel for shared(vcoul, wtilde_array, aqsntemp, aqsmtemp, I_eps_array, wx_array, ssx_array) reduction(+: achtemp_re[0:3], achtemp_im[0:3]) schedule(dynamic) 
