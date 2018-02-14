#include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
#include <chrono>

#include "GPUComplex.h"

using namespace std;
int debug = 0;

#pragma omp declare target
inline void flagOCC_solver(double wxt, GPUComplex *wtilde_array, int my_igp, int n1, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, GPUComplex &ssxt, GPUComplex &scht,int ncouls, int igp, int number_bands, int ngpown);
inline void reduce_achstemp(int n1, int number_bands, int* inv_igp_index, int ncouls, GPUComplex  *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, GPUComplex achstemp,  int* indinv, int ngpown, double* vcoul, int numThreads);
#pragma omp end declare target

int main(int argc, char** argv)
{
    auto start_totalTime = std::chrono::high_resolution_clock::now();

    int number_bands = 512;
    int nvband = 2;
    int ncouls = 512;
    int nodes_per_group = 20;


    int ngpown = ncouls / nodes_per_group; 

    double e_lk = 10;
    double dw = 1;
    int nstart = 0, nend = 3;

    int inv_igp_index[ngpown];
    int indinv[ncouls+1];

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


   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsmtemp[i*ncouls+j] = expr;
           aqsntemp[i*ncouls+j] = expr;
       }

   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i*ncouls+j] = expr;
           wtilde_array[i*ncouls+j] = expr;
       }

   for(int i=0; i<ncouls; i++)
       vcoul[i] = 1.0;


    for(int ig=0, tmp=1; ig < ngpown; ++ig,tmp++)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0, tmp=1; ig<ncouls; ++ig,tmp++)
        indinv[ig] = ig;
        indinv[ncouls] = ncouls-1;

       for(int iw=nstart; iw<nend; ++iw)
       {
           asxtemp[iw] = expr0;
           achtemp_re[iw] = 0.00;
           achtemp_im[iw] = 0.00;
       }

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(wx_array[iw] < to1) wx_array[iw] = to1;
        }

    auto start_chrono_withDataMovement = std::chrono::high_resolution_clock::now();

    double achtemp_re0 = 0.00, achtemp_re1 = 0.00, achtemp_re2 = 0.00, \
        achtemp_im0 = 0.00, achtemp_im1 = 0.00, achtemp_im2 = 0.00;

#pragma omp target enter data map(alloc: acht_n1_loc[0:number_bands], aqsmtemp[0:number_bands*ncouls],aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wtilde_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1], achtemp_re[nstart:nend], achtemp_im[nstart:nend])

#pragma omp target update to(aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1], wtilde_array[0:ngpown*ncouls], asxtemp[nstart:nend], achtemp_re[nstart:nend], achtemp[nstart:nend])

    auto start_chrono = std::chrono::high_resolution_clock::now();

#pragma omp target teams distribute parallel for collapse(2) shared(vcoul, aqsntemp, aqsmtemp, I_eps_array) firstprivate(achstemp) map(to:wx_array[nstart:nend], aqsmtemp[0:number_bands*ncouls],aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], wtilde_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], indinv[0:ncouls+1])\
    map(tofrom:acht_n1_loc[0:number_bands], achtemp_re[nstart:nend], achtemp_im[nstart:nend], achtemp_re0, achtemp_re1, achtemp_re2, achtemp_im0, achtemp_im1, achtemp_im2) \
    reduction(+:achtemp_re0, achtemp_re1, achtemp_re2, achtemp_im0, achtemp_im1, achtemp_im2)
    for(int n1 = 0; n1<number_bands; ++n1) 
    {
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            GPUComplex wdiff, delw;

            double * achtemp_re_loc = new double[3];
            double * achtemp_im_loc = new double[3];
            for(int iw = 0; iw < 3; ++iw)
            { achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

#pragma simd
            for(int ig = 0; ig<ncouls; ++ig)
            {
                int iw = 0;
                wdiff = doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]);
                delw = GPUComplex_mult(GPUComplex_product(wtilde_array[my_igp*ncouls+ig] , GPUComplex_conj(wdiff)), 1/GPUComplex_real(GPUComplex_product(wdiff, GPUComplex_conj(wdiff)))); 
                GPUComplex sch_array = GPUComplex_mult(GPUComplex_product(GPUComplex_product(GPUComplex_conj(aqsmtemp[n1*ncouls+igp]), aqsntemp[n1*ncouls+ig]), GPUComplex_product(delw , I_eps_array[my_igp*ncouls+ig])), 0.5*vcoul[igp]);
                achtemp_re_loc[iw] += GPUComplex_real(sch_array);
                achtemp_im_loc[iw] += GPUComplex_imag(sch_array);

                iw++;
                wdiff = doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]);
                delw = GPUComplex_mult(GPUComplex_product(wtilde_array[my_igp*ncouls+ig] , GPUComplex_conj(wdiff)), 1/GPUComplex_real(GPUComplex_product(wdiff, GPUComplex_conj(wdiff)))); 
                sch_array = GPUComplex_mult(GPUComplex_product(GPUComplex_product(GPUComplex_conj(aqsmtemp[n1*ncouls+igp]), aqsntemp[n1*ncouls+ig]), GPUComplex_product(delw , I_eps_array[my_igp*ncouls+ig])), 0.5*vcoul[igp]);
                achtemp_re_loc[iw] += GPUComplex_real(sch_array);
                achtemp_im_loc[iw] += GPUComplex_imag(sch_array);

                iw++;
                wdiff = doubleMinusGPUComplex(wx_array[iw] , wtilde_array[my_igp*ncouls+ig]);
                delw = GPUComplex_mult(GPUComplex_product(wtilde_array[my_igp*ncouls+ig] , GPUComplex_conj(wdiff)), 1/GPUComplex_real(GPUComplex_product(wdiff, GPUComplex_conj(wdiff)))); 
                sch_array = GPUComplex_mult(GPUComplex_product(GPUComplex_product(GPUComplex_conj(aqsmtemp[n1*ncouls+igp]), aqsntemp[n1*ncouls+ig]), GPUComplex_product(delw , I_eps_array[my_igp*ncouls+ig])), 0.5*vcoul[igp]);
                achtemp_re_loc[iw] += GPUComplex_real(sch_array);
                achtemp_im_loc[iw] += GPUComplex_imag(sch_array);

            }

            achtemp_re0 += achtemp_re_loc[0];
            achtemp_im0 += achtemp_im_loc[0];
            achtemp_re1 += achtemp_re_loc[1];
            achtemp_im1 += achtemp_im_loc[1];
            achtemp_re2 += achtemp_re_loc[2];
            achtemp_im2 += achtemp_im_loc[2];

        } //ngpown
    } // number-bands

    std::chrono::duration<double> elapsed_chrono = std::chrono::high_resolution_clock::now() - start_chrono;

#pragma omp target update from (acht_n1_loc[0:number_bands])

#pragma omp target exit data map(delete: acht_n1_loc[:0], aqsmtemp[:0],aqsntemp[:0], I_eps_array[:0], wtilde_array[:0], vcoul[:0], inv_igp_index[:0], indinv[:0], asxtemp[:0])

    std::chrono::duration<double> elapsed_chrono_withDataMovement = std::chrono::high_resolution_clock::now() - start_chrono_withDataMovement;

    achtemp_re[0] = achtemp_re0;
    achtemp_re[1] = achtemp_re1;
    achtemp_re[2] = achtemp_re2;
    achtemp_im[0] = achtemp_im0;
    achtemp_im[1] = achtemp_im1;
    achtemp_im[2] = achtemp_im2;

    printf(" \n Final achstemp\n");
    achstemp.print();

    printf("\n Final achtemp\n");

    for(int iw=nstart; iw<nend; ++iw)
    {
        GPUComplex tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
        achtemp[iw].print();
    }

    std::chrono::duration<double> elapsed_totalTime = std::chrono::high_resolution_clock::now() - start_totalTime;
    cout << "********** Kernel Time Taken **********= " << elapsed_chrono.count() << " secs" << endl;
    cout << "********** Kernel+DataMov Time Taken **********= " << elapsed_chrono_withDataMovement.count() << " secs" << endl;
    cout << "********** Total Time Taken **********= " << elapsed_totalTime.count() << " secs" << endl;

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
