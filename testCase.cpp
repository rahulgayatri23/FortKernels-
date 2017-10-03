  #include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
#include <chrono>

using namespace std;
int debug = 0;


int main(int argc, char** argv)
{


    int number_bands = 64;
    int ncouls = 64;
    int nodes_per_group = 20;

    int ngpown = ncouls / nodes_per_group; //Number of gvectors per mpi task

    int nstart = 0, nend = 3;

    //OpenMP variables
    int tid, numThreads;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;

    //ALLOCATE statements from fortran gppkernel.

    std::complex<double> expr0( 0.0 , 0.0);
    std::complex<double> expr( 0.5 , 0.5);

    std::complex<double> *achtemp = new std::complex<double>[nend-nstart];
    std::complex<double> *aqsmtemp = new std::complex<double> [number_bands*ncouls];
    std::complex<double> *aqsntemp = new std::complex<double> [number_bands*ncouls];
    std::complex<double> *wtilde_array = new std::complex<double> [ncouls];
    std::complex<double> *I_eps_array = new std::complex<double>[ngpown*ncouls] ;


    double wx_array[3];

    std::complex<double> *ssx_array = new std::complex<double>[3];
    std::complex<double> *sch_array = new std::complex<double>[3];
    std::complex<double> *scha = new std::complex<double>[ncouls];

    double occ=1.0;
    bool flag_occ;
    double achstemp_real = 0.00, achstemp_imag = 0.00;
  for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsmtemp[i*ncouls+j] = expr;
           aqsntemp[i*ncouls+j] = expr;
       }

       for(int j=0; j<ncouls; j++)
       {
           wtilde_array[j] = expr;
       }


    auto start_chrono = std::chrono::high_resolution_clock::now();
#pragma omp target enter data map(alloc:achtemp[0:(nend-nstart)], aqsntemp[0:number_bands*ncouls], wtilde_array[0:ncouls], I_eps_array[0:ngpown*ncouls])

#pragma omp target update to( aqsntemp[0:number_bands*ncouls], wtilde_array[0:ngpown*ncouls], I_eps_array[0:ngpown*ncouls])

#pragma omp target map(to:ssx_array[0:3], sch_array[0:3], wx_array[0:3], scha[0:ncouls]) map(tofrom:achstemp_real, achstemp_imag)
{
//       for(int iw=nstart; iw<nend; ++iw)
//           achtemp[iw] = expr0;

#pragma target teams distribute shared(aqsntemp, ssx_array, I_eps_array)
    for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
    {
       std::complex<double> wdiff, delw;
       std::complex<double> scht, ssxt;
       double wxt, rden;

#pragma omp parallel for
    for(int iw=nstart; iw<nend; ++iw)
    {
        for(int ig = 0; ig<ncouls; ++ig)
        {
            wdiff = wxt - wtilde_array[ig];
            rden = std::real(wdiff * std::conj(wdiff));
            rden = 1/rden;
            delw = wtilde_array[ig] * conj(wdiff) * rden ; //*rden

//             scha[ig] = aqsntemp[n1*ncouls+ig] * delw ;
            scha[ig] = aqsntemp[n1*ncouls+ig] * delw * I_eps_array[ncouls+ig];

        }

        for(int ig = 0; ig<ncouls; ++ig)
         scht += scha[ig];

        sch_array[iw] +=(double) 0.5*scht;
    }

#pragma omp critical
            for(int iw=nstart; iw<nend; ++iw)
                achtemp[iw] += sch_array[iw];

    } // number-bands
} //TARGET
#pragma omp target update from (achtemp[0:(nend-nstart)])
#pragma omp target exit data map(delete: achtemp[:0], aqsntemp[:0], wtilde_array[:0], I_eps_array[:0])

    std::chrono::duration<double> elapsed_chrono = std::chrono::high_resolution_clock::now() - start_chrono;

    for(int iw=nstart; iw<nend; ++iw)
        cout << "achtemp[" << iw << "] = " << std::setprecision(15) << achtemp[iw] << endl;

    cout << "********** Chrono Time Taken **********= " << elapsed_chrono.count() << " secs" << endl;

    free(achtemp);
    free(aqsmtemp);
    free(aqsntemp);
    free(wtilde_array);
    free(ssx_array);

    return 0;
	}
