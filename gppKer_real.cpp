#include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>

using namespace std;
int debug = 0;


#define CACHE_LINE 32
#define CACHE_ALIGN __declspec(align(CACHE_LINE)) 

void ssxt_scht_solver(double wxt, int igp, int my_igp, int ig, double wtilde, double wtilde2, double Omega2, double matngmatmgp, double matngpmatmg, double mygpvar1, double mygpvar2, double& ssxa, double& scha, double I_eps_array_igp_myIgp)
{
    double expr0 = 0.0;
    double delw2, scha_mult, ssxcutoff;
    double to1 = 1e-6;
    double sexcut = 4.0;
    double gamma = 0.5;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);
    double sch, ssx;

    double wdiff = wxt - wtilde;

    double cden = wdiff;
    double rden = 1/(cden * (cden));
    double delw = wtilde * (cden) * rden;
    double delwr = (delw * (delw));
    double wdiffr = (wdiff * (wdiff));

    if((wdiffr > limittwo) && (delwr < limitone))
    {
        sch = delw * I_eps_array_igp_myIgp;
        cden = pow(wxt,2);
        rden = (cden * (cden));
        rden = 1.00 / rden;
        ssx = Omega2 * (cden) * rden;
    }
    else if (delwr > to1)
    {
        sch = expr0;
        cden = (double) 4.00 * wtilde2 * (delw + (double)0.50);
        rden = (cden * (cden));
        rden = 1.00/rden;
        ssx = -Omega2 * (cden) * rden * delw;
    }
    else
    {
        sch = expr0;
        ssx = expr0;
    }

    ssxcutoff = sexcut*abs(I_eps_array_igp_myIgp);
    if((abs(ssx) > ssxcutoff) && (abs(wxt) < 0.00)) ssx = 0.00;

    ssxa = matngmatmgp*ssx;
    scha = matngmatmgp*sch;
}

void reduce_achstemp(int n1, int* inv_igp_index, int ncouls, double **aqsmtemp, double **aqsntemp, double **I_eps_array, double& achstemp,  int* indinv, int ngpown, double* vcoul)
{
    double to1 = 1e-6;
    int igmax;
    double schstemp = 0;

//Variables to get around the inability to reduce complex variables && avoid critical.
    int numThreads = omp_get_thread_num();
    double achstemp_localArr[numThreads];
    double achstemp_localReal = 0.00, achstemp_localImag = 0.00;

#pragma omp parallel for private(n1, ncouls, ngpown, indinv, inv_igp_index) schedule(dynamic)
    for(int my_igp = 0; my_igp< ngpown; my_igp++)
    {
        int tid = omp_get_thread_num();
        double schs = 0.0;
        double matngmatmgp = 0.0;
        double matngpmatmg = 0.0;
        double halfinvwtilde, delw, ssx, sch, wdiff, cden , eden, mygpvar1, mygpvar2;
        int igmax;
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        if(indigp == ncouls)
            igp = ncouls-1;

        if(!(igp > ncouls || igp < 0)){

            igmax = ncouls;

            double mygpvar1 = aqsmtemp[n1][igp];
            double mygpvar2 = aqsntemp[n1][igp];

            schs = -I_eps_array[my_igp][igp];
            matngmatmgp = aqsntemp[n1][igp] * mygpvar1;

            if(abs(schs) > to1)
                schstemp = schstemp + matngmatmgp * schs;
        }
        else
        {
            for(int ig=1; ig<igmax; ++ig)
                schstemp = schstemp - aqsntemp[n1][igp] * I_eps_array[my_igp][ig] * mygpvar1;
        }
        achstemp_localArr[tid] += schstemp * vcoul[igp] *(double) 0.5;
    }

#pragma omp parallel for reduction(+:achstemp_localReal, achstemp_localImag)
    for(int i = 0; i < numThreads; i++)
    {
        achstemp_localImag += (achstemp_localArr[i]);
        achstemp_localReal += (achstemp_localArr[i]);
    }
}

void flagOCC_solver(double wxt, double **wtilde_array, int my_igp, int n1, double **aqsmtemp, double **aqsntemp, double **I_eps_array, double &ssxt, double &scht, int igmax, int ncouls, int igp)
{
    double matngmatmgp = 0.0;
    double matngpmatmg = 0.0;
    double ssxa[ncouls], scha[ncouls];
    for(int ig=0; ig<igmax; ++ig)
    {
        double wtilde = wtilde_array[my_igp][ig];
        double wtilde2 = std::pow(wtilde,2);
        double Omega2 = wtilde2*I_eps_array[my_igp][ig];
        double mygpvar1 = (aqsmtemp[n1][igp]);
        double mygpvar2 = aqsmtemp[n1][igp];
        double matngmatmgp = aqsntemp[n1][ig] * mygpvar1;
        if(ig != igp) matngpmatmg = (aqsmtemp[n1][ig]) * mygpvar2;

        ssxt_scht_solver(wxt, igp, my_igp, ig, wtilde, wtilde2, Omega2, matngmatmgp, matngpmatmg, mygpvar1, mygpvar2, ssxa[ig], scha[ig], I_eps_array[my_igp][ig]); 
        ssxt += ssxa[ig];
        scht += scha[ig];
    }
}

void noflagOCC_solver(int igbeg, int igend, int igblk, double wxt, double **wtilde_array, int my_igp, int n1, double **aqsmtemp, double **aqsntemp, double **i_eps_array, double &ssxt, double &scht, int igmax, int ncouls, int igp)
{
    double scha[ncouls]/*, sch, delw, wdiff, cden*/;
    double to1 = 1e-6;
    double sexcut = 4.0;
    double gamma = 0.5;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);
    double mygpvar1 = (aqsmtemp[n1][igp]);
    double rden, cden, wdiff, delw; 
    double delwr, wdiffr; //rden
    bool test[ncouls]; 
    
    for(int ig = igbeg; ig<min(igend,igmax); ++ig)
    {
        wdiff = wxt - wtilde_array[my_igp][ig];
        rden = 1/(wdiff * (wdiff));
        delw = wtilde_array[my_igp][ig] * (wdiff) * rden ; //*rden
        scha[ig] = mygpvar1 * aqsntemp[n1][ig] * delw * i_eps_array[my_igp][ig];
//        test[ig] = ((delwr < limitone) && (wdiffr > limittwo));
    }
    for(int ig = igbeg; ig<min(igend,igmax); ++ig)
        scht += scha[ig];
}

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
#pragma omp parallel private(tid) 
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;

    double to1 = 1e-6;

    double gamma = 0.5;
    double sexcut = 4.0;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);

    double e_n1kq= 6.0; //This in the fortran code is derived through the double dimenrsion array ekq whose 2nd dimension is 1 and all the elements in the array have the same value
//    MyAllocator<64> alloc;


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

    double expr0 = 0.0;
    double expr = 0.5;
    double asxtemp[nend-nstart];

    double acht_n1_loc[number_bands];
    double **acht_n1_loc_threadArr;
    {
        acht_n1_loc_threadArr = new double *[numThreads];
        for(int i=0; i<numThreads; i++)
            acht_n1_loc_threadArr[i] = new double[number_bands] ;
    }

    double achtemp[nend-nstart];
    double **achtemp_threadArr;
    {
        achtemp_threadArr = new double *[numThreads];
        for(int i=0; i<numThreads; i++)
            achtemp_threadArr[i] = new double[nend-nstart] ;
    }

    double **aqsmtemp;
    {
        aqsmtemp = new double *[number_bands];
        for(int i=0; i<number_bands; i++)
            aqsmtemp[i] = new double[ncouls] ;
    }

    double **aqsntemp;
    {
        aqsntemp = new double *[number_bands];
        for(int i=0; i<number_bands; i++)
            aqsntemp[i] = new double[ncouls];
    }

    double **I_eps_array;
    {
        I_eps_array = new CACHE_ALIGN double *[ngpown];
        for(int i=0; i<ngpown; i++)
            I_eps_array[i] = new CACHE_ALIGN double[ncouls];
    }

    double **wtilde_array;
    {
        wtilde_array = new CACHE_ALIGN double *[ngpown];
        for(int i=0; i<ngpown; i++)
            wtilde_array[i] = new CACHE_ALIGN double[ncouls];
    }

    double vcoul[ncouls];
    double wx_array[3];

    double achstemp = 0.0;
    double ssx_array[3], \
        sch_array[3], \
        ssxa[ncouls], \
        scha[ncouls], \
        scht, ssxt, wtilde;

    double wxt;
    double occ=1.0;
    bool flag_occ;

    double start_time = omp_get_wtime(); //Start timing here.

    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;

   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsntemp[i][j] = expr;
           aqsmtemp[i][j] = expr;
       }


   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i][j] = expr;
           wtilde_array[i][j] = expr;
       }

   for(int i=0; i<ncouls; i++)
       vcoul[i] = 1.0;


    for(int ig=0, tmp=1; ig < ngpown; ++ig,tmp++)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0, tmp=1; ig<ncouls; ++ig,tmp++)
        indinv[ig] = ig;

    for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
    {
        flag_occ = n1 < nvband;


        reduce_achstemp(n1, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, indinv, ngpown, vcoul);

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(abs(wx_array[iw]) < to1) wx_array[iw] = to1;
        }

#pragma omp parallel for shared(wtilde_array, aqsntemp, aqsmtemp, I_eps_array, scha,wx_array)  firstprivate(ssx_array, sch_array, \
        scht, ssxt, wxt) schedule(dynamic) \
        private(wtilde, tid)
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            tid = omp_get_thread_num();
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            if(indigp == ncouls)
                igp = ncouls-1;
            int igmax;

            if(!(igp > ncouls || igp < 0)) {
                igmax = ncouls;

            for(int i=0; i<3; i++)
            {
                ssx_array[i] = expr0;
                sch_array[i] = expr0;
            }

            if(flag_occ)
            {
                for(int iw=nstart; iw<nend; iw++)
                {
                    scht = ssxt = expr0;
                    wxt = wx_array[iw];
                    flagOCC_solver(wxt, wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, igmax, ncouls, igp);

                    ssx_array[iw] += ssxt;
                    sch_array[iw] +=(double) 0.5*scht;
                }
            }
            else
            {
                int igblk = 512;
                double scha[ncouls]/*, sch, delw, wdiff, cden*/;
                double to1 = 1e-6;
                double limitone = 1.0/(to1*4.0);
                double limittwo = pow(0.5,2);
                double mygpvar1 = (aqsmtemp[n1][igp]);
                double rden, wdiff, delw; 
                double delwr, wdiffr; //rden

                for(int igbeg=0; igbeg<igmax; igbeg+=igblk)
                {
                    int igend = min(igbeg+igblk-1, igmax);
                    for(int iw=nstart; iw<nend; ++iw)
                    {
                        scht = ssxt = expr0;
                        wxt = wx_array[iw];

                        for(int ig = igbeg; ig<min(igend,igmax); ++ig)
                        {
                            wdiff = wxt - wtilde_array[my_igp][ig];
                            rden = 1/(wdiff * wdiff);
                            delw = wtilde_array[my_igp][ig] * (wdiff) * rden ; //*rden
                            delwr = delw*(delw);
                            wdiffr = wdiff*wdiff;

//                            if ((wdiffr > limittwo) &&  (delwr < limitone)) 
                                scha[ig] = mygpvar1 * aqsntemp[n1][ig] * delw * I_eps_array[my_igp][ig];
                                scht += scha[ig];
//                                else
//                                    scha[ig] = 0;

//                            if ((wdiffr > limittwo) &&  (delwr < limitone)) 
//                                scht += scha[ig];
                        }

//                        for(int ig = igbeg; ig<min(igend,igmax); ++ig)
//                            scht += scha[ig];

                        sch_array[iw] +=(double) 0.5*scht;
                    }
                }
            }

            if(flag_occ)
            {
                for(int iw=nstart; iw<nend; ++iw)
                {
//#pragma omp critical
                    asxtemp[iw] += ssx_array[iw] * occ * vcoul[igp]; 
                }
            }

            for(int iw=nstart; iw<nend; ++iw)
                achtemp_threadArr[tid][iw] += sch_array[iw] * vcoul[igp];

            acht_n1_loc_threadArr[tid][n1] += sch_array[2] * vcoul[igp];

            } //for the if-loop to avoid break inside an openmp pragma statment
        } //ngpown
    } // number-bands


#pragma omp simd
    for(int iw=nstart; iw<nend; ++iw)
        for(int i = 0; i < numThreads; i++)
            achtemp[iw] += achtemp_threadArr[i][iw];

#pragma omp simd
    for(int n1 = 0; n1<number_bands; ++n1) 
        for(int i = 0; i < numThreads; i++)
            acht_n1_loc[n1] += acht_n1_loc_threadArr[i][n1];

    double end_time = omp_get_wtime(); //End timing here

    for(int iw=nstart; iw<nend; ++iw)
        cout << "achtemp[" << iw << "] = " << std::setprecision(15) << achtemp[iw] << endl;

//    cout << "********** noflagOCC_timing =  **********= " << noflagOCC_totalTime << " secs" << endl;
    cout << "********** Time Taken **********= " << end_time - start_time << " secs" << endl;

    return 0;
}

//Almost done code
