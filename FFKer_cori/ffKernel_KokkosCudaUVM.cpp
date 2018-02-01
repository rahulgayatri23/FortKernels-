#include <iostream>
#include <chrono>
#include <complex>
#include <omp.h>
#include "ffKerKokkos.h"

using namespace std;

int main(int argc, char** argv)
{

    Kokkos::initialize(argc, argv);
    {
    if(argc != 7)
    {
        cout << "Incorrect Parameters!!! The correct form is " << endl;
        cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << endl;
        exit(0);
    }

    auto startTimer = std::chrono::high_resolution_clock::now();
    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int ngpown = atoi(argv[4]);
    int nFreq = atoi(argv[5]);
    int nfreqeval = atoi(argv[6]);

    if(ngpown > ncouls)
    {
        cout << "Incorrect Parameters!!! ngpown cannot be greater than ncouls. The correct form is " << endl;
        cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << endl;
        exit(0);
    }

    //OpenMP variables
    int tid, numThreads;

    cout << "Number of Threads = " << numThreads << \
        "\n number_bands = " << number_bands << \
        "\n nvband = " << nvband << \
        "\n ncouls = " << ncouls << \
        "\n ngpown = " << ngpown << \
        "\n nFreq = " << nFreq << \
        "\n nfreqeval = " << nfreqeval << endl;

    GPUComplex expr0( 0.0 , 0.0);
    GPUComplex expr( 0.5 , 0.5);
    GPUComplex expR( 0.5 , 0.5);
    GPUComplex expA( 0.5 , -0.5);
    GPUComplex exprP1( 0.5 , 0.1);
    double pref_zb = 0.5 / 3.14;

//Start to allocate the data structures;
    ViewVectorTypeInt inv_igp_index("inv_igp_index", number_bands * ncouls);
    ViewVectorTypeInt indinv("indinv", number_bands * ncouls);
    ViewVectorTypeDouble vcoul("vcoul", number_bands * ncouls);
    ViewVectorTypeDouble ekq("ekq", number_bands * ncouls);
    ViewVectorTypeDouble dFreqGrid("dFreqGrid", number_bands * ncouls);
    ViewVectorTypeDouble pref("pref", nFreq);
    long double mem_alloc = 0.00;

    ViewVectorTypeComplex aqsntemp("aqsntemp", number_bands * ncouls);
    mem_alloc += (number_bands * ncouls * sizeof(GPUComplex));

    ViewVectorTypeComplex aqsmtemp("aqsmtemp", number_bands * ncouls);
    mem_alloc += (number_bands * ncouls * sizeof(GPUComplex));

    ViewVectorTypeComplex I_epsR_array("I_epsR_array", nFreq * ngpown * ncouls);
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(GPUComplex));

    ViewVectorTypeComplex I_epsA_array("I_epsA_array", nFreq * ngpown * ncouls);
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(GPUComplex));

    ViewVectorTypeComplex ssxDi("ssxDi", nfreqeval);
    ViewVectorTypeComplex schDi("schDi", nfreqeval);
    ViewVectorTypeComplex sch2Di("sch2Di", nfreqeval);
    ViewVectorTypeComplex schDi_cor("schDi_cor", nfreqeval);
    ViewVectorTypeComplex schDi_corb("schDi_corb", nfreqeval);
    ViewVectorTypeComplex achDtemp("achDtemp", nfreqeval);
    ViewVectorTypeComplex ach2Dtemp("ach2Dtemp", nfreqeval);
    ViewVectorTypeComplex achDtemp_cor("achDtemp_cor", nfreqeval);
    ViewVectorTypeComplex achDtemp_corb("achDtemp_corb", nfreqeval);
    ViewVectorTypeComplex asxDtemp("asxDtemp", nfreqeval);
    ViewVectorTypeComplex dFreqBrd("dFreqBrd", nFreq);

    mem_alloc += (nfreqeval * 10 * sizeof(GPUComplex));
    mem_alloc += (nFreq * sizeof(GPUComplex)) ;

    ViewVectorTypeComplex schDt_matrix("schDt_matrix", number_bands * nFreq);
    mem_alloc += (nFreq * number_bands * sizeof(GPUComplex));

    //Variables used : 
//    GPUComplex achsDtemp ;
    ViewScalarTypeComplex achsDtemp ("achsDtemp");
    GPUComplex *achsDtemp_threadArr = new GPUComplex[numThreads];
    for(int i = 0; i < numThreads; ++i)
    {
        achsDtemp_threadArr[i] = expr0;
    }


    double freqevalmin = 0.00;
    double freqevalstep = 0.50;
    double dw = -10;

    //Initialize the data structures
    for(int ig = 0; ig < ngpown; ++ig)
        inv_igp_index(ig) = ig;

    for(int ig = 0; ig < ncouls; ++ig)
        indinv(ig) = ig;

    for(int i=0; i<number_bands; ++i)
    {
        ekq(i) = dw;
        dw += 1.00;

        for(int j=0; j<ncouls; ++j)
        {
            aqsmtemp(i*ncouls+j) = expr;
            aqsntemp(i*ncouls+j) = expr;
        }

        for(int j=0; j<nFreq; ++j)
            schDt_matrix(i*nFreq + j) = expr0;
    }

    for(int i=0; i<ncouls; ++i)
        vcoul(i) = 1.00;

    for(int i=0; i<nFreq; ++i)
    {
        for(int j=0; j<ngpown; ++j)
        {
            for(int k=0; k<ncouls; ++k)
            {
                I_epsR_array(i*ngpown*ncouls + j * ncouls + k) = expR;
                I_epsA_array(i*ngpown*ncouls + j * ncouls + k) = expA;
            }
        }
    }

    dw = 0.00;
    for(int ijk = 0; ijk < nFreq; ++ijk)
    {
        dFreqBrd(ijk) = exprP1;
        dFreqGrid(ijk) = dw;
        dw += 2.00;
    }

    for(int ifreq = 0; ifreq < nFreq; ++ifreq)
    {
        if(ifreq < nFreq-1)
            pref(ifreq) = (dFreqGrid(ifreq+1) - dFreqGrid(ifreq)) / 3.14;
            else
                pref(ifreq) = pref(ifreq-1);

    }
    pref(0) *= 0.5; pref(nFreq-1) *= 0.5;

    for(int i = 0; i < nfreqeval; ++i)
    {
        ssxDi(i) = expr0;
        schDi(i) = expr0;
        sch2Di(i) = expr0;
        schDi_corb(i) = expr0;
        schDi_cor(i) = expr0;
        asxDtemp(i) = expr0;
        achDtemp(i) = expr0;
        ach2Dtemp(i) = expr0;
        achDtemp_cor(i) = expr0;
        achDtemp_corb(i) = expr0;
    }
//    GPUComplex *ssxDittt = new GPUComplex[numThreads];
    ViewScalarTypeComplex ssxDittt ("ssxDittt");

    cout << "Memory Used = " << mem_alloc/(1024 * 1024 * 1024) << " GB" << endl;
    std::chrono::duration<double> elapsedTime_preloop = std::chrono::high_resolution_clock::now() - startTimer;

    cout << "starting loop" << endl;
    auto startTimer_firstloop = std::chrono::high_resolution_clock::now();
    auto startTimer_kernel = std::chrono::high_resolution_clock::now();
    GPUComplStruct achsDtemp_struct1;

    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        bool flag_occ = n1 < nvband;
        double occ = 1.00;

        GPUComplStruct achsDtemp_struct2;
        Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, GPUComplStruct & achsDtemp_structUpdate)
        {
            int indigp = inv_igp_index(my_igp);
            int igp = indinv(indigp);

            GPUComplex schsDtemp = expr0;

            for(int ig = 0; ig < ncouls; ++ig)
                schsDtemp = GPUComplex_minus(schsDtemp , GPUComplex_product(GPUComplex_product(aqsntemp(n1*ncouls + ig) , GPUComplex_conj(aqsmtemp(n1*ncouls + igp))) , I_epsR_array(1*ngpown*ncouls + my_igp*ncouls + ig)));

             
            achsDtemp_structUpdate.re += GPUComplex_real(GPUComplex_mult(schsDtemp , vcoul(igp) * 0.5));
            achsDtemp_structUpdate.im += GPUComplex_imag(GPUComplex_mult(schsDtemp , vcoul(igp) * 0.5));
        }, achsDtemp_struct2);

        achsDtemp_struct1.re += achsDtemp_struct2.re;
        achsDtemp_struct1.im += achsDtemp_struct2.im;
    } //n1

    GPUComplex tmp(achsDtemp_struct1.re, achsDtemp_struct1.im);
    achsDtemp() = tmp;

    for(int n1 = 0; n1 < nvband; ++n1)
    {
        double occ = 1.00;
        GPUComplex ssxDit = expr0;
        GPUComplStruct ssxDittt_agg ;

        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            double wx = freqevalmin - ekq(n1) + freqevalstep;
            ssxDi(iw) = expr0;
            GPUComplex ssxDittt_tmp = expr0;

            int ifreq = 0;
            if(wx > 0.00)
            {
                for(int ijk = 0; ijk < nFreq-1; ++ijk)
                {
                    if(wx > dFreqGrid(ijk) && wx < dFreqGrid(ijk+1))
                    ifreq = ijk;
                }
            }
            else
            {
                int ifreq = 0;
                for(int ijk = 0; ijk < nFreq-1; ++ijk)
                {
                    if(-wx > dFreqGrid(ijk) && -wx < dFreqGrid(ijk+1))
                        ifreq = ijk;
                }
            }
            if(ifreq == 0) ifreq = nFreq-2;

            if(wx > 0.00)
            {
                double fact1 = (dFreqGrid(ifreq+1) - wx) / (dFreqGrid(ifreq+1) - dFreqGrid(ifreq));
                double fact2 = (wx - dFreqGrid(ifreq)) / (dFreqGrid(ifreq+1) - dFreqGrid(ifreq));

                Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, GPUComplStruct& ssxDittt_aggUpdate)
                {
                    int indigp = inv_igp_index(my_igp);
                    int igp = indinv(indigp);
                    int igmax = ncouls;
                    GPUComplex ssxDitt = expr0;

                    for(int ig = 0; ig < igmax; ++ig)
                    {
                        GPUComplex ssxDit = GPUComplex_plus(GPUComplex_mult(I_epsR_array(ifreq*ngpown*ncouls + my_igp*ncouls + ig) , fact1 ) , GPUComplex_mult(I_epsR_array((ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig) , fact2));
    
                        ssxDitt += GPUComplex_product(aqsntemp(n1*ncouls + ig) , GPUComplex_product(GPUComplex_conj(aqsmtemp(n1*ncouls + igp)) , ssxDit));
                    }
                    ssxDittt_aggUpdate.re += GPUComplex_real(GPUComplex_mult(ssxDitt , vcoul(igp)));
                    ssxDittt_aggUpdate.im += GPUComplex_imag(GPUComplex_mult(ssxDitt , vcoul(igp)));
                }, ssxDittt_agg);

            }
            else
            {
                double fact1 = (dFreqGrid(ifreq+1) + wx) / (dFreqGrid(ifreq+1) - dFreqGrid(ifreq));
                double fact2 = (-dFreqGrid(ifreq) - wx) / (dFreqGrid(ifreq+1) - dFreqGrid(ifreq));
                ssxDittt_tmp = expr0;

                Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, GPUComplStruct& ssxDittt_aggUpdate)
                {
                    int indigp = inv_igp_index(my_igp);
                    int igp = indinv(indigp);
                    int igmax = ncouls;
                    GPUComplex ssxDitt = expr0;

                    for(int ig = 0; ig < igmax; ++ig)
                    {
                        GPUComplex ssxDit = GPUComplex_plus(GPUComplex_mult(I_epsA_array(ifreq*ngpown*ncouls + my_igp*ncouls + ig) , fact1 ) , \
                                                     GPUComplex_mult(I_epsA_array((ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig) , fact2));
    
                        ssxDitt += GPUComplex_product(aqsntemp(n1*ncouls + ig) , GPUComplex_product(GPUComplex_conj(aqsmtemp(n1*ncouls + igp)) , ssxDit));
                    }
                    ssxDittt_aggUpdate.re += GPUComplex_real(GPUComplex_mult(ssxDitt , vcoul(igp)));
                    ssxDittt_aggUpdate.im += GPUComplex_imag(GPUComplex_mult(ssxDitt , vcoul(igp)));
                }, ssxDittt_agg);
            }

            GPUComplex tmp(ssxDittt_agg.re, ssxDittt_agg.im);
            ssxDi(iw) = tmp;

            asxDtemp(iw) += GPUComplex_mult(ssxDi(iw) , occ);
        } // iw
    }

    std::chrono::duration<double> elapsedTime_firstloop = std::chrono::high_resolution_clock::now() - startTimer_firstloop;

//    /******************************Done with the First Part of the Code*****************************************************************************/

    auto startTimer_secondloop = std::chrono::high_resolution_clock::now();

    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        bool flag_occ = n1 < nvband;
        double occ = 1.00;
        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            schDi(iw) = expr0;
            sch2Di(iw) = expr0;
            schDi_corb(iw) = expr0;
            schDi_cor(iw) = expr0;
        }

        Kokkos::parallel_for(nFreq, KOKKOS_LAMBDA (int ifreq)
        {
            GPUComplex schDt = schDt_matrix(n1*nFreq + ifreq);
            double cedifft_zb = dFreqGrid(ifreq);
            double cedifft_zb_right, cedifft_zb_left;
            GPUComplex schDt_right, schDt_left, schDt_avg, schDt_lin, schDt_lin2, schDt_lin3;
            GPUComplex cedifft_compl(cedifft_zb, 0.00);
            GPUComplex cedifft_cor;
            GPUComplex cedifft_coh = GPUComplex_minus(cedifft_compl , dFreqBrd(ifreq));

            if(flag_occ)
                cedifft_cor = GPUComplex_minus(GPUComplex_mult(cedifft_compl, -1) , dFreqBrd(ifreq));
                else
                    cedifft_cor = GPUComplex_minus(cedifft_compl , dFreqBrd(ifreq));

            if(ifreq != 0)
            {
                cedifft_zb_right = cedifft_zb;
                cedifft_zb_left = dFreqGrid(ifreq-1);
                schDt_right = schDt;
                schDt_left = schDt_matrix(n1*nFreq + ifreq-1);
                schDt_avg = GPUComplex_mult((GPUComplex_plus(schDt_right , schDt_left)) , 0.5);
                schDt_lin = GPUComplex_minus(schDt_right , schDt_left);
                schDt_lin2 = GPUComplex_divide(schDt_lin , (cedifft_zb_right - cedifft_zb_left));
            }

            if(ifreq != nFreq)
            {
                for(int iw = 0; iw < nfreqeval; ++iw)
                {
                    double wx = freqevalmin - ekq(n1) + (iw-1) * freqevalstep;
                    GPUComplex tmp(0.00, pref(ifreq));
                    schDi(iw) = GPUComplex_minus(schDi(iw) , GPUComplex_divide(GPUComplex_product(tmp,schDt) , doubleMinusGPUComplex(wx, cedifft_coh)));
                }
            }

            if(ifreq != 0)
            {
                for(int iw = 0; iw < nfreqeval; ++iw)
                {
                    double intfact = (freqevalmin - ekq(n1) + (iw-1) * freqevalstep - cedifft_zb_right) / (freqevalmin - ekq(n1) + (iw-1) * freqevalstep - cedifft_zb_left);
                    if(intfact < 0.0001) intfact = 0.0001;
                    if(intfact > 10000) intfact = 10000;
                    intfact = -log(intfact);
                    GPUComplex pref_zb_compl(0.00, pref_zb);
                    sch2Di(iw) = GPUComplex_minus(sch2Di(iw) , GPUComplex_mult(GPUComplex_product(pref_zb_compl , schDt_avg) , intfact));
                    if(flag_occ)
                    {
                        intfact = abs((freqevalmin - ekq(n1) + (iw-1)*freqevalstep + cedifft_zb_right) / (freqevalmin - ekq(n1) + (iw-1)*freqevalstep + cedifft_zb_left));
                        if(intfact < 0.0001) intfact = 0.0001;
                        if(intfact > 10000) intfact = 10000;
                        intfact = log(intfact);
                        schDt_lin3 = GPUComplex_mult((GPUComplex_plus(schDt_left , schDt_lin2)) , (-freqevalmin - ekq(n1) + (iw-1)*freqevalstep - cedifft_zb_left)*intfact) ;
                    }
                    else
                        schDt_lin3 = GPUComplex_mult((GPUComplex_plus(schDt_left , schDt_lin2)) , (freqevalmin - ekq(n1) + (iw-1)*freqevalstep - cedifft_zb_left)*intfact);

                    schDt_lin3 += schDt_lin;
                    schDi_cor(iw) = GPUComplex_minus(schDi_cor(iw) ,  GPUComplex_product(pref_zb_compl , schDt_lin3));
                }
            }
        });

        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            double wx = freqevalmin - ekq(n1) + freqevalstep;
            GPUComplex schDttt_cor = expr0;
            GPUComplStruct schDttt_corStruct;

            if(wx > 0)
            {
                int ifreq = -1;
                for(int ijk = 0; ijk < nFreq-1; ++ijk)
                {
                    if(wx > dFreqGrid(ijk) && wx < dFreqGrid(ijk+1))
                        ifreq = ijk;
                }
                if(ifreq == -1) ifreq = nFreq-2;

                double fact1 = -0.5 * (dFreqGrid(ifreq+1) - wx) / (dFreqGrid(ifreq+1) - dFreqGrid(ifreq)); 
                double fact2 = -0.5 * (wx - dFreqGrid(ifreq)) / (dFreqGrid(ifreq+1) - dFreqGrid(ifreq)); 

                GPUComplStruct schDttt_corStruct;
                ViewScalarTypeComplex schDttt("schDttt");

                Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, GPUComplStruct& schDtttStructUpdate)
                {
                    int indigp = inv_igp_index(my_igp) ;
                    int igp = indinv(indigp);
                    int igmax = ncouls;
                    GPUComplex sch2Dtt(0.00, 0.00);

                    for(int ig = 0; ig < igmax; ++ig)
                    {
                        GPUComplex sch2Dt = GPUComplex_plus(GPUComplex_mult((GPUComplex_minus(I_epsR_array(ifreq*ngpown*ncouls + my_igp*ncouls + ig), I_epsA_array(ifreq*ngpown*ncouls + my_igp*ncouls + ig))) , fact1) , \
                                                    GPUComplex_mult((GPUComplex_minus(I_epsR_array((ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig), I_epsA_array((ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig))) , fact2));
                        sch2Dtt += GPUComplex_product(GPUComplex_product(aqsntemp(n1*ncouls + ig) , GPUComplex_conj(aqsmtemp(n1*ncouls + igp))) , sch2Dt);
                    }

                    schDttt() += GPUComplex_mult(sch2Dtt , vcoul(igp));

                    if(flag_occ){}
                    else
                    {
                        schDtttStructUpdate.re = GPUComplex_real(GPUComplex_mult(sch2Dtt , vcoul(igp)));
                        schDtttStructUpdate.im = GPUComplex_imag(GPUComplex_mult(sch2Dtt , vcoul(igp)));
                    }
                }, schDttt_corStruct);

                sch2Di(iw) += schDttt();

                GPUComplex tmp(schDttt_corStruct.re, schDttt_corStruct.im);
                schDttt_cor += tmp;
            }
            else if(flag_occ)
            {
                wx = -wx; int ifreq = 0;
                GPUComplStruct schDttt_corStruct;
                for(int ijk = 0; ijk < nFreq-1; ++ijk)
                {
                    if(wx > dFreqGrid(ijk) && wx < dFreqGrid(ijk+1))
                        ifreq = ijk;
                }
                if(ifreq == 0) ifreq = nFreq-2;

                double fact1 = (dFreqGrid(ifreq+1) - wx) / (dFreqGrid(ifreq+1) - dFreqGrid(ifreq)); 
                double fact2 = (wx - dFreqGrid(ifreq)) / (dFreqGrid(ifreq+1) - dFreqGrid(ifreq)); 

                Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, GPUComplStruct& schDttt_corStructUpdate)
                {
                    int indigp = inv_igp_index(my_igp) ;
                    int igp = indinv(indigp);
                    int igmax = ncouls;
                    GPUComplex sch2Dtt(0.00, 0.00);

                    for(int ig = 0; ig < igmax; ++ig)
                    {
                        GPUComplex sch2Dt = GPUComplex_mult(GPUComplex_plus(GPUComplex_mult((GPUComplex_minus(I_epsR_array(ifreq*ngpown*ncouls + my_igp*ncouls + ig), I_epsA_array(ifreq*ncouls*ngpown + my_igp*ncouls + ig))) , fact1) , \
                                                    GPUComplex_mult((GPUComplex_minus(I_epsR_array((ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig), I_epsA_array((ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig))) , fact2)), -0.5);
                        sch2Dtt += GPUComplex_product(GPUComplex_product(aqsntemp(n1*ncouls + ig) , GPUComplex_conj(aqsmtemp(n1*ncouls + igp))) , sch2Dt);
                    }
                    schDttt_corStructUpdate.re += GPUComplex_real(GPUComplex_mult(sch2Dtt , vcoul(igp)));
                    schDttt_corStructUpdate.im += GPUComplex_imag(GPUComplex_mult(sch2Dtt , vcoul(igp)));
                }, schDttt_corStruct);

                GPUComplex tmp(schDttt_corStruct.re, schDttt_corStruct.im);
                schDttt_cor += tmp;
            }


            schDi_cor(iw) += schDttt_cor;

//Summing up at the end of iw loop
            achDtemp(iw) += schDi(iw);
            ach2Dtemp(iw) += sch2Di(iw);
            achDtemp_cor(iw) += schDi_cor(iw);
            achDtemp_corb(iw) += schDi_corb(iw);
        }// iw
    } //n1
    std::chrono::duration<double> elapsedTime_secondloop = std::chrono::high_resolution_clock::now() - startTimer_secondloop;

    cout << "achsDtemp = " ;
    achsDtemp().print();
    cout << "asxDtemp = " ;
    asxDtemp(0).print();
    cout << "achDtemp_cor = " ;
    achDtemp_cor(0).print();

    std::chrono::duration<double> elapsedTime = std::chrono::high_resolution_clock::now() - startTimer;
    std::chrono::duration<double> elapsedKernelTime = std::chrono::high_resolution_clock::now() - startTimer_kernel;
    cout << "********** PreLoop **********= " << elapsedTime_preloop.count() << " secs" << endl;
    cout << "********** Kenel Time **********= " << elapsedKernelTime.count() << " secs" << endl;
    cout << "********** Total Time Taken **********= " << elapsedTime.count() << " secs" << endl;

    }
    Kokkos::finalize();

    return 0;
}
