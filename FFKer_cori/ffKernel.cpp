#include <iostream>
#include <chrono>
#include <complex>
#include <omp.h>

using namespace std;

int main(int argc, char** argv)
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
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }

    cout << "Number of Threads = " << numThreads << \
        "\n number_bands = " << number_bands << \
        "\n nvband = " << nvband << \
        "\n ncouls = " << ncouls << \
        "\n ngpown = " << ngpown << \
        "\n nFreq = " << nFreq << \
        "\n nfreqeval = " << nfreqeval << endl;

    std::complex<double> expr0( 0.0 , 0.0);
    std::complex<double> expr( 0.5 , 0.5);
    std::complex<double> expR( 0.5 , 0.5);
    std::complex<double> expA( 0.5 , -0.5);
    std::complex<double> exprP1( 0.5 , 0.1);
    double pref_zb = 0.5 / 3.14;

//Start to allocate the data structures;
    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls];
    double *vcoul = new double[ncouls];
    double *ekq = new double[number_bands];
    double *dFreqGrid = new double[nFreq];
    double *pref = new double[nFreq];

    std::complex<double> aqsntemp[number_bands][ncouls];
    std::complex<double> aqsmtemp[number_bands][number_bands];
    std::complex<double> I_epsR_array[nFreq][ngpown][ncouls];
    std::complex<double> I_epsA_array[nFreq][ngpown][ncouls];
    std::complex<double> ssxDi[nfreqeval];
    std::complex<double> schDi[nfreqeval];
    std::complex<double> sch2Di[nfreqeval];
    std::complex<double> schDi_cor[nfreqeval];
    std::complex<double> schDi_corb[nfreqeval];
    std::complex<double> achDtemp[nfreqeval];
    std::complex<double> ach2Dtemp[nfreqeval];
    std::complex<double> achDtemp_cor[nfreqeval];
    std::complex<double> achDtemp_corb[nfreqeval];
    std::complex<double> asxDtemp[nfreqeval];
    std::complex<double> dFreqBrd[nFreq];
    std::complex<double> schDt_matrix[number_bands][nFreq];


    //Variables used : 
    std::complex<double> achsDtemp = expr0;
    double freqevalmin = 0.00;
    double freqevalstep = 0.50;
    double dw = -10;

    //Initialize the data structures
    for(int ig = 0; ig < ngpown; ++ig)
        inv_igp_index[ig] = ig;

    for(int ig = 0; ig < ncouls; ++ig)
        indinv[ig] = ig;

    for(int i=0; i<number_bands; ++i)
    {
        ekq[i] = dw;
        dw += 1.00;

        for(int j=0; j<ncouls; ++j)
        {
            aqsmtemp[i][j] = expr;
            aqsntemp[i][j] = expr;
        }

        for(int j=0; j<nFreq; ++j)
            schDt_matrix[i][j] = expr0;
    }

    for(int i=0; i<ncouls; ++i)
        vcoul[i] = 1.00;

    for(int i=0; i<nFreq; ++i)
    {
        for(int j=0; j<ngpown; ++j)
        {
            for(int k=0; k<ncouls; ++k)
            {
                I_epsR_array[i][j][k] = expR;
                I_epsA_array[i][j][k] = expA;
            }
        }
    }

    dw = 0.00;
    for(int ijk = 0; ijk < nFreq; ++ijk)
    {
        dFreqBrd[ijk] = exprP1;
        dFreqGrid[ijk] = dw;
        dw += 2.00;
    }

    for(int ifreq = 0; ifreq < nFreq; ++ifreq)
    {
        if(ifreq < nFreq-1)
            pref[ifreq] = (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]) / 3.14;
            else
                pref[ifreq] = pref[ifreq-1];

    }
    pref[0] *= 0.5; pref[nFreq-1] *= 0.5;

    for(int i = 0; i < nfreqeval; ++i)
    {
        ssxDi[i] = expr0;
        schDi[i] = expr0;
        sch2Di[i] = expr0;
        schDi_corb[i] = expr0;
        schDi_cor[i] = expr0;
        asxDtemp[i] = expr0;
        achDtemp[i] = expr0;
        ach2Dtemp[i] = expr0;
        achDtemp_cor[i] = expr0;
        achDtemp_corb[i] = expr0;
    }

    cout << "starting loop" << endl;
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        bool flag_occ = n1 < nvband;
        double occ = 1.00;

        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            if(igp < ncouls && igp >= 0)
            {
                int igmax = ncouls;
                std::complex<double> schsDtemp = expr0;

                for(int ig = 0; ig < igmax; ++ig)
                    schsDtemp -= aqsntemp[n1][ig] * std::conj(aqsmtemp[n1][igp]) * I_epsR_array[1][my_igp][ig];

                achsDtemp += schsDtemp * vcoul[igp] * 0.5;
            }
        }

        std::complex<double> ssxDit = expr0;
        std::complex<double> ssxDitt = expr0;
        std::complex<double> ssxDittt = expr0;

        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            double wx = freqevalmin - ekq[n1] + freqevalstep;
            ssxDi[iw] = expr0;

            if(flag_occ)
            {
                int ifreq = 0;
                if(wx > 0.00)
                {
                    for(int ijk = 0; ijk < nFreq-1; ++ijk)
                    {
                        if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                        ifreq = ijk;
                    }
                }
                else
                {
                    int ifreq = 0;
                    for(int ijk = 0; ijk < nFreq-1; ++ijk)
                    {
                        if(-wx > dFreqGrid[ijk] && -wx < dFreqGrid[ijk+1])
                            ifreq = ijk;
                    }
                }
                if(ifreq == 0) ifreq = nFreq-1;

                if(wx > 0.00)
                {
                    double fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
                    double fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);

                    ssxDittt = expr0;

                    for(int my_igp = 0; my_igp < ngpown; ++my_igp)
                    {
                        int indigp = inv_igp_index[my_igp];
                        int igp = indinv[indigp];
                        int igmax = ncouls;
                        ssxDitt = expr0;

                        if(igp < ncouls && igp >= 0)
                        {
                            for(int ig = 0; ig < igmax; ++ig)
                            {
                                ssxDit = I_epsR_array[ifreq][my_igp][ig] * fact1 + \
                                                             I_epsR_array[ifreq+1][my_igp][ig] * fact2;
            
                                ssxDitt += aqsntemp[n1][ig] * std::conj(aqsmtemp[n1][igp]) * ssxDit;
                            }
                            ssxDittt += ssxDitt * vcoul[igp];
                        }
                    }
                }
                else
                {
                    double fact1 = (dFreqGrid[ifreq+1] + wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
                    double fact2 = (-dFreqGrid[ifreq] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);

                    ssxDittt = expr0;

                    for(int my_igp = 0; my_igp < ngpown; ++my_igp)
                    {
                        int indigp = inv_igp_index[my_igp];
                        int igp = indinv[indigp];
                        int igmax = ncouls;
                        ssxDitt = expr0;

                        if(igp < ncouls && igp >= 0)
                        {
                            for(int ig = 0; ig < igmax; ++ig)
                            {
                                ssxDit = I_epsR_array[ifreq][my_igp][ig] * fact1 + \
                                                             I_epsR_array[ifreq+1][my_igp][ig] * fact2;
            
                                ssxDitt += aqsntemp[n1][ig] * std::conj(aqsmtemp[n1][igp]) * ssxDit;
                            }
                        }
                    }
                }

                ssxDi[iw] += ssxDittt;
                asxDtemp[iw] += ssxDi[iw] * occ;
            }
        } // iw
    } //n1

//    /******************************Done with the First Part of the Code*****************************************************************************/


    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        bool flag_occ = n1 < nvband;
        double occ = 1.00;
        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            schDi[iw] = expr0;
            sch2Di[iw] = expr0;
            schDi_corb[iw] = expr0;
            schDi_cor[iw] = expr0;
        }

        for(int ifreq = 0; ifreq < nFreq; ++ifreq)
        {
            std::complex<double> schDt = schDt_matrix[n1][ifreq];
            double cedifft_zb = dFreqGrid[ifreq];
            double cedifft_zb_right, cedifft_zb_left;
            std::complex<double> schDt_right, schDt_left, schDt_avg, schDt_lin, schDt_lin2, schDt_lin3;
            std::complex<double> cedifft_compl(cedifft_zb, 0.00);
            std::complex<double> cedifft_cor;
            std::complex<double> cedifft_coh = cedifft_compl - dFreqBrd[ifreq];

            if(flag_occ)
                cedifft_cor = -1 * cedifft_compl - dFreqBrd[ifreq];
                else
                    cedifft_cor = cedifft_compl - dFreqBrd[ifreq];

            if(ifreq != 0)
            {
                cedifft_zb_right = cedifft_zb;
                cedifft_zb_left = dFreqGrid[ifreq-1];
                schDt_right = schDt;
                schDt_left = schDt_matrix[n1][ifreq-1];
                schDt_avg = (schDt_right + schDt_left) * 0.5;
                schDt_lin = schDt_right - schDt_left;
                schDt_lin2 = schDt_lin / (cedifft_zb_right - cedifft_zb_left);
            }

            if(ifreq != nFreq)
            {
                for(int iw = 0; iw < nfreqeval; ++iw)
                {
                    double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep;
                    std::complex<double> tmp(0.00, pref[ifreq]);
                    schDi[iw] -= tmp*schDt / (wx - cedifft_coh);
                    schDi_corb[iw] -= tmp*schDt / (wx-cedifft_cor);
                }
            }

            if(ifreq != 0)
            {
                for(int iw = 0; iw < nfreqeval; ++iw)
                {
                    double intfact = std::abs((freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_left));
                    if(intfact < 0.0001) intfact = 0.0001;
                    if(intfact > 10000) intfact = 10000;
                    intfact = -log(intfact);
                    std::complex<double> pref_zb_compl(0.00, pref_zb);
                    sch2Di[iw] -= pref_zb_compl * schDt_avg * intfact;
                    if(flag_occ)
                    {
                        intfact = abs((freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_left));
                        if(intfact < 0.0001) intfact = 0.0001;
                        if(intfact > 10000) intfact = 10000;
                        intfact = log(intfact);
                        schDt_lin3 = (schDt_left + schDt_lin2 * (-freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)) * intfact;
                    }
                    else
                        schDt_lin3 = (schDt_left + schDt_lin2 * (freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)) * intfact;

                    schDt_lin3 += schDt_lin;
                    schDi_cor[iw] -= pref_zb_compl * schDt_lin3;
                }
            }
        }

        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            double wx = freqevalmin - ekq[n1] + freqevalstep;

            if(wx > 0)
            {
                int ifreq = -1;
                for(int ijk = 0; ijk < nFreq-1; ++ijk)
                {
                    if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                        ifreq = ijk;
                }
                if(ifreq == -1) ifreq = nFreq-2;

                double fact1 = -0.5 * (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
                double fact2 = -0.5 * (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 

                std::complex<double> schDttt = expr0;
                std::complex<double> schDttt_cor = expr0;

                for(int my_igp = 0; my_igp < ngpown; ++my_igp)
                {
                    int indigp = inv_igp_index[my_igp] ;
                    int igp = indinv[indigp];
                    int igmax = ncouls;
                    std::complex<double> sch2Dtt(0.00, 0.00);

                    if(igp < ncouls && igp >= 0)
                    {
                        for(int ig = 0; ig < igmax; ++ig)
                        {
                            std::complex<double> sch2Dt = (I_epsR_array[ifreq][my_igp][ig] - I_epsA_array[ifreq][my_igp][ig]) * fact1 + \
                                                        (I_epsR_array[ifreq+1][my_igp][ig] - I_epsA_array[ifreq+1][my_igp][ig]) * fact2;
                            sch2Dtt += aqsntemp[n1][ig] * std::conj(aqsmtemp[n1][igp]) * sch2Dt;
                        }
                        schDttt += sch2Dtt * vcoul[igp];
                        if(flag_occ){}
                        else
                            schDttt_cor += sch2Dtt * vcoul[igp];
                    }
                }

                sch2Di[iw] += schDttt;
                schDi_cor[iw] += schDttt_cor;
            }
            else if(flag_occ)
            {
                wx = -wx; int ifreq = 0;
                for(int ijk = 0; ijk < nFreq-1; ++ijk)
                {
                    if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                        ifreq = ijk;
                }
                if(ifreq == 0) ifreq = nFreq-1;

                double fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
                double fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 

                std::complex<double> schDttt_cor = expr0;
                for(int my_igp = 0; my_igp < ngpown; ++my_igp)
                {
                    int indigp = inv_igp_index[my_igp] ;
                    int igp = indinv[indigp];
                    int igmax = ncouls;
                    std::complex<double> sch2Dtt(0.00, 0.00);

                    if(igp < ncouls && igp >= 0)
                    {
                        for(int ig = 0; ig < igmax; ++ig)
                        {
                            std::complex<double> sch2Dt = -0.5*((I_epsR_array[ifreq][my_igp][ig] - I_epsA_array[ifreq][my_igp][ig]) * fact1 + \
                                                        (I_epsR_array[ifreq+1][my_igp][ig] - I_epsA_array[ifreq+1][my_igp][ig]) * fact2);
                            sch2Dtt += aqsntemp[n1][ig] * std::conj(aqsmtemp[n1][igp]) * sch2Dt;
                        }
                        schDttt_cor += sch2Dtt * vcoul[igp];
                    }
                }
                schDi_cor[iw] += schDttt_cor;
            }
        }


//Summing up at the end of iw loop
        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            achDtemp[iw] += schDi[iw];
            ach2Dtemp[iw] += sch2Di[iw];
            achDtemp_cor[iw] += schDi_cor[iw];
            achDtemp_corb[iw] += schDi_corb[iw];
        }// iw
    } //n1

    cout << "achsDtemp = " << achsDtemp << endl;
    cout << "asxDtemp = " << asxDtemp[0] << endl;
    cout << "achDtemp_cor = " << achDtemp_cor[0] << endl;

    std::chrono::duration<double> elapsedTime = std::chrono::high_resolution_clock::now() - startTimer;
    cout << "********** Total Time Taken **********= " << elapsedTime.count() << " secs" << endl;

//Free the allocated memory since you are a good programmer :D
    free(inv_igp_index);
    free(indinv);
    free(vcoul);
    free(ekq);
    free(dFreqGrid);
    free(pref);

    return 0;
}
