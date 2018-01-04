#include <iostream>
#include "Complex.h"
#include <chrono>

using namespace std;

int main(int argc, char** argv)
{

    int NTHREADS, TIF, n1, ispin, iw, ifreq, ijk, iwlda, igt, igtt, iglimit, \
        my_igp, indigp, ig, igp, igmax, gppsum;

    GPUComplex achstemp(0.00, 0.00);
    GPUComplex achxtemp(0.00, 0.00);
    GPUComplex matngmatmgp(0.00, 0.00);
    GPUComplex matngpmatmg(0.00, 0.00);
    GPUComplex mygpvar1(0.00, 0.00);
    GPUComplex mygpvar2(0.00, 0.00);
    GPUComplex schstemp(0.00, 0.00);
    GPUComplex schs(0.00, 0.00);
    GPUComplex sch(0.00, 0.00);
    GPUComplex ssx(0.00, 0.00);
    GPUComplex ssxt(0.00, 0.00);
    GPUComplex scht(0.00, 0.00);
    GPUComplex schD(0.00, 0.00);
    GPUComplex achsDtemp(0.00, 0.00);
    GPUComplex schsDtemp(0.00, 0.00);
    GPUComplex ssxDit(0.00, 0.00);
    GPUComplex ssxDitt(0.00, 0.00);
    GPUComplex schDt(0.00, 0.00);
    GPUComplex sch2dt(0.00, 0.00);
    GPUComplex sch2dtt(0.00, 0.00);
    GPUComplex I_epsRggp_int(0.00, 0.00);
    GPUComplex I_epsAggp_int(0.00, 0.00);
    GPUComplex scjDttt(0.00, 0.00);
    GPUComplex scjDttt_cor(0.00, 0.00);
    GPUComplex schDt_avg(0.00, 0.00);
    GPUComplex schDt_right(0.00, 0.00);
    GPUComplex schDt_left(0.00, 0.00);
    GPUComplex schDt_lin(0.00, 0.00);
    GPUComplex schDt_lin2(0.00, 0.00);
    GPUComplex schDt_lin3(0.00, 0.00);


    double cedifft_zb, intfact,cedifft_zb_left, cedifft_zb_right, \
        e_n1kq, \
        tol, fact1, fact2, wx, occ, occfact, \
        starttime, endtime ;

    bool flag_occ;

    auto startTimer = std::chrono::high_resolution_clock::now();

    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int ngpown = atoi(argv[4]);
    int nFreq = atoi(argv[5]);
    int nfreqval = atoi(argv[6]);


    cout << " number_bands = " << number_bands << \
        "\t nvband = " << nvband << \
        "\t ncouls = " << ncouls << \
        "\t ngpown = " << ngpown << \
        "\t nFreq = " << nFreq << \
        "\t nfreqval = " << nfreqval << endl;

    double freqevalmin = 0.00;
    double freqevalstep = 0.5;

    int  ggpsum = 2;
    int e_lk = 1.00;
    double dw = -10.00;
    double pref_zb = 0.5 / 3.14;


    double *vcoul = new double[ncouls];
    double *ekq = new double[number_bands];
    for(int igk = 0; igk < number_bands; ++igk)
    {
        ekq[igk] = dw;
        dw += 1.00;
    }

    GPUComplex exprP(0.5, 0.5);
    GPUComplex exprN(0.5, 0.5);
    GPUComplex expr0(0.0, 0.0);
    GPUComplex exprIm1(0.0, 0.1);

    cout << "Allocating aqsmtemp & aqsntemp, size = " << ncouls*number_bands*2*16/1024 << endl;
    GPUComplex *aqsntemp = new GPUComplex[ncouls*number_bands];
    GPUComplex *aqsmtemp = new GPUComplex[ncouls*number_bands];
    for(int i = 0; i < ncouls*number_bands; ++i)
    {
        aqsntemp[iw] = exprP;
        aqsmtemp[iw] = exprP;
    }

    cout << "Allocating I_epsR_array, size = " << endl;
    GPUComplex *I_epsR_array = new GPUComplex[ncouls * ngpown * nFreq];
    GPUComplex *I_epsA_array = new GPUComplex[ncouls * ngpown * nFreq];
    for(int i = 0; i < ncouls*ngpown*nFreq; ++i)
    {
        I_epsR_array[i] = exprP;
        I_epsA_array[i] = exprN;
    }

    cout << "Allocating I_inv_igp, size = " << endl;
    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls];

    double *dFreqGrid = new double[nFreq];
    dw = 0.00;
    for(int igk = 0; igk < nFreq; ++igk)
    {
        dFreqGrid[igk] = dw;
        dw += 2.00;
    }

    double *pref = new double[nFreq];
    GPUComplex *dFreqBrd = new GPUComplex[nFreq];
    for(int ifreq = 0; ifreq < nFreq; ++ifreq)
    {
        if(ifreq < nFreq)
            pref[ifreq] = (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]) / 3.14;
            else
                pref[ifreq] = pref[ifreq-1];

        dFreqBrd[ifreq] = exprIm1;
    }
    pref[0] *= 0.5;
    pref[nFreq] *= 0.5;

    GPUComplex *asxDtemp = new GPUComplex[nfreqval];
    GPUComplex *achDtemp = new GPUComplex[nfreqval];
    GPUComplex *achDtemp_cor = new GPUComplex[nfreqval];
    GPUComplex *achDtemp_corb = new GPUComplex[nfreqval];
    GPUComplex *ach2Dtemp = new GPUComplex[nfreqval];
    GPUComplex *schDi = new GPUComplex[nfreqval];
    GPUComplex *schDi_cor = new GPUComplex[nfreqval];
    GPUComplex *schDi_corb = new GPUComplex[nfreqval];
    GPUComplex *sch2Di = new GPUComplex[nfreqval];
    GPUComplex *ssxDi = new GPUComplex[nfreqval];
    double *wxi = new double[nfreqval];

    for(int infreq = 0; infreq < nfreqval; ++infreq)
    {
        asxDtemp[infreq] = expr0;
        achDtemp[infreq] = expr0;
        achDtemp_cor[infreq] = expr0;
        achDtemp_corb[infreq] = expr0;
        ach2Dtemp[infreq] = expr0;
        schDi[infreq] = expr0;
        schDi_cor[infreq] = expr0;
        schDi_corb[infreq] = expr0;
        sch2Di[infreq] = expr0;
        ssxDi[infreq] = expr0;
        wxi[infreq] = 0.00;
    }

   GPUComplex *schDt_matrix = new GPUComplex[nFreq * number_bands]; 
   for(int i = 0; i < nFreq*number_bands; ++i)
       schDt_matrix[i] = expr0;

    for(int i = 0; i < ngpown; ++i)
        inv_igp_index[i] = i;

    for(int i = 0; i < ncouls; ++i)
        indinv[i] = i;


    double to1 = 1e-6; 
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);



    printf("Starting Loop\n");


    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        flag_occ = n1 < nvband;

        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            if (igp > ncouls || igp < 0 )
            {
                int igmax = ncouls;
                schsDtemp = expr0;

                for(int ig = 0; ig < igmax; ++ig)
                {
                    schsDtemp -= GPUComplex_product( aqsntemp[n1*number_bands +ig] , GPUComplex_conj(GPUComplex_product(aqsmtemp[n1*number_bands + ig] , I_epsR_array[0*nFreq + my_igp * ngpown + ig])));
                }
                schsDtemp += GPUComplex_mult(schsDtemp , vcoul[igp] * 0.5);
            }
        }

        for(int infreq = 0; infreq < nfreqval; ++infreq)
            ssxDi[infreq] = expr0;

        for(int iw = 0; iw < nfreqval; ++iw)
        {
            double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep ;
            GPUComplex ssxDittt(0.00, 0.00);

            if(flag_occ)
            {
                if(wx > 0.00)
                {
                    ifreq = 0.00;
                    for(int ijk = 0; ijk < nFreq-1; ++ijk)
                    {
                        if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                            ifreq = ijk;
                    }
                    if(ifreq == 0)
                    {
                        ifreq = nFreq + 3;
                    }
                }
                else
                {
                    ifreq = 0.00;
                    for(int ijk = 0; ijk < nFreq-1; ++ijk)
                    {
                        if(-wx > dFreqGrid[ijk] && -wx < dFreqGrid[ijk+1])
                            ifreq = ijk;
                    }
                    if(ifreq == 0)
                    {
                        ifreq = nFreq + 3;
                    }
                }

                if(ifreq > nFreq)
                    ifreq = nFreq - 1;

                if(wx > 0.00)
                {
                    fact1 = (dFreqGrid[ifreq +1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
                    fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);

                    for(int my_igp = 0; my_igp < ngpown; ++my_igp)
                    {
                        int indigp = inv_igp_index[my_igp];
                        int igp = indinv[indigp];

                        if(!(igp > ncouls || igp < 0))
                        {
                            int igmax = ncouls;
                            ssxDitt = expr0;

                            for(int ig = 0; ig < igmax; ++ig)
                            {
                                    ssxDit = GPUComplex_mult(I_epsR_array[ifreq*nFreq + my_igp * ngpown + ig],fact1) + GPUComplex_mult(I_epsR_array[(ifreq+1) * nFreq + my_igp * ngpown +ig],fact2) ;

                                    ssxDitt += GPUComplex_product(aqsntemp[n1*number_bands + ig] , GPUComplex_product(GPUComplex_conj(aqsmtemp[n1 * number_bands + igp]) , ssxDit));
                            }
                        }
                        ssxDittt += GPUComplex_mult(ssxDitt , vcoul[igp]);
                    }
                }
                else
                {
                    fact1 = (dFreqGrid[ifreq +1] + wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
                    fact2 = (- dFreqGrid[ifreq] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);

                    for(int my_igp = 0; my_igp < ngpown; ++my_igp)
                    {
                        int indigp = inv_igp_index[my_igp];
                        int igp = indinv[indigp];

                        if(!(igp > ncouls || igp < 0))
                        {
                            int igmax = ncouls;
                            ssxDitt = expr0;

                            for(int ig = 0; ig < igmax; ++ig)
                            {
                                    ssxDit = GPUComplex_mult(I_epsR_array[ifreq*nFreq + my_igp * ngpown + ig],fact1) + GPUComplex_mult(I_epsR_array[(ifreq+1) * nFreq + my_igp * ngpown +ig],fact2) ;
                                    ssxDitt += GPUComplex_product(aqsntemp[n1*number_bands + ig] , GPUComplex_product(GPUComplex_conj(aqsmtemp[n1 * number_bands + igp]) , ssxDit));
                            }
                        }
                        ssxDittt += GPUComplex_mult(ssxDitt , vcoul[igp]);
                    }
                }
            }

            ssxDi[iw] += ssxDittt;

            if(flag_occ)
            {
                asxDtemp[iw] += GPUComplex_mult(ssxDi[iw], occ);
            }
        }
    }
//Till 341

    std::chrono::duration<double> elapsedTime = std::chrono::high_resolution_clock::now() - startTimer;


//    for(int n1 = 0; n1 < number_bands; ++n1)
//    {
//        flag_occ = n1 < nvband;
//
//
//        for(int ifreq = 0; ifreq < nFreq; ++ifreq)
//        {
//            GPUComplex cedifft_cor = expr0;
//            GPUComplex schDt_right = expr0;
//            GPUComplex schDt_left = expr0;
//            GPUComplex schDt_avg = expr0;
//            GPUComplex schDt_lin = expr0;
//            GPUComplex schDt_lin2 = expr0;
//
//            double cedifft_zb_right, cedifft_zb_left;
//
//            GPUComplex schDr = schDt_matrix[n1*number_bands + ifreq];
//            double cedifft_zb = dFreqGrid[ifreq];
//            GPUComplex cedifft_complex = GPUComplex(0.00, cedifft_zb);
//            GPUComplex cedifft_coh = doubleMinusGPUComplex(cedifft_zb, cedifft_complex);
//
//            if(flag_occ)
//                cedifft_cor = GPUComplex_mult(cedifft_coh, -1.0);
//                else
//                    cedifft_cor = cedifft_coh;
//
//
//            if(ifreq != 0)
//            {
//                cedifft_zb_right = cedifft_zb;
//                cedifft_zb_left = dFreqGrid[ifreq-1];
//                schDt_right = schDt;
//                schDt_left = schDt_matrix[n1*number_bands + ifreq-1];
//                schDt_avg = GPUComplex_mult(schDt_right+schDt_left, 0.5);
//                schDt_lin = schDt_right - schDt_left;
//                schDt_lin2 = GPUComplex_divide(schDt_lin , (cedifft_zb_right - cedifft_zb_left));
//            }
//
//            if(ifreq != nFreq)
//            {
//                for(int iw=0; iw < nfreqval; ++iw)
//                {
//                    double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalmin;
//                    if(iw == 0) wx = freqevalmin - ekq[n1] ; 
//
//                    schDi[iw] -= GPUComplex_product(GPUComplex(0.00, pref[ifreq]) , schDt) / doubleMinusGPUComplex(wx, cedifft_coh);
//                    schDi_corb[iw] -= GPUComplex_product(GPUComplex(0.00, pref[ifreq]) , schDt) / doubleMinusGPUComplex(wx, cedifft_cor);
//
//                }
//            }
//
//            if(ifreq != 0)
//            {
//                for(int iw = 0; iw < nfreqval; ++iw)
//                {
//                    double intfact = (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left);
//                    if(intfact < 0.0001) intfact = 0.0001;
//                    if(intfact > 10000) intfact = 10000;
//                    intfact = -log(intfact);
//
//                    sch2Di[iw] -= GPUComplex_mult(GPUComplex_product(GPUComplex(0.00, pref_zb), schDt_avg), intfact);
//
//                    if(flag_occ)
//                    {
//                        intfact = (freqevalmin - ekq[n1] + (iw-1) * freqevalstep + cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_left);
//                        if(intfact < 0.0001) intfact = 0.0001;
//                        if(intfact > 10000) intfact = 10000;
//                        intfact = -log(intfact);
//
//                        schDt_lin3 = GPUComplex_mult(GPUComplex_mult((schDt_left + schDt_lin2) , (-freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_left)), intfact) ;
//                    }
//                    else
//                        schDt_lin3 = GPUComplex_mult(GPUComplex_mult((schDt_left + schDt_lin2) , (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_left)), intfact) ;
//
//                    schDt_lin3 += schDt_lin;
//                    schDi_cor[iw] -= GPUComplex_product(GPUComplex(0.00, pref_zb), schDt_lin3);
//                }
//            }
//        }
//
//        for(int iw = 0; iw < nfreqval; ++iw)
//        {
//            double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep;
//            if(wx > 0.00)
//            {
//                int ifreq = 0;
//                for(int ijk = 0; ijk < nFreq-1; ++ijk)
//                {
//                    if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
//                        ifreq = ijk;
//                }
//                if(ifreq == 0.00 ) ifreq = nFreq-1;
//
//                fact1 = -0.5*(dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
//                fact2 = -0.5*(wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
//
//                GPUComplex schDttt = expr0;
//                GPUComplex schDttt_cor = expr0;
//                for(int my_igp = 0; my_igp < ngpown; ++my_igp)
//                {
//                    int indigp = inv_igp_index[my_igp];
//                    int igp = indinv[indigp];
//                    if(!(igp > ncouls || igp < 0))
//                    {
//                        GPUComplex sch2Dt = expr0;
//                        GPUComplex sch2Dtt = expr0;
//                        igmax = ncouls;
//                        for(int ig = 0; ig < igmax; ++ig)
//                        {
//                            sch2Dt =GPUComplex_mult( I_epsR_array[ifreq*nFreq + my_igp * ngpown + ig] - I_epsA_array[ifreq*nFreq + my_igp*ngpown + ig] , fact1) + \
//                                    GPUComplex_mult(I_epsR_array[(ifreq+1)*nFreq + my_igp*ngpown + ig] - I_epsA_array[(ifreq+1)*nFreq + my_igp*ngpown + ig], fact2);
//
//                            sch2Dtt += GPUComplex_product(GPUComplex_product(aqsntemp[n1*number_bands + ig] , GPUComplex_conj(aqsmtemp[n1*number_bands + igp])) , sch2Dt);
//
//                        }
//                        schDttt += GPUComplex_mult(sch2Dtt, vcoul[igp]);
//
//                        if(!flag_occ) schDttt_cor += GPUComplex_mult(sch2Dtt, vcoul[igp]);
//                    }
//                }
//
//                sch2Di[iw] += schDttt;
//                schDi_cor[iw] += schDttt_cor;
//            }
//            else if (flag_occ)
//            {
//                wx = -wx;
//                int ifreq = 0;
//                for(int ijk = 0; ijk < nFreq-1; ++ijk)
//                {
//                    if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
//                        ifreq = ijk;
//                }
//                if(ifreq == 0.00 ) ifreq = nFreq-1;
//
//                fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
//                fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
//                GPUComplex schDttt_cor = expr0;
//                for(int my_igp = 0; my_igp < ngpown; ++my_igp)
//                {
//                    int indigp = inv_igp_index[my_igp];
//                    int igp = indinv[indigp];
//                    if(!(igp > ncouls || igp < 0))
//                    {
//                        GPUComplex sch2Dt = expr0;
//                        GPUComplex sch2Dtt = expr0;
//                        igmax = ncouls;
//                        for(int ig = 0; ig < igmax; ++ig)
//                        {
//                            sch2Dt =GPUComplex_mult( I_epsR_array[ifreq*nFreq + my_igp * ngpown + ig] - I_epsA_array[ifreq*nFreq + my_igp*ngpown + ig] , fact1*-0.5) + \
//                                    GPUComplex_mult(I_epsR_array[(ifreq+1)*nFreq + my_igp*ngpown + ig] - I_epsA_array[(ifreq+1)*nFreq + my_igp*ngpown + ig], fact2*-0.5);
//
//                            sch2Dtt += GPUComplex_product(GPUComplex_product(aqsntemp[n1*number_bands + ig] , GPUComplex_conj(aqsmtemp[n1*number_bands + igp])) , sch2Dt);
//
//                        }
//                        schDttt_cor += GPUComplex_mult(sch2Dtt, vcoul[igp]);
//                    }
//                }
//            }
//        }
//
//        for(int iw = 0; iw < nfreqval; ++iw)
//        {
//            achDtemp[iw] += schDi[iw];
//            achDtemp_cor[iw] += schDi_cor[iw];
//            achDtemp_corb[iw] += schDi_corb[iw];
//            ach2Dtemp[iw] += sch2Di[iw];
//        }
//    }


    cout << "asxDtemp " << endl;
    asxDtemp[0].print();

    cout << "********** Total Time Taken **********= " << elapsedTime.count() << " secs" << endl;


    free(vcoul);
    free(ekq);
    free(aqsntemp);
    free(aqsmtemp);
    free(I_epsR_array);
    free(I_epsA_array);
    free(inv_igp_index);
    free(indinv);
    free(pref);
    free(dFreqBrd);

   free( asxDtemp) ;
   free( achDtemp) ;
   free( achDtemp_cor) ;
   free( achDtemp_corb);
   free( ach2Dtemp) ;
   free( schDi );
   free( schDi_cor) ;
   free( schDi_corb ) ;
   free( sch2Di ) ;
   free( ssxDi) ;

   free(wxi);
   free(schDt_matrix);

    return 0;
}
