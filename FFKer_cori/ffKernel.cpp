#include <iostream>
#include <chrono>
#include <complex>
#include <omp.h>
#include "Complex.h"

using namespace std;

void schDttt_corKernel1(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex &schDttt, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

void schDttt_corKernel2(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

void calculate_schDt_lin3(GPUComplex& schDt_lin3, GPUComplex* sch2Di, bool flag_occ, int freqevalmin, double *ekq, int iw, int freqevalstep, double cedifft_zb_right, double cedifft_zb_left, GPUComplex schDt_left, GPUComplex schDt_lin2, int n1, double pref_zb, GPUComplex pref_zb_compl, GPUComplex schDt_avg)
{
    double intfact = (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_left);
    if(intfact < 0.0001) intfact = 0.0001;
    if(intfact > 10000) intfact = 10000;
    intfact = -log(intfact);
    sch2Di[iw] = sch2Di[iw] - GPUComplex_mult(GPUComplex_product(pref_zb_compl , schDt_avg) , intfact);
    if(flag_occ)
    {
       double  intfact = abs((freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_left));
        if(intfact < 0.0001) intfact = 0.0001;
        if(intfact > 10000) intfact = 10000;
        intfact = log(intfact);
        schDt_lin3 = GPUComplex_mult((schDt_left + schDt_lin2) , (-freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact) ;
    }
    else
        schDt_lin3 = GPUComplex_mult((schDt_left + schDt_lin2) , (freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact);

}

void compute_fact(double wx, int nFreq, double *dFreqGrid, double &fact1, double &fact2, int &ifreq, int loop, bool flag_occ)
{
    if(loop == 1 && wx > 0.00)
    {
            for(int ijk = 0; ijk < nFreq-1; ++ijk)
            {
                if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                ifreq = ijk;
            }
            if(ifreq == 0) ifreq = nFreq-2;
            fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
            fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
    }
    else if(loop == 1)
    {
        for(int ijk = 0; ijk < nFreq-1; ++ijk)
        {
            if(-wx > dFreqGrid[ijk] && -wx < dFreqGrid[ijk+1])
                ifreq = ijk;
        }
        if(ifreq == 0) ifreq = nFreq-2;
        fact1 = (dFreqGrid[ifreq+1] + wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
        fact2 = (-dFreqGrid[ifreq] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
    }
    if(loop == 2 && wx > 0.00)
    {
            for(int ijk = 0; ijk < nFreq-1; ++ijk)
            {
                if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                    ifreq = ijk;
            }
            if(ifreq == -1) ifreq = nFreq-2;
            fact1 = -0.5 * (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
            fact2 = -0.5 * (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
    }
    else if(loop == 2 && flag_occ)
    {
        wx = -wx; ifreq = 0;
        for(int ijk = 0; ijk < nFreq-1; ++ijk)
        {
            if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                ifreq = ijk;
        }
        if(ifreq == 0) ifreq = nFreq-2;
        fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
        fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 

    }
}

void ssxDittt_kernel(int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_eps_array, GPUComplex *ssxDittt, int ngpown, int ncouls, int n1,int ifreq, double fact1, double fact2)
{
#pragma omp parallel for default(shared)
    for(int my_igp = 0; my_igp < ngpown; ++my_igp)
    {
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        int tid = omp_get_thread_num();
        GPUComplex ssxDit(0.00, 0.00);
        GPUComplex ssxDitt(0.00, 0.00);

        for(int ig = 0; ig < ncouls; ++ig)
        {
            ssxDit = GPUComplex_mult(I_eps_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] , fact1 ) + \
                                         GPUComplex_mult(I_eps_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] , fact2);

            ssxDitt += GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_product(GPUComplex_conj(aqsmtemp[n1*ncouls + igp]) , ssxDit));
        }
        ssxDittt[tid] += GPUComplex_mult(ssxDitt , vcoul[igp]);
    }
}


void achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, double *vcoul, GPUComplex *achsDtemp_threadArr)
{
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
#pragma omp parallel for default(shared)
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            int tid = omp_get_thread_num();

            GPUComplex schsDtemp(0.00, 0.00);

            for(int ig = 0; ig < ncouls; ++ig)
                schsDtemp = schsDtemp - GPUComplex_product(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_conj(aqsmtemp[n1*ncouls + igp])) , I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]);

            achsDtemp_threadArr[tid] += GPUComplex_mult(schsDtemp , vcoul[igp] * 0.5);
        }
    } //n1

}

void asxDtemp_Kernel(int nvband, int nfreqeval, int ncouls, int ngpown, int numThreads, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *ssxDittt, GPUComplex *asxDtemp)
{
    GPUComplex expr0(0.00, 0.00);
    for(int n1 = 0; n1 < nvband; ++n1)
    {
        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            for(int i = 0; i < numThreads; ++i)
                ssxDittt[i] = expr0;

            double wx = freqevalmin - ekq[n1] + freqevalstep;
            double fact1 = 0.00, fact2 = 0.00;
            int ifreq = 0;
//            ssxDi[iw] = expr0;
            GPUComplex ssxDittt_agg = expr0;

            for(int i = 0; i < numThreads; ++i)
                ssxDittt[i] = expr0;

            compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 1, 0);

//The ssxDittt_kernel is OMP parallelized.
                ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);

            for(int i = 0; i < numThreads; ++i)
                ssxDittt_agg += ssxDittt[i];

//            ssxDi[iw] += ssxDittt_agg;
            asxDtemp[iw] += GPUComplex_mult(ssxDittt_agg , occ);
        } // iw
    }
}

void achDtemp_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int numThreads, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double pref_zb, double *pref, double *dFreqGrid, GPUComplex *dFreqBrd, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *schDt_matrix, GPUComplex *schDi, GPUComplex *schDi_cor, GPUComplex *sch2Di, GPUComplex *achDtemp)
{
    bool flag_occ;
    GPUComplex expr0(0.00, 0.00);
#pragma omp parallel for default(shared) collapse(2)
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        for(int ifreq = 0; ifreq < nFreq; ++ifreq)
        {
            flag_occ = n1 < nvband;
            GPUComplex schDt = schDt_matrix[n1*nFreq + ifreq];
            double cedifft_zb = dFreqGrid[ifreq];
            double cedifft_zb_right, cedifft_zb_left;
            GPUComplex schDt_right, schDt_left, schDt_avg, schDt_lin, schDt_lin2, schDt_lin3;
            GPUComplex cedifft_compl(cedifft_zb, 0.00);
            GPUComplex cedifft_cor;
            GPUComplex cedifft_coh = cedifft_compl - dFreqBrd[ifreq];
            GPUComplex pref_zb_compl(0.00, pref_zb);

            if(flag_occ)
                cedifft_cor = GPUComplex_mult(cedifft_compl, -1) - dFreqBrd[ifreq];
                else
                    cedifft_cor = cedifft_compl - dFreqBrd[ifreq];

            if(ifreq != 0)
            {
                cedifft_zb_right = cedifft_zb;
                cedifft_zb_left = dFreqGrid[ifreq-1];
                schDt_right = schDt;
                schDt_left = schDt_matrix[n1*nFreq + ifreq-1];
                schDt_avg = GPUComplex_mult((schDt_right + schDt_left) , 0.5);
                schDt_lin = schDt_right - schDt_left;
                schDt_lin2 = GPUComplex_divide(schDt_lin , (cedifft_zb_right - cedifft_zb_left));

                for(int iw = 0; iw < nfreqeval; ++iw)
                {
                    sch2Di[iw] = expr0;
                    calculate_schDt_lin3(schDt_lin3, sch2Di, flag_occ, freqevalmin, ekq, iw, freqevalstep, cedifft_zb_right, cedifft_zb_left, schDt_left, schDt_lin2, n1, pref_zb, pref_zb_compl, schDt_avg);

                    schDt_lin3 += schDt_lin;
                    schDi_cor[iw] = schDi_cor[iw] -  GPUComplex_product(pref_zb_compl , schDt_lin3);
                }
            }

                for(int iw = 0; iw < nfreqeval; ++iw)
                {
                    schDi[iw] = expr0;
                    double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep;
                    GPUComplex tmp(0.00, pref[ifreq]);
                    schDi[iw] = schDi[iw] - GPUComplex_divide(GPUComplex_product(tmp,schDt) , doubleMinusGPUComplex(wx, cedifft_coh));
                    achDtemp[iw] += schDi[iw];
                }
        }
    }

}

void achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int numThreads, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double pref_zb, double *pref, double *dFreqGrid, GPUComplex *dFreqBrd, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *schDt_matrix, GPUComplex *schDi, GPUComplex *schDi_cor, GPUComplex *schDi_corb, GPUComplex *sch2Di, GPUComplex *ach2Dtemp, GPUComplex *achDtemp_cor, GPUComplex *achDtemp_corb)
{
    bool flag_occ;
    GPUComplex expr0(0.00, 0.00);
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        flag_occ = n1 < nvband;

        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            schDi_corb[iw] = expr0;
            schDi_cor[iw] = expr0;
            double wx = freqevalmin - ekq[n1] + freqevalstep;

            double fact1 = 0.00, fact2 = 0.00;
            int ifreq = 0.00;

            compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 2, flag_occ);

            if(wx > 0)
            {
                if(!flag_occ)
                schDttt_corKernel1(schDi_cor[iw], inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, sch2Di[iw],vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);
            }
            else if(flag_occ)
                schDttt_corKernel2(schDi_cor[iw], inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);


//Summing up at the end of iw loop
            ach2Dtemp[iw] += sch2Di[iw];
            achDtemp_cor[iw] += schDi_cor[iw];
            achDtemp_corb[iw] += schDi_corb[iw];
        }// iw
    } //n1
}

void schDttt_corKernel1(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex &schDttt, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
{
    int blkSize = 128;
    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00, \
        schDttt_re = 0.00, schDttt_im = 0.00;
#pragma omp parallel for default(shared) collapse(2) reduction(+:schDttt_cor_re, schDttt_cor_im, schDttt_re, schDttt_im)
    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            for(int ig = igbeg; ig < min(ncouls, igbeg+blkSize); ++ig)
            {
                int indigp = inv_igp_index[my_igp] ;
                int igp = indinv[indigp];
                GPUComplex sch2Dt = GPUComplex_mult((GPUComplex_minus(I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig])) , fact1) + \
                                            GPUComplex_mult((GPUComplex_minus(I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig])) , fact2);
                GPUComplex sch2Dtt = GPUComplex_mult(GPUComplex_product(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_conj(aqsmtemp[n1*ncouls + igp])) , sch2Dt), vcoul[igp]);
                schDttt_re += GPUComplex_real(sch2Dtt) ;
                schDttt_im += GPUComplex_imag(sch2Dtt) ;
                schDttt_cor_re += GPUComplex_real(sch2Dtt) ;
                schDttt_cor_im += GPUComplex_imag(sch2Dtt) ;
            }
        }
    }
    GPUComplex tmp(schDttt_cor_re, schDttt_cor_im);
    schDttt_cor = tmp;

}

void schDttt_corKernel2(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
{
    int blkSize = 128;
    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00;
#pragma omp parallel for default(shared) collapse(2) reduction(+:schDttt_cor_re, schDttt_cor_im)
    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            for(int ig = igbeg; ig < min(ncouls, igbeg+blkSize); ++ig)
            {
                int indigp = inv_igp_index[my_igp] ;
                int igp = indinv[indigp];
                GPUComplex sch2Dt = GPUComplex_mult(GPUComplex_mult((GPUComplex_minus(I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[ifreq*ncouls*ngpown + my_igp*ncouls + ig])) , fact1) + \
                                            GPUComplex_mult((GPUComplex_minus(I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig])) , fact2), -0.5);
                GPUComplex sch2Dtt = GPUComplex_mult(GPUComplex_product(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_conj(aqsmtemp[n1*ncouls + igp])) , sch2Dt), vcoul[igp]);
                schDttt_cor_re += GPUComplex_real(sch2Dtt) ;
                schDttt_cor_im += GPUComplex_imag(sch2Dtt) ;
            }
        }
    }
    GPUComplex tmp(schDttt_cor_re, schDttt_cor_im);
    schDttt_cor = tmp;
}

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

    GPUComplex expr0( 0.0 , 0.0);
    GPUComplex expr( 0.5 , 0.5);
    GPUComplex expR( 0.5 , 0.5);
    GPUComplex expA( 0.5 , -0.5);
    GPUComplex exprP1( 0.5 , 0.1);
    double pref_zb = 0.5 / 3.14;

//Start to allocate the data structures;
    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls];
    double *vcoul = new double[ncouls];
    double *ekq = new double[number_bands];
    double *dFreqGrid = new double[nFreq];
    double *pref = new double[nFreq];
    long double mem_alloc = 0.00;

    GPUComplex *aqsntemp = new GPUComplex[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(GPUComplex));

    GPUComplex *aqsmtemp= new GPUComplex[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(GPUComplex));

    GPUComplex *I_epsR_array = new GPUComplex[nFreq * ngpown * ncouls];
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(GPUComplex));

    GPUComplex *I_epsA_array = new GPUComplex[nFreq * ngpown * ncouls];
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(GPUComplex));

    GPUComplex *ssxDi = new GPUComplex[nfreqeval];
    GPUComplex *schDi = new GPUComplex[nfreqeval];
    GPUComplex *sch2Di = new GPUComplex[nfreqeval];
    GPUComplex *schDi_cor = new GPUComplex[nfreqeval];
    GPUComplex *schDi_corb = new GPUComplex[nfreqeval];
    GPUComplex *achDtemp = new GPUComplex[nfreqeval];
    GPUComplex *ach2Dtemp = new GPUComplex[nfreqeval];
    GPUComplex *achDtemp_cor = new GPUComplex[nfreqeval];
    GPUComplex *achDtemp_corb = new GPUComplex[nfreqeval];
    GPUComplex *asxDtemp = new GPUComplex[nfreqeval];
    GPUComplex *dFreqBrd = new GPUComplex[nFreq];
    mem_alloc += (nfreqeval * 10 * sizeof(GPUComplex));
    mem_alloc += (nFreq * sizeof(GPUComplex)) ;

    GPUComplex *schDt_matrix = new GPUComplex[number_bands * nFreq];
    mem_alloc += (nFreq * number_bands * sizeof(GPUComplex));

    //Variables used : 
    GPUComplex achsDtemp = expr0;
    GPUComplex *achsDtemp_threadArr = new GPUComplex[numThreads];
    for(int i = 0; i < numThreads; ++i)
    {
        achsDtemp_threadArr[i] = expr0;
    }


    double freqevalmin = 0.00;
    double freqevalstep = 0.50;
    double dw = -10;
    double occ = 1.00;

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
            aqsmtemp[i*ncouls+j] = expr;
            aqsntemp[i*ncouls+j] = expr;
        }

        for(int j=0; j<nFreq; ++j)
            schDt_matrix[i*nFreq + j] = expr0;
    }

    for(int i=0; i<ncouls; ++i)
        vcoul[i] = 1.00;

    for(int i=0; i<nFreq; ++i)
    {
        for(int j=0; j<ngpown; ++j)
        {
            for(int k=0; k<ncouls; ++k)
            {
                I_epsR_array[i*ngpown*ncouls + j * ncouls + k] = expR;
                I_epsA_array[i*ngpown*ncouls + j * ncouls + k] = expA;
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
    GPUComplex *ssxDittt = new GPUComplex[numThreads];

    cout << "Memory Used = " << mem_alloc/(1024 * 1024 * 1024) << " GB" << endl;
    std::chrono::duration<double> elapsedTime_preloop = std::chrono::high_resolution_clock::now() - startTimer;
    cout << "********** PreLoop **********= " << elapsedTime_preloop.count() << " secs" << endl;

    cout << "starting Kernels" << endl;
    auto startTime_Kernel = std::chrono::high_resolution_clock::now();

    /***********achsDtemp Kernel ****************/
    auto startTimer_achsDtemp = std::chrono::high_resolution_clock::now();
    achsDtemp_Kernel(number_bands, ngpown, ncouls, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, vcoul, achsDtemp_threadArr);
    for(int i = 0; i < numThreads; ++i)
        achsDtemp += achsDtemp_threadArr[i];
    std::chrono::duration<double> elapsedTime_achsDtemp = std::chrono::high_resolution_clock::now() - startTimer_achsDtemp;
    cout << "********** achsDtemp Kernel time  **********= " << elapsedTime_achsDtemp.count() << " secs" << endl;

    /***********asxDtemp Kernel ****************/
    auto startTimer_asxDtemp = std::chrono::high_resolution_clock::now();
    asxDtemp_Kernel(nvband, nfreqeval, ncouls, ngpown, numThreads, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, ssxDittt, asxDtemp);
    std::chrono::duration<double> elapsedTime_asxDtemp = std::chrono::high_resolution_clock::now() - startTimer_asxDtemp;
    cout << "********** asxDtemp Kernel time  **********= " << elapsedTime_asxDtemp.count() << " secs" << endl;

    /***********achDtemp Kernel ****************/
    auto startTimer_achDtemp = std::chrono::high_resolution_clock::now();
    achDtemp_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, numThreads, nFreq, freqevalmin, freqevalstep, occ, ekq, pref_zb, pref, dFreqGrid, dFreqBrd, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, schDt_matrix, schDi, schDi_cor, sch2Di, asxDtemp);
    std::chrono::duration<double> elapsedTime_achDtemp = std::chrono::high_resolution_clock::now() - startTimer_achDtemp;
    cout << "********** achDtemp Kernel time **********= " << elapsedTime_achDtemp.count() << " secs" << endl;

    /***********achDtemp_cor Kernel ****************/
    auto startTimer_achDtemp_cor = std::chrono::high_resolution_clock::now();
    achDtemp_cor_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, numThreads, nFreq, freqevalmin, freqevalstep, occ, ekq, pref_zb, pref, dFreqGrid, dFreqBrd, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, schDt_matrix, schDi, schDi_cor, schDi_corb, sch2Di, ach2Dtemp, achDtemp_cor, achDtemp_corb);
    std::chrono::duration<double> elapsedTime_achDtemp_cor = std::chrono::high_resolution_clock::now() - startTimer_achDtemp_cor;
    cout << "********** achDtemp_cor Kernel time **********= " << elapsedTime_achDtemp_cor.count() << " secs" << endl;

    std::chrono::duration<double> elapsedTime_Kernel = std::chrono::high_resolution_clock::now() - startTime_Kernel;

    cout << "achsDtemp = " ;
    achsDtemp.print();
    cout << "asxDtemp = " ;
    asxDtemp[0].print();
    cout << "achDtemp_cor = " ;
    achDtemp_cor[0].print();

    std::chrono::duration<double> elapsedTime = std::chrono::high_resolution_clock::now() - startTimer;
    cout << "********** Total Kernel Time **********= " << elapsedTime_Kernel.count() << " secs" << endl;
//    cout << "********** Total Time Taken **********= " << elapsedTime.count() << " secs" << endl;

//Free the allocated memory 
    free(aqsntemp);
    free(aqsmtemp);
    free(I_epsA_array);
    free(I_epsR_array);
    free(inv_igp_index);
    free(indinv);
    free(vcoul);
    free(ekq);
    free(dFreqGrid);
    free(pref);
    free(ssxDi);
    free(schDi);
    free(sch2Di);
    free(schDi_cor);
    free(schDi_corb);
    free(achDtemp);
    free(ach2Dtemp);
    free(achDtemp_cor);
    free(achDtemp_corb);
    free(asxDtemp);
    free(dFreqBrd);
    free(schDt_matrix);

    return 0;
}
