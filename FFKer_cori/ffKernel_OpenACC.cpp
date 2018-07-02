#include "Complex_OpenACC.h"
#define OpenACC 1

using namespace std;

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
void compute_sch2Dt(GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, int ifreq, int ngpown, int my_igp, int ncouls, int n1, int ig, double fact1, double fact2, int igp, GPUComplex &sch2Dtt)
{
    GPUComplex sch2Dt = GPUComplex_mult((GPUComplex_minus(I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig])) , fact1) + \
                                GPUComplex_mult((GPUComplex_minus(I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig])) , fact2);
    sch2Dtt = GPUComplex_mult(GPUComplex_product(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_conj(aqsmtemp[n1*ncouls + igp])) , sch2Dt), vcoul[igp]);
}

#pragma acc routine
void ssxDittt_kernel(int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex &ssxDittt, int number_bands, int ngpown, int ncouls, int nFreq, int n1,int ifreq, double fact1, double fact2)
{
    double ssxDittt_re = 0.00, ssxDittt_im = 0.00;
//#if OpenACC
//#pragma acc loop worker reduction(+:ssxDittt_re, ssxDittt_im)
//#endif
    for(long int my_igp = 0; my_igp < ngpown; ++my_igp)
    {
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        GPUComplex ssxDit(0.00, 0.00);
        GPUComplex ssxDitt(0.00, 0.00);

        for(long int ig = 0; ig < ncouls; ++ig)
        {
            ssxDit = GPUComplex_mult(I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] , fact1 ) + \
                                         GPUComplex_mult(I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] , fact2);

            ssxDitt += GPUComplex_mult(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_product(GPUComplex_conj(aqsmtemp[n1*ncouls + igp]) , ssxDit)), vcoul[igp]);
        }
        ssxDittt_re += GPUComplex_real(ssxDitt);
        ssxDittt_im += GPUComplex_imag(ssxDitt);
    }
    GPUComplex tmp(ssxDittt_re, ssxDittt_im);
    ssxDittt = tmp;
}


void achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int nFreq, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, double *vcoul, GPUComplex &achsDtemp)
{
    double achsDtemp_re = 0.00, achsDtemp_im = 0.00;
#if OpenACC
#pragma acc parallel loop gang present(inv_igp_index[0:ngpown], indinv[0:ncouls], aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls]) reduction(+:achsDtemp_re, achsDtemp_im)
#endif
    for(long int n1 = 0; n1 < number_bands; ++n1)
    {

#if OpenACC
#pragma acc loop worker reduction(+:achsDtemp_re, achsDtemp_im)
#endif
        for(long int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            GPUComplex schsDtemp(0.00, 0.00);

#if OpenACC
#pragma acc loop vector
#endif
            for(int ig = 0; ig < ncouls; ++ig)
                schsDtemp = schsDtemp - GPUComplex_product(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_conj(aqsmtemp[n1*ncouls + igp])) , I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]);

            achsDtemp_re += GPUComplex_real(GPUComplex_mult(schsDtemp , vcoul[igp] * 0.5));
            achsDtemp_im += GPUComplex_imag(GPUComplex_mult(schsDtemp , vcoul[igp] * 0.5));
        }
    } //n1
    GPUComplex tmp(achsDtemp_re, achsDtemp_im);
    achsDtemp = tmp;
}

void asxDtemp_Kernel(int nvband, int nfreqeval, int number_bands, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, double *asxDtemp_re, double* asxDtemp_im)
{
    GPUComplex expr0(0.00, 0.00);
    for(long int n1 = 0; n1 < nvband; ++n1)
    {
#if OpenACC
#pragma acc parallel loop gang present(inv_igp_index[0:ngpown], indinv[0:ncouls], aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls], \
        ekq[0:number_bands], dFreqGrid[0:nFreq]) 
#endif
        for(long int iw = 0; iw < nfreqeval; ++iw)
        {
//            asxDtemp_re[iw] += 0.5;
//            asxDtemp_im[iw] += 0.5;

            double wx = freqevalmin - ekq[n1] + freqevalstep;
            double fact1 = 0.00, fact2 = 0.00;
            int ifreq = 0;
            GPUComplex ssxDittt = expr0;
            compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 1, 0);

//The ssxDittt_kernel is OMP parallelized.
//                ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, ssxDittt, number_bands, ngpown, ncouls, nFreq, n1, ifreq, fact1, fact2);
//Rahul - Unfunctionified ssxDittt_kernel because cant separate gangs and workers into different functions for now...o


                double ssxDittt_re = 0.00, ssxDittt_im = 0.00;
//#if OpenACC
//#pragma acc loop worker reduction(+:ssxDittt_re, ssxDittt_im)
//#endif
                for(long int my_igp = 0; my_igp < ngpown; ++my_igp)
                {
                    int indigp = inv_igp_index[my_igp];
                    int igp = indinv[indigp];
                    GPUComplex ssxDit(0.00, 0.00);
                    GPUComplex ssxDitt(0.00, 0.00);
        
//#if OpenACC
//#pragma acc loop worker
//#endif
                    for(long int ig = 0; ig < ncouls; ++ig)
                    {
                        ssxDit = GPUComplex_mult(I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] , fact1 ) + \
                                                     GPUComplex_mult(I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] , fact2);
        
                        ssxDitt += GPUComplex_mult(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_product(GPUComplex_conj(aqsmtemp[n1*ncouls + igp]) , ssxDit)), vcoul[igp]);
                    }
                    ssxDittt_re += GPUComplex_real(ssxDitt);
                    ssxDittt_im += GPUComplex_imag(ssxDitt);
                }
                GPUComplex tmp(ssxDittt_re, ssxDittt_im);
                ssxDittt = tmp;

//            asxDtemp[iw] += GPUComplex_mult(ssxDittt, occ);
            asxDtemp_re[iw] += GPUComplex_real(GPUComplex_mult(ssxDittt, occ));
            asxDtemp_im[iw] += GPUComplex_imag(GPUComplex_mult(ssxDittt, occ));
        } // iw
    }
}

void achDtemp_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double pref_zb, double *pref, double *dFreqGrid, GPUComplex *dFreqBrd, GPUComplex *schDt_matrix, GPUComplex *schDi, GPUComplex *schDi_cor, GPUComplex *sch2Di, GPUComplex *achDtemp)
{
    bool flag_occ;
    GPUComplex expr0(0.00, 0.00);
#pragma acc parallel loop copyin(flag_occ, expr0, number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq[0:number_bands], pref_zb, pref[0:nFreq], dFreqGrid[0:nFreq], dFreqBrd[0:nFreq], schDt_matrix[0:number_bands*nFreq], schDi[0:nfreqeval], schDi_cor[0:nfreqeval], sch2Di[0:nfreqeval], achDtemp[0:nfreqeval])
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

#pragma acc routine
void schDttt_corKernel1(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex &schDttt, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2, int nFreq, int number_bands)
{
    int blkSize = 128;
    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00, \
        schDttt_re = 0.00, schDttt_im = 0.00;
//#pragma acc parallel loop copyin(inv_igp_index[0:ngpown], indinv[0:ncouls], aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls],I_epsA_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls]) reduction(+:schDttt_cor_re, schDttt_cor_im, schDttt_re, schDttt_im)
    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            for(int ig = igbeg; ig < min(ncouls, igbeg+blkSize); ++ig)
            {
                int indigp = inv_igp_index[my_igp] ;
                int igp = indinv[indigp];
                GPUComplex sch2Dtt(0.00, 0.00);
                compute_sch2Dt(I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul, ifreq, ngpown, my_igp, ncouls, n1, ig, fact1, fact2, igp, sch2Dtt);

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

void schDttt_corKernel2(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, int number_bands, int nFreq, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
{
    int blkSize = 128;
    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00;
//#pragma omp parallel for default(shared) collapse(2) reduction(+:schDttt_cor_re, schDttt_cor_im)
//#pragma acc parallel loop copyin(inv_igp_index[0:ngpown], indinv[0:ncouls], aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls],I_epsA_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls]) reduction(+:schDttt_cor_re, schDttt_cor_im)
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

void achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *schDi_cor, GPUComplex *schDi_corb, GPUComplex *sch2Di, GPUComplex *ach2Dtemp, double *achDtemp_cor_re, double* achDtemp_cor_im, GPUComplex *achDtemp_corb)
{
    bool flag_occ;
    GPUComplex expr0(0.00, 0.00);
    double achtemp_cor_loc_re = 0.00, achtemp_cor_loc_im = 0.00;

#if OpenACC
#pragma acc parallel loop gang present(inv_igp_index[0:ngpown], indinv[0:ncouls], aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], I_epsA_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls], \
        ekq[0:number_bands], dFreqGrid[0:nFreq], schDi_cor[0:nfreqeval], schDi_corb[0:nfreqeval], sch2Di[0:nfreqeval], ach2Dtemp[0:nfreqeval], achDtemp_corb[0:nfreqeval]) \
        reduction(+:achtemp_cor_loc_re, achtemp_cor_loc_im)
#endif
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        flag_occ = n1 < nvband;

        for(int iw = 0; iw < 1; ++iw)
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
                schDttt_corKernel1(schDi_cor[iw], inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, sch2Di[iw],vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2, nFreq, number_bands);
            }
            else if(flag_occ)
                schDttt_corKernel2(schDi_cor[iw], inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  number_bands, nFreq, ncouls, ifreq, ngpown, n1, fact1, fact2);


//Summing up at the end of iw loop
            ach2Dtemp[iw] += sch2Di[iw];
//            achDtemp_cor_re[iw] += GPUComplex_real(schDi_cor[iw]);
//            achDtemp_cor_im[iw] += GPUComplex_imag(schDi_cor[iw]);
            achDtemp_corb[iw] += schDi_corb[iw];
            achtemp_cor_loc_re += GPUComplex_real(schDi_cor[iw]);
            achtemp_cor_loc_im += GPUComplex_imag(schDi_cor[iw]);
        }// iw
    } //n1

    achDtemp_cor_re[0] = achtemp_cor_loc_re;
    achDtemp_cor_im[0] = achtemp_cor_loc_im;
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
    cout << "\n number_bands = " << number_bands << \
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

    double *achDtemp_cor_re = new double[nfreqeval];
    double *achDtemp_cor_im = new double[nfreqeval];

    GPUComplex *schDt_matrix = new GPUComplex[number_bands * nFreq];
    mem_alloc += (nFreq * number_bands * sizeof(GPUComplex));
    cout << "Memory Used = " << mem_alloc/(1024 * 1024 * 1024) << " GB" << endl;

    //Variables used : 
    GPUComplex achsDtemp = expr0;

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

       achDtemp_cor_re[i] = 0.00;
       achDtemp_cor_im[i] = 0.00;
   }

    std::chrono::duration<double> elapsedTime_preloop = std::chrono::high_resolution_clock::now() - startTimer;
    cout << "********** PreLoop **********= " << elapsedTime_preloop.count() << " secs" << endl;

    cout << "starting Kernels" << endl;
    auto startTime_Kernel = std::chrono::high_resolution_clock::now();

#pragma acc enter data copyin(inv_igp_index[0:ngpown], indinv[0:ncouls], aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], I_epsA_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls], \
            ekq[0:number_bands], dFreqGrid[0:nFreq], asxDtemp[0:nfreqeval], schDi_cor[0:nfreqeval], schDi_corb[0:nfreqeval], sch2Di[0:nfreqeval], ach2Dtemp[0:nfreqeval], achDtemp_corb[0:nfreqeval])


    /***********achsDtemp Kernel ****************/
    auto startTimer_achsDtemp = std::chrono::high_resolution_clock::now();
    achsDtemp_Kernel(number_bands, ngpown, ncouls, nFreq, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, vcoul, achsDtemp);

    std::chrono::duration<double> elapsedTime_achsDtemp = std::chrono::high_resolution_clock::now() - startTimer_achsDtemp;
    cout << "achsDtemp = " ;
    achsDtemp.print();
#if OpenACC
    cout << "********** achsDtemp+OpenACC Kernel time  **********= " << elapsedTime_achsDtemp.count() << " secs" << endl;
#else
    cout << "********** achsDtemp Kernel time  **********= " << elapsedTime_achsDtemp.count() << " secs" << endl;
#endif

    /***********asxDtemp Kernel ****************/
    double *asxDtemp_re = new double[nfreqeval];
    double *asxDtemp_im = new double[nfreqeval];
    auto startTimer_asxDtemp = std::chrono::high_resolution_clock::now();
    asxDtemp_Kernel(nvband, nfreqeval, number_bands, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, asxDtemp_re, asxDtemp_im);
    std::chrono::duration<double> elapsedTime_asxDtemp = std::chrono::high_resolution_clock::now() - startTimer_asxDtemp;
    cout << "asxDtemp = " ;
    cout << "(" <<  asxDtemp_re[0] << ", " << asxDtemp_im[0] << ")" << endl;
#if OpenACC
    cout << "********** asxDtemp+OpenACC Kernel time  **********= " << elapsedTime_asxDtemp.count() << " secs" << endl;
#else
    cout << "********** asxDtemp Kernel time  **********= " << elapsedTime_asxDtemp.count() << " secs" << endl;
#endif

//    /***********achDtemp Kernel ****************/
//    auto startTimer_achDtemp = std::chrono::high_resolution_clock::now();
//    achDtemp_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, pref_zb, pref, dFreqGrid, dFreqBrd, schDt_matrix, schDi, schDi_cor, sch2Di, achDtemp);
//    std::chrono::duration<double> elapsedTime_achDtemp = std::chrono::high_resolution_clock::now() - startTimer_achDtemp;
//#if OpenACC
//    cout << "********** achDtemp+OpenACC Kernel time **********= " << elapsedTime_achDtemp.count() << " secs" << endl;
//#else
//    cout << "********** achDtemp Kernel time **********= " << elapsedTime_achDtemp.count() << " secs" << endl;
//#endif

    /***********achDtemp_cor Kernel ****************/
    auto startTimer_achDtemp_cor = std::chrono::high_resolution_clock::now();
    achDtemp_cor_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, schDi_cor, schDi_corb, sch2Di, ach2Dtemp, achDtemp_cor_re, achDtemp_cor_im, achDtemp_corb);
    cout << "achDtemp_cor = " ;
    cout << "(" <<  achDtemp_cor_re[0] << ", " << achDtemp_cor_im[0] << ")" << endl;
    std::chrono::duration<double> elapsedTime_achDtemp_cor = std::chrono::high_resolution_clock::now() - startTimer_achDtemp_cor;
#if OpenACC
    cout << "********** achDtemp_cor+OpenACC Kernel time **********= " << elapsedTime_achDtemp_cor.count() << " secs" << endl;
#else
    cout << "********** achDtemp_cor Kernel time **********= " << elapsedTime_achDtemp_cor.count() << " secs" << endl;
#endif

    std::chrono::duration<double> elapsedTime_Kernel = std::chrono::high_resolution_clock::now() - startTime_Kernel;
#pragma acc exit data delete(inv_igp_index[0:ngpown], indinv[0:ncouls], aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], I_epsA_array[0:nFreq*ngpown*ncouls], vcoul[0:ncouls], ekq[0:number_bands], dFreqGrid[0:nFreq], asxDtemp[0:nfreqeval])


    std::chrono::duration<double> elapsedTime = std::chrono::high_resolution_clock::now() - startTimer;
    cout << "********** Total Kernel Time **********= " << elapsedTime_Kernel.count() << " secs" << endl;
    cout << "********** Total Time Taken **********= " << elapsedTime.count() << " secs" << endl;

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

    free(achDtemp_cor_re);
    free(achDtemp_cor_im);

    return 0;
}
