#include "CustomComplex.h"

// Atomic add operation for double
#if defined( __CUDA_ARCH__ ) && __CUDA_ARCH__ >= 600
#define atomicAdd2 atomicAdd
#else
__device__ double atomicAdd2( double *address, double val )
{
    unsigned long long int *address_as_ull = (unsigned long long int *) address;
    unsigned long long int old             = *address_as_ull, assumed;
    do {
        assumed = old;
        old     = atomicCAS( address_as_ull, assumed,
            __double_as_longlong( val + __longlong_as_double( assumed ) ) );
    } while ( assumed != old );
    return __longlong_as_double( old );
}
#endif

__device__ void d_compute_fact(double wx, int nFreq, double *dFreqGrid, double &fact1, double &fact2, int &ifreq, int loop, bool flag_occ)
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

__device__ void d_ssxDittt_kernel(int *inv_igp_index, int *indinv, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, double *vcoul, CustomComplex *I_eps_array, CustomComplex &ssxDittt, int ngpown, int ncouls, int n1,int ifreq, double fact1, double fact2)
{
    double ssxDittt_re = 0.00, ssxDittt_im = 0.00;
    for(int my_igp = 0; my_igp < ngpown; ++my_igp)
    {
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        CustomComplex ssxDit(0.00, 0.00);
        CustomComplex ssxDitt(0.00, 0.00);

        for(int ig = 0; ig < ncouls; ++ig)
        {
            ssxDit = I_eps_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] * fact1 + \
                                         I_eps_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] * fact2;

            ssxDitt += aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * ssxDit * vcoul[igp];
        }
        ssxDittt_re += CustomComplex_real(ssxDitt);
        ssxDittt_im += CustomComplex_imag(ssxDitt);
    }
    ssxDittt = CustomComplex (ssxDittt_re, ssxDittt_im);
}

__device__ void d_schDttt_corKernel1(CustomComplex &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex *I_epsR_array, CustomComplex *I_epsA_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
{
    int blkSize = 512;
    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00, \
        schDttt_re = 0.00, schDttt_im = 0.00;
    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            for(int ig = igbeg; ig < min(ncouls, igbeg+blkSize); ++ig)
            {
                int indigp = inv_igp_index[my_igp] ;
                int igp = indinv[indigp];
                CustomComplex sch2Dt = (I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig]) * fact1 + \
                                            (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2;
                CustomComplex sch2Dtt = aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];


                schDttt_re += CustomComplex_real(sch2Dtt) ;
                schDttt_im += CustomComplex_imag(sch2Dtt) ;
                schDttt_cor_re += CustomComplex_real(sch2Dtt) ;
                schDttt_cor_im += CustomComplex_imag(sch2Dtt) ;
            }
        }
    }
    schDttt_cor = CustomComplex (schDttt_cor_re, schDttt_cor_im);
    printf("From schDttt_corKernel1, schDttt_cor = \n");
    schDttt_cor.print();

}


__device__ void d_schDttt_corKernel2(CustomComplex &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex *I_epsR_array, CustomComplex *I_epsA_array, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
{
    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00;
    for(int my_igp = 0; my_igp < ngpown; ++my_igp)
    {
        for(int ig = 0; ig < ncouls; ++ig)
        {
            int indigp = inv_igp_index[my_igp] ;
            int igp = indinv[indigp];
            CustomComplex sch2Dt = ((I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ncouls*ngpown + my_igp*ncouls + ig]) * fact1 + \
                                        (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2) * -0.5;
            CustomComplex sch2Dtt = aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];
            schDttt_cor_re += CustomComplex_real(sch2Dtt) ;
            schDttt_cor_im += CustomComplex_imag(sch2Dtt) ;
        }
    }
    schDttt_cor = CustomComplex (schDttt_cor_re, schDttt_cor_im);
}

__global__ void achsDtemp_solver(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, CustomComplex *aqsntemp, CustomComplex *aqsmtemp, CustomComplex *I_epsR_array, double *vcoul, double *achsDtemp_re, double *achsDtemp_im, int numThreadsPerBlock)
{
    int n1 = blockIdx.x;
    int my_igp = blockIdx.y;
    int loopOverncouls=1, leftOverncouls=0;
    if(ncouls > numThreadsPerBlock)
    {
        loopOverncouls = ncouls / numThreadsPerBlock;
        leftOverncouls = ncouls % numThreadsPerBlock;
    }

    if( n1 < number_bands && my_igp < ngpown)
    {
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        CustomComplex schsDtemp(0.00, 0.00);

        for( int x = 0; x < loopOverncouls && threadIdx.x < numThreadsPerBlock ; ++x)
        { 
            int ig = x*numThreadsPerBlock + threadIdx.x;
            schsDtemp = schsDtemp - aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]* vcoul[ig] * 0.5;
        }
        if(leftOverncouls)
        {
            int ig = loopOverncouls*numThreadsPerBlock + threadIdx.x;
            schsDtemp = schsDtemp - aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]* vcoul[ig] * 0.5;
        }

        atomicAdd(achsDtemp_re, CustomComplex_real(schsDtemp));
        atomicAdd(achsDtemp_im, CustomComplex_imag(schsDtemp));
    }
}

__global__ void asxDtemp_solver(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, double *vcoul, CustomComplex *I_epsR_array, CustomComplex *I_epsA_array, double *asxDtemp_re, double *asxDtemp_im)
{
    CustomComplex ssxDittt(0.00, 0.00);
    int n1 = blockIdx.x;
    int iw = blockIdx.y;
    if(n1 < nvband && iw < nfreqeval)
    {
        double wx = freqevalmin - ekq[n1] + freqevalstep;
        double fact1 = 0.00, fact2 = 0.00;
        int ifreq = 0;
        CustomComplex ssxDittt(0.00, 0.00);

        d_compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 1, 0);

        if(wx > 0)
            d_ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);
        else
            d_ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsA_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);

        atomicAdd(&asxDtemp_re[iw], CustomComplex_real(ssxDittt * occ));
        atomicAdd(&asxDtemp_im[iw], CustomComplex_imag(ssxDittt * occ));
    }
}


__global__ void achDtemp_cor_solver(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, double *vcoul, CustomComplex *I_epsR_array, CustomComplex *I_epsA_array, CustomComplex *ach2Dtemp, double *achDtemp_cor_re, double *achDtemp_cor_im, CustomComplex *achDtemp_corb, int numThreadsPerBlock)
{
    bool flag_occ;
    int n1 = blockIdx.x;
    if(n1 < number_bands)
    {
        flag_occ = n1 < nvband;

        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            CustomComplex schDi_cor(0.00, 0.00);
            CustomComplex schDi_corb(0.00, 0.00);
            double wx = freqevalmin - ekq[n1] + freqevalstep;

            double fact1 = 0.00, fact2 = 0.00;
            int ifreq = 0.00;

            d_compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 2, flag_occ);

            if(wx > 0)
            {
                if(!flag_occ)
                    d_schDttt_corKernel1(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);
            }
            else if(flag_occ)
                d_schDttt_corKernel2(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);


//Summing up at the end of iw loop
//            ach2Dtemp[iw] += sch2Di[iw];
//            achDtemp_corb[iw] += schDi_corb[iw];
            atomicAdd2(&achDtemp_cor_re[iw], CustomComplex_real(schDi_cor));
            atomicAdd2(&achDtemp_cor_im[iw], CustomComplex_imag(schDi_cor));
//            achDtemp_cor_re[iw] += CustomComplex_real(schDi_cor);
//            achDtemp_cor_im[iw] += CustomComplex_imag(schDi_cor);

        }// iw
    } //n1
}


void d_achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, CustomComplex *aqsntemp, CustomComplex *aqsmtemp, CustomComplex *I_epsR_array, double *vcoul, double *achsDtemp_re, double *achsDtemp_im)
{
    dim3 numBlocks(number_bands, ngpown);
    int numThreadsPerBlock=32;

    achsDtemp_solver<<<numBlocks, numThreadsPerBlock>>>(number_bands, ngpown, ncouls, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, vcoul, achsDtemp_re, achsDtemp_im, numThreadsPerBlock); 
}

void d_asxDtemp_Kernel(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, double *vcoul, CustomComplex *I_epsR_array, CustomComplex *I_epsA_array, double *asxDtemp_re, double *asxDtemp_im)
{
    dim3 numBlocks(nvband, nfreqeval);
    int numThreadsPerBlock=1;

    asxDtemp_solver<<<numBlocks, numThreadsPerBlock>>>(nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, asxDtemp_re, asxDtemp_im);

}

void d_achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex *aqsmtemp, CustomComplex *aqsntemp, double *vcoul, CustomComplex *I_epsR_array, CustomComplex *I_epsA_array, CustomComplex *ach2Dtemp, double *achDtemp_cor_re, double *achDtemp_cor_im, CustomComplex *achDtemp_corb)
{
    dim3 numBlocks = number_bands;;
    int numThreadsPerBlock=1;

    achDtemp_cor_solver<<<numBlocks, numThreadsPerBlock>>>(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, ach2Dtemp, achDtemp_cor_re, achDtemp_cor_im, achDtemp_corb, numThreadsPerBlock);
}
