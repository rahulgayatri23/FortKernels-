#include <iostream>
#include <chrono>
#include <complex>
#include <omp.h>
#include "GPUComplex.h"

using namespace std;

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}

void compute_ssxDitt(GPUComplex *I_eps_array, double fact1, double fact2, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, int ifreq, int ngpown, int ncouls, int my_igp, int ig, int n1, GPUComplex &ssxDitt, int igp)
{
    GPUComplex ssxDit(0.00, 0.00);
    ssxDit = GPUComplex_plus(GPUComplex_mult(I_eps_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] , fact1 ) , \
                                 GPUComplex_mult(I_eps_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] , fact2));

    ssxDitt += GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_product(GPUComplex_conj(aqsmtemp[n1*ncouls + igp]) , ssxDit));
}

void asxDtemp_solver(GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, int ncouls, int ngpown, int nfreqeval, double *vcoul, int *inv_igp_index, int *indinv, double fact1, double fact2, double wx, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *asxDtemp, int ifreq, int n1, double occ, GPUComplex *ssxDi)
{
    GPUComplex ssxDittt(0.00, 0.00);
    for(int iw = 0; iw < nfreqeval; ++iw)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            GPUComplex ssxDitt(0.00, 0.00);

            for(int ig = 0; ig < ncouls; ++ig)
            {
                GPUComplex ssxDit(0.00, 0.00);
                if(wx > 0)
                    compute_ssxDitt(I_epsR_array, fact1, fact2, aqsntemp, aqsmtemp, ifreq, ngpown, ncouls, my_igp, ig, n1, ssxDitt, igp);
                else
                    compute_ssxDitt(I_epsA_array, fact1, fact2, aqsntemp, aqsmtemp, ifreq, ngpown, ncouls, my_igp, ig, n1, ssxDitt, igp);
            }
            ssxDittt += GPUComplex_mult(ssxDitt , vcoul[igp]);
        }
        ssxDi[iw] += ssxDittt;
        asxDtemp[iw] = GPUComplex_mult(ssxDittt, occ);
    }
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

    GPUComplex *schDt_matrix = new GPUComplex[number_bands * nFreq];
    mem_alloc += (nFreq * number_bands * sizeof(GPUComplex));

    GPUComplex *achsDtemp = new GPUComplex;
    
    //Allocate cuda memory on GPU
    GPUComplex *d_schDt_matrix, *d_aqsmtemp, *d_aqsntemp, *d_I_epsR_array, *d_I_epsA_array, *d_ssxDi, *d_schDi, *d_sch2Di, *d_schDi_cor, \
                *d_schDi_corb, *d_achDtemp, *d_ach2Dtemp, *d_achDtemp_cor, *d_achDtemp_corb, *d_asxDtemp, *d_dFreqBrd, *d_achsDtemp;

    int *d_indinv, *d_inv_igp_index;
    double *d_vcoul, *d_dFreqGrid;

    //Create data-structs on the device
#if CUDA
    CudaSafeCall(cudaMalloc((void**) &d_achsDtemp, sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_schDt_matrix, number_bands*nFreq*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsmtemp, number_bands*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsntemp, number_bands*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_I_epsR_array, nFreq*ngpown*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_I_epsA_array, nFreq*ngpown*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_ssxDi, nfreqeval*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_schDi, nfreqeval*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_sch2Di, nfreqeval*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_schDi_cor, nfreqeval*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_schDi_corb, nfreqeval*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_achDtemp, nfreqeval*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_ach2Dtemp, nfreqeval*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_achDtemp_cor, nfreqeval*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_achDtemp_corb, nfreqeval*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_asxDtemp, nfreqeval*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_dFreqBrd, nFreq*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_indinv, ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_inv_igp_index, ngpown*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_vcoul, ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_dFreqGrid, nFreq*sizeof(GPUComplex)));
#endif

    //Variables used : 

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

    cout << "Memory Used = " << mem_alloc/(1024 * 1024 * 1024) << " GB" << endl;
    std::chrono::duration<double> elapsedTime_preloop = std::chrono::high_resolution_clock::now() - startTimer;

    //Copy to Device
#if CUDA
    printf("Running the CUDA version of asxDtemp_kernel\n");
    CudaSafeCall(cudaMemcpy(d_schDt_matrix, schDt_matrix, number_bands*nFreq*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsntemp, aqsntemp, number_bands*ncouls*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsmtemp, aqsmtemp, number_bands*ncouls*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsntemp, aqsntemp, number_bands*ncouls*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_I_epsR_array, I_epsR_array, nFreq*ngpown*ncouls*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_I_epsA_array, I_epsA_array, nFreq*ngpown*ncouls*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_ssxDi, ssxDi, nfreqeval*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_schDi, schDi, nfreqeval*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_sch2Di, sch2Di, nfreqeval*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_schDi_cor, schDi_cor, nfreqeval*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_schDi_corb, schDi_corb, nfreqeval*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achDtemp, achDtemp, nfreqeval*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_ach2Dtemp, ach2Dtemp, nfreqeval*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achDtemp_cor, achDtemp_cor, nfreqeval*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achDtemp_corb, achDtemp_corb, nfreqeval*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_asxDtemp, asxDtemp, nfreqeval*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_dFreqBrd, dFreqBrd, nfreqeval*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_indinv, indinv, ncouls*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_inv_igp_index, inv_igp_index, ngpown*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_vcoul, vcoul, ncouls*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_dFreqGrid, dFreqGrid, nFreq*sizeof(double), cudaMemcpyHostToDevice));
#else
    printf("Running the CPU version of asxDtemp_kernel\n");
#endif

    cout << "starting loop" << endl;
    auto startTimer_firstloop = std::chrono::high_resolution_clock::now();
    auto startTimer_kernel = std::chrono::high_resolution_clock::now();


    achsDtemp_kernel(d_achsDtemp, d_aqsntemp, d_aqsmtemp, d_I_epsR_array, d_inv_igp_index, d_indinv, d_vcoul, number_bands, ncouls, ngpown);
    CudaSafeCall(cudaMemcpy(achsDtemp, d_achsDtemp, sizeof(GPUComplex), cudaMemcpyDeviceToHost));


    for(int n1 = 0; n1 < nvband; ++n1)
    {
        double occ = 1.00, fact1 = 0.00, fact2 = 0.00;
        GPUComplex ssxDit = expr0;
        double wx = freqevalmin - ekq[n1] + freqevalstep;
        int ifreq = 0;
        if(wx > 0.00)
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
        else
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

#if CUDA
        asxDtemp_kernel(d_I_epsR_array, d_I_epsA_array, ncouls, ngpown,nfreqeval, d_vcoul, d_inv_igp_index, d_indinv, fact1, fact2, wx, d_aqsmtemp, d_aqsntemp, d_asxDtemp, ifreq, n1, d_ssxDi, occ);
#else 
        asxDtemp_solver(I_epsR_array, I_epsA_array, ncouls, ngpown, nfreqeval, vcoul, inv_igp_index, indinv, fact1, fact2, wx, aqsmtemp, aqsntemp, asxDtemp, ifreq, n1, occ, ssxDi);
#endif
    }
#if CUDA
    cudaDeviceSynchronize();
    CudaSafeCall(cudaMemcpy(asxDtemp, d_asxDtemp, nfreqeval*sizeof(GPUComplex), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(ssxDi, d_ssxDi, nfreqeval*sizeof(GPUComplex), cudaMemcpyDeviceToHost));
#endif


    std::chrono::duration<double> elapsedTime_firstloop = std::chrono::high_resolution_clock::now() - startTimer_firstloop;

//    /******************************Done with the First Part of the Code*****************************************************************************/

    auto startTimer_secondloop = std::chrono::high_resolution_clock::now();

//    for(int n1 = 0; n1 < number_bands; ++n1)
//    {
//        bool flag_occ = n1 < nvband;
//        double occ = 1.00;
//        for(int iw = 0; iw < nfreqeval; ++iw)
//        {
//            schDi[iw] = expr0;
//            sch2Di[iw] = expr0;
//            schDi_corb[iw] = expr0;
//            schDi_cor[iw] = expr0;
//        }
//
//#pragma omp parallel for default(shared)
//        for(int ifreq = 0; ifreq < nFreq; ++ifreq)
//        {
//            GPUComplex schDt = schDt_matrix[n1*nFreq + ifreq];
//            double cedifft_zb = dFreqGrid[ifreq];
//            double cedifft_zb_right, cedifft_zb_left;
//            GPUComplex schDt_right, schDt_left, schDt_avg, schDt_lin, schDt_lin2, schDt_lin3;
//            GPUComplex cedifft_compl(cedifft_zb, 0.00);
//            GPUComplex cedifft_cor;
//            GPUComplex cedifft_coh = cedifft_compl - dFreqBrd[ifreq];
//
//            if(flag_occ)
//                cedifft_cor = GPUComplex_mult(cedifft_compl, -1) - dFreqBrd[ifreq];
//                else
//                    cedifft_cor = cedifft_compl - dFreqBrd[ifreq];
//
//            if(ifreq != 0)
//            {
//                cedifft_zb_right = cedifft_zb;
//                cedifft_zb_left = dFreqGrid[ifreq-1];
//                schDt_right = schDt;
//                schDt_left = schDt_matrix[n1*nFreq + ifreq-1];
//                schDt_avg = GPUComplex_mult((schDt_right + schDt_left) , 0.5);
//                schDt_lin = schDt_right - schDt_left;
//                schDt_lin2 = GPUComplex_divide(schDt_lin , (cedifft_zb_right - cedifft_zb_left));
//            }
//
//            if(ifreq != nFreq)
//            {
//                for(int iw = 0; iw < nfreqeval; ++iw)
//                {
//                    double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep;
//                    GPUComplex tmp(0.00, pref[ifreq]);
//                    schDi[iw] = schDi[iw] - GPUComplex_divide(GPUComplex_product(tmp,schDt) , doubleMinusGPUComplex(wx, cedifft_coh));
//                }
//            }
//
//            if(ifreq != 0)
//            {
//                for(int iw = 0; iw < nfreqeval; ++iw)
//                {
//                    double intfact = (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_left);
//                    if(intfact < 0.0001) intfact = 0.0001;
//                    if(intfact > 10000) intfact = 10000;
//                    intfact = -log(intfact);
//                    GPUComplex pref_zb_compl(0.00, pref_zb);
//                    sch2Di[iw] = sch2Di[iw] - GPUComplex_mult(GPUComplex_product(pref_zb_compl , schDt_avg) , intfact);
//                    if(flag_occ)
//                    {
//                        intfact = abs((freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_left));
//                        if(intfact < 0.0001) intfact = 0.0001;
//                        if(intfact > 10000) intfact = 10000;
//                        intfact = log(intfact);
//                        schDt_lin3 = GPUComplex_mult((schDt_left + schDt_lin2) , (-freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact) ;
//                    }
//                    else
//                        schDt_lin3 = GPUComplex_mult((schDt_left + schDt_lin2) , (freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact);
//
//                    schDt_lin3 += schDt_lin;
//                    schDi_cor[iw] = schDi_cor[iw] -  GPUComplex_product(pref_zb_compl , schDt_lin3);
//                }
//            }
//        }
//
//        for(int iw = 0; iw < nfreqeval; ++iw)
//        {
//            double wx = freqevalmin - ekq[n1] + freqevalstep;
//            GPUComplex schDttt_cor = expr0;
//            for(int i = 0; i < numThreads; ++i)
//                schDttt_cor_threadArr[i] = expr0;
//
//            if(wx > 0)
//            {
//                int ifreq = -1;
//                for(int ijk = 0; ijk < nFreq-1; ++ijk)
//                {
//                    if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
//                        ifreq = ijk;
//                }
//                if(ifreq == -1) ifreq = nFreq-2;
//
//                double fact1 = -0.5 * (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
//                double fact2 = -0.5 * (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
//
//                GPUComplex schDttt = expr0;
//
////#pragma omp parallel for default(shared)
//                for(int my_igp = 0; my_igp < ngpown; ++my_igp)
//                {
//                    int tid = omp_get_thread_num();
//                    int indigp = inv_igp_index[my_igp] ;
//                    int igp = indinv[indigp];
//                    int igmax = ncouls;
//                    GPUComplex sch2Dtt(0.00, 0.00);
//
//                    for(int ig = 0; ig < igmax; ++ig)
//                    {
//                        GPUComplex sch2Dt = GPUComplex_mult((GPUComplex_minus(I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig])) , fact1) + \
//                                                    GPUComplex_mult((GPUComplex_minus(I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig])) , fact2);
//                        sch2Dtt += GPUComplex_product(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_conj(aqsmtemp[n1*ncouls + igp])) , sch2Dt);
//                    }
//                    schDttt += GPUComplex_mult(sch2Dtt , vcoul[igp]);
//                    if(flag_occ){}
//                    else
//                        schDttt_cor += GPUComplex_mult(sch2Dtt , vcoul[igp]);
////                        schDttt_cor_threadArr[tid] += GPUComplex_mult(sch2Dtt , vcoul[igp]);
//                }
//                sch2Di[iw] += schDttt;
//            }
//            else if(flag_occ)
//            {
//                wx = -wx; int ifreq = 0;
//                for(int ijk = 0; ijk < nFreq-1; ++ijk)
//                {
//                    if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
//                        ifreq = ijk;
//                }
//                if(ifreq == 0) ifreq = nFreq-2;
//
//                double fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
//                double fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
//
////#pragma omp parallel for default(shared)
//                for(int my_igp = 0; my_igp < ngpown; ++my_igp)
//                {
//                    int tid = omp_get_thread_num();
//                    int indigp = inv_igp_index[my_igp] ;
//                    int igp = indinv[indigp];
//                    int igmax = ncouls;
//                    GPUComplex sch2Dtt(0.00, 0.00);
//
//                    for(int ig = 0; ig < igmax; ++ig)
//                    {
//                        GPUComplex sch2Dt = GPUComplex_mult(GPUComplex_mult((GPUComplex_minus(I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[ifreq*ncouls*ngpown + my_igp*ncouls + ig])) , fact1) + \
//                                                    GPUComplex_mult((GPUComplex_minus(I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig], I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig])) , fact2), -0.5);
//                        sch2Dtt += GPUComplex_product(GPUComplex_product(aqsntemp[n1*ncouls + ig] , GPUComplex_conj(aqsmtemp[n1*ncouls + igp])) , sch2Dt);
//                    }
//                    schDttt_cor += GPUComplex_mult(sch2Dtt , vcoul[igp]);
//                    schDttt_cor_threadArr[tid] += GPUComplex_mult(sch2Dtt , vcoul[igp]);
//                }
//            }
//
////            for(int i = 0; i < numThreads; ++i)
////                schDttt_cor += schDttt_cor_threadArr[i];
//
//            schDi_cor[iw] += schDttt_cor;
//
////Summing up at the end of iw loop
//            achDtemp[iw] += schDi[iw];
//            ach2Dtemp[iw] += sch2Di[iw];
//            achDtemp_cor[iw] += schDi_cor[iw];
//            achDtemp_corb[iw] += schDi_corb[iw];
//        }// iw
//    } //n1
    std::chrono::duration<double> elapsedTime_secondloop = std::chrono::high_resolution_clock::now() - startTimer_secondloop;

    cout << "achsDtemp = " ;
    achsDtemp->print();
//    cout << "asxDtemp = " ;
//    asxDtemp[0].print();
    cout << "ssxDi = " ;
    ssxDi[0].print();
//    cout << "achDtemp_cor = " ;
//    achDtemp_cor[0].print();
//
//    std::chrono::duration<double> elapsedTime = std::chrono::high_resolution_clock::now() - startTimer;
//    std::chrono::duration<double> elapsedKernelTime = std::chrono::high_resolution_clock::now() - startTimer_kernel;
//    cout << "********** PreLoop **********= " << elapsedTime_preloop.count() << " secs" << endl;
//    cout << "********** Kenel Time **********= " << elapsedKernelTime.count() << " secs" << endl;
////    cout << "********** FirtLoop **********= " << elapsedTime_firstloop.count() << " secs" << endl;
////    cout << "********** SecondLoop  **********= " << elapsedTime_secondloop.count() << " secs" << endl;
//    cout << "********** Total Time Taken **********= " << elapsedTime.count() << " secs" << endl;

//Free the allocated memory since you are a good programmer :D
    cudaFree(d_schDt_matrix);
    cudaFree(d_aqsntemp);
    cudaFree(d_aqsmtemp);
    cudaFree(d_I_epsR_array);
    cudaFree(d_I_epsA_array);
    cudaFree(d_ssxDi);
    cudaFree(d_schDi);
    cudaFree(d_sch2Di);
    cudaFree(d_schDi_cor);
    cudaFree(d_schDi_corb);
    cudaFree(d_achDtemp);
    cudaFree(d_ach2Dtemp);
    cudaFree(d_achDtemp_cor);
    cudaFree(d_achDtemp_corb);
    cudaFree(d_asxDtemp);
    cudaFree(d_dFreqBrd);

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
