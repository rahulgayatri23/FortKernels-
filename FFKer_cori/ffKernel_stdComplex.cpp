#include <iostream>
#include <complex>
#include <omp.h>
#include <sys/time.h>

using namespace std;

double timeBetween(timeval startTimer, timeval endTimer)
{
    return ((endTimer.tv_sec - startTimer.tv_sec) +1e-6*(endTimer.tv_usec - startTimer.tv_usec));
}

inline void achsDtemp_kernel(int number_bands, int ngpown, int ncouls, std::complex<double> *aqsntemp, std::complex<double> *aqsmtemp, std::complex<double> *I_epsR_array, std::complex<double> *I_epsA_array, int *inv_igp_index, int *indinv, double *vcoul, std::complex<double> &achsDtemp)
{

    for(int n1 = 0; n1 < number_bands; ++n1)
    {
#pragma omp parallel for default(shared)
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            if(igp < ncouls && igp >= 0)
            {
                std::complex<double> schsDtemp(0.00, 0.00) ;

//#pragma ivdep
                for(int ig = 0; ig < ncouls; ++ig)
                    schsDtemp = schsDtemp - aqsntemp[n1*ncouls+ig] * std::conj(aqsmtemp[n1*ncouls+igp]) * I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig];

#pragma omp critical
                achsDtemp += schsDtemp * vcoul[igp] * 0.5;
            }
        } //ngpown
    } //number_bands
}


//output - fact1 and fact2
inline void compute_fact(double wx, int ifreq, double *dFreqGrid, double &fact1, double &fact2)
{
    if(wx > 0.00)
    {
        fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
        fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
    }
    else
    {
        fact1 = (dFreqGrid[ifreq+1] + wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
        fact2 = (-dFreqGrid[ifreq] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
    }
}

//output ssxDittt
inline void compute_ssxDittt(int ifreq, int n1, int ngpown, int ncouls, int *inv_igp_index, int *indinv, std::complex<double> *I_eps_array, std::complex<double> *aqsmtemp, std::complex<double> *aqsntemp, double *vcoul, double fact1, double fact2, std::complex<double> &ssxDittt)
{
#pragma omp parallel for default(shared)
    for(int my_igp = 0; my_igp < ngpown; ++my_igp)
    {
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        std::complex<double> ssxDitt(0.00, 0.00);
        if(igp < ncouls && igp >= 0)
        {
            for(int ig = 0; ig < ncouls; ++ig)
            {
                std::complex<double> ssxDit = I_eps_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] * fact1 + \
                                             I_eps_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] * fact2;

                ssxDitt += aqsntemp[n1*ncouls+ig] * std::conj(aqsmtemp[n1*ncouls+igp]) * ssxDit;
            }
#pragma omp critical
            ssxDittt += ssxDitt * vcoul[igp];
        }
    }
}

//output - asxDtemp
inline void asxDtemp_kernel(int nvband, int number_bands, int ngpown, int ncouls, int nfreqeval, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, std::complex<double> *aqsntemp, std::complex<double> *aqsmtemp, std::complex<double> *I_epsR_array, std::complex<double> *I_epsA_array, int *inv_igp_index, int *indinv, double *vcoul, std::complex<double> *asxDtemp)
{
    const double occ = 1;
    std::complex<double> expr0( 0.0 , 0.0);
    std::complex<double> ssxDittt( 0.0 , 0.0);
    double fact1 = 0.00, fact2 = 0.00;

    for(int n1 = 0; n1 < nvband; ++n1)
    {
        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            double wx = freqevalmin - ekq[n1] + freqevalstep;
            std::complex<double> ssxDi(0.00, 0.00);

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
            if(ifreq == 0) ifreq = nFreq-2;

            compute_fact(wx, ifreq, dFreqGrid, fact1, fact2);
            ssxDittt = expr0;

            if(wx > 0.00)
                compute_ssxDittt(ifreq, n1, ngpown, ncouls, inv_igp_index, indinv, I_epsR_array, aqsmtemp, aqsntemp, vcoul, fact1, fact2, ssxDittt);
            else
                compute_ssxDittt(ifreq, n1, ngpown, ncouls, inv_igp_index, indinv, I_epsA_array, aqsmtemp, aqsntemp, vcoul, fact1, fact2, ssxDittt);

            ssxDi+= ssxDittt;
            asxDtemp[iw] += ssxDi* occ;
        } // iw
    } //nvband
}

inline void compute_schDi_cor(bool flag_occ, int n1, int ifreq, int nfreqeval, int nFreq, double freqevalstep, double freqevalmin, double *ekq, double *dFreqGrid, double cedifft_zb, std::complex<double> *schDt_matrix, double pref_zb, std::complex<double> *schDi_cor, std::complex<double> *sch2Di)
{
    std::complex<double> schDt = schDt_matrix[n1*nFreq + ifreq];
    double cedifft_zb_right = cedifft_zb;
    double cedifft_zb_left = dFreqGrid[ifreq-1];
    std::complex<double> schDt_right = schDt;
    std::complex<double> schDt_left = schDt_matrix[n1*nFreq + ifreq-1];
    std::complex<double> schDt_avg = (schDt_right + schDt_left) * 0.5;
    std::complex<double> schDt_lin = schDt_right - schDt_left;
    std::complex<double> schDt_lin2 = schDt_lin / (cedifft_zb_right - cedifft_zb_left);
    std::complex<double> schDt_lin3(0.00, 0.00);

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


inline void compute_schDi_corb(int n1, int ifreq, int nfreqeval, double freqevalmin, double freqevalstep, double *ekq, std::complex<double> schDt, std::complex<double> cedifft_coh, std::complex<double> cedifft_cor, double *pref, std::complex<double> *schDi, std::complex<double> *schDi_corb)
{
    for(int iw = 0; iw < nfreqeval; ++iw)
    {
        double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep;
        std::complex<double> tmp(0.00, pref[ifreq]);
        schDi[iw] -= tmp*schDt / (wx - cedifft_coh);
        schDi_corb[iw] = schDi_corb[iw] - tmp*schDt / (wx-cedifft_cor);
    }
}

inline void compute_schDi(bool flag_occ, int n1, int nfreqeval, int nFreq, double freqevalmin, double freqevalstep, double *ekq, std::complex<double> *schDt_matrix, double *dFreqGrid, std::complex<double> *dFreqBrd, double *pref, double pref_zb, std::complex<double> *schDi, std::complex<double> *schDi_corb, std::complex<double> *schDi_cor, std::complex<double> *sch2Di)
{
#pragma omp parallel for default(shared)
    for(int ifreq = 0; ifreq < nFreq; ++ifreq)
    {
        double cedifft_zb = dFreqGrid[ifreq];
        double cedifft_zb_right, cedifft_zb_left;
        std::complex<double> cedifft_cor(0.00, 0.00);
        std::complex<double> cedifft_compl(cedifft_zb, 0.00);
        std::complex<double> cedifft_coh = cedifft_compl - dFreqBrd[ifreq];

        if(flag_occ)
            cedifft_cor = -1 * cedifft_compl - dFreqBrd[ifreq];
            else
                cedifft_cor = cedifft_compl - dFreqBrd[ifreq];

        if(ifreq != 0)
            compute_schDi_cor(flag_occ, n1, ifreq, nfreqeval, nFreq, freqevalstep, freqevalmin, ekq, dFreqGrid, cedifft_zb, schDt_matrix, pref_zb, schDi_cor, sch2Di);
        else
            compute_schDi_corb(n1, ifreq, nfreqeval, freqevalmin, freqevalstep, ekq, schDt_matrix[n1*nFreq + ifreq], cedifft_coh, cedifft_cor, pref, schDi, schDi_corb);
    }
}

inline void compute_fact2(bool flag_occ, double wx, int ifreq, double *dFreqGrid, double &fact1, double &fact2)
{
    if(wx > 0)
    {
        fact1 = -0.5 * (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
        fact2 = -0.5 * (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
    }
    else if(flag_occ)
    {
        fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
        fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
    }
}

inline void compute_sch2Dtt(bool flag_occ, int ncouls, int ngpown, int ifreq, int n1, int my_igp, int igp, double wx, std::complex<double> *I_epsR_array, std::complex<double> *I_epsA_array, std::complex<double> *aqsmtemp, std::complex<double> *aqsntemp, double *vcoul, double fact1, double fact2, std::complex<double> &sch2Dtt, std::complex<double> &schDttt)
{
    if(wx>0)
    {
        for(int ig = 0; ig < ncouls; ++ig)
        {
            std::complex<double> sch2Dt = (I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig]) * fact1 + \
                                        (I_epsR_array[(ifreq+1)*ngpown*ncouls+ my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls+ my_igp*ncouls + ig]) * fact2;
            sch2Dtt = sch2Dtt + aqsntemp[n1*ncouls+ig] * std::conj(aqsmtemp[n1*ncouls+igp]) * sch2Dt;
        }
        schDttt += sch2Dtt * vcoul[igp];
    }
//else if(flag_occ)
//    {
//        for(int ig = 0; ig < ncouls; ++ig)
//        {
//            std::complex<double> sch2Dt = -0.5*((I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig]) * fact1 + \
//                                        (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp * ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2);
//            sch2Dtt += aqsntemp[n1*ncouls+ig] * std::conj(aqsmtemp[n1*ncouls+igp]) * sch2Dt;
//        }
//    }
}

void compute_schDi_cor2( int iw, int n1, int nfreqeval, int ngpown, int ncouls, int nFreq, bool flag_occ, double wx, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, std::complex<double> *I_epsR_array, std::complex<double> *I_epsA_array, std::complex<double> *aqsmtemp, std::complex<double> *aqsntemp, double *vcoul, std::complex<double> *schDi_cor, std::complex<double> *sch2Di)
{
    double fact1 = 0.00, fact2 = 0.00;
    std::complex<double> schDttt(0.00, 0.00);
    std::complex<double> schDttt_cor(0.00, 0.00);

    if(wx > 0)
    {
        int ifreq = -1;
        for(int ijk = 0; ijk < nFreq-1; ++ijk)
        {
            if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                ifreq = ijk;
        }
        if(ifreq == -1) ifreq = nFreq-2;


        compute_fact2(flag_occ, wx, ifreq, dFreqGrid, fact1, fact2);


#pragma omp parallel for default(shared)
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp] ;
            int igp = indinv[indigp];

            std::complex<double> sch2Dtt(0.00, 0.00);
            if(igp < ncouls && igp >= 0)
            {
                compute_sch2Dtt(flag_occ, ncouls, ngpown, ifreq, n1, my_igp, igp, wx, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul, fact1, fact2, sch2Dtt, schDttt);
                if(flag_occ){}
                else
                {
#pragma omp critical
                    schDttt_cor += sch2Dtt * vcoul[igp];
                }
            }
        }

        sch2Di[iw] += schDttt;
    } //if-loop
    else if(flag_occ)
    {
        wx = -wx; int ifreq = 0;
        for(int ijk = 0; ijk < nFreq-1; ++ijk)
        {
            if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                ifreq = ijk;
        }
        if(ifreq == 0) ifreq = nFreq-2;

//                compute_fact2(flag_occ, wx, ifreq, dFreqGrid, fact1, fact2);
        fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 
        fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]); 

        std::complex<double> schDttt_cor(0.00, 0.00);
#pragma omp parallel for default(shared)
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp] ;
            int igp = indinv[indigp];
            std::complex<double> sch2Dtt(0.00, 0.00);

            if(igp < ncouls && igp >= 0)
            {
//                compute_sch2Dtt(flag_occ, ncouls, ngpown, ifreq, n1, my_igp, igp, wx, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul, fact1, fact2, sch2Dtt, schDttt);
                for(int ig = 0; ig < ncouls; ++ig)
                {
                    std::complex<double> sch2Dt = -0.5*((I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig]) * fact1 + \
                                                (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp * ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2);
                    sch2Dtt += aqsntemp[n1*ncouls+ig] * std::conj(aqsmtemp[n1*ncouls+igp]) * sch2Dt;
                }
#pragma omp critical
                schDttt_cor += sch2Dtt * vcoul[igp];
            }
        }
        schDi_cor[iw] += schDttt_cor;
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

    const int number_bands = atoi(argv[1]);
    const int nvband = atoi(argv[2]);
    const int ncouls = atoi(argv[3]);
    const int ngpown = atoi(argv[4]);
    const int nFreq = atoi(argv[5]);
    const int nfreqeval = atoi(argv[6]);

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

    long double mem_alloc = 0.00;

    std::complex<double> *aqsntemp = new std::complex<double>[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(std::complex<double>));

    std::complex<double> *aqsmtemp = new std::complex<double>[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(std::complex<double>));

    std::complex<double> *I_epsR_array = new std::complex<double>[nFreq * ngpown * ncouls];
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(std::complex<double>));

    std::complex<double> *I_epsA_array = new std::complex<double>[nFreq * ngpown * ncouls];
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(std::complex<double>));

    std::complex<double> *schDt_matrix = new std::complex<double>[number_bands * nFreq];
    mem_alloc += (nFreq * number_bands * sizeof(std::complex<double>));

    std::complex<double> *schDi = new std::complex<double>[nfreqeval];
    std::complex<double> *sch2Di = new std::complex<double>[nfreqeval];
    std::complex<double> *schDi_cor = new std::complex<double>[nfreqeval];
    std::complex<double> *schDi_corb = new std::complex<double>[nfreqeval];
    std::complex<double> *ach2Dtemp = new std::complex<double>[nfreqeval];
    std::complex<double> *achDtemp = new std::complex<double>[nfreqeval];
    std::complex<double> *achDtemp_cor = new std::complex<double>[nfreqeval];
    std::complex<double> *achDtemp_corb = new std::complex<double>[nfreqeval];
    std::complex<double> *asxDtemp = new std::complex<double>[nfreqeval];
    std::complex<double> *dFreqBrd = new std::complex<double>[nFreq];
    mem_alloc += (nfreqeval * 10 * sizeof(std::complex<double>));
    mem_alloc += (nFreq * sizeof(std::complex<double>)) ;

    std::complex<double> achsDtemp(0.00, 0.00);


    //Variables used : 
    const double freqevalmin = 0.00;
    const double freqevalstep = 0.50;
    const double occ = 1.00;
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
                I_epsR_array[i*ngpown*ncouls + j*ncouls + k] = expR;
                I_epsA_array[i*ngpown*ncouls + j*ncouls + k] = expA;
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
    cout << "starting loop" << endl;


    //All timer variables
    timeval achsDtemp_startTimer, achsDtemp_endTimer, \
        asxDtemp_startTimer, asxDtemp_endTimer, 
        achDtemp_startTimer, achDtemp_endTimer, 
        kernel_startTimer, kernel_endTimer;


    //reduction on achsDtemp
    gettimeofday(&kernel_startTimer, NULL);

    gettimeofday(&achsDtemp_startTimer, NULL);
    //kernel
    achsDtemp_kernel(number_bands, ngpown, ncouls, aqsntemp, aqsmtemp, I_epsR_array, I_epsA_array, inv_igp_index, indinv, vcoul, achsDtemp);
    gettimeofday(&achsDtemp_endTimer, NULL);

    gettimeofday(&asxDtemp_startTimer, NULL);
    //kernel
    asxDtemp_kernel(nvband, number_bands, ngpown, ncouls, nfreqeval, nFreq, freqevalmin, freqevalstep, ekq, dFreqGrid, aqsntemp, aqsmtemp, I_epsR_array, I_epsA_array, inv_igp_index, indinv, vcoul, asxDtemp);
    gettimeofday(&asxDtemp_endTimer, NULL);


//    /******************************Done with the First Part of the Code*****************************************************************************/
    gettimeofday(&achDtemp_startTimer, NULL);


    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        bool flag_occ = n1 < nvband;

        //resetting all temp arrays to 0's
        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            schDi[iw] = expr0;
            sch2Di[iw] = expr0;
            schDi_corb[iw] = expr0;
            schDi_cor[iw] = expr0;
        }

        compute_schDi(flag_occ, n1, nfreqeval, nFreq, freqevalmin, freqevalstep, ekq, schDt_matrix, dFreqGrid, dFreqBrd, pref, pref_zb, schDi, schDi_corb, schDi_cor, sch2Di);

        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            double wx = freqevalmin - ekq[n1] + freqevalstep;

            compute_schDi_cor2( iw, n1, nfreqeval, ngpown, ncouls, nFreq, flag_occ, wx, freqevalmin, freqevalstep, ekq, dFreqGrid, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul, schDi_cor, sch2Di);

//Summing up at the end of iw loop
            achDtemp[iw] += schDi[iw];
            ach2Dtemp[iw] += sch2Di[iw];
            achDtemp_cor[iw] += schDi_cor[iw];
            achDtemp_corb[iw] += schDi_corb[iw];
        }// iw
    } //n1
    gettimeofday(&achDtemp_endTimer, NULL);
    gettimeofday(&kernel_endTimer, NULL);

    double elapsed_achsDtemp = timeBetween(achsDtemp_startTimer, achsDtemp_endTimer);
    double elapsed_asxDtemp = timeBetween(asxDtemp_startTimer, asxDtemp_endTimer);
    double elapsed_achDtemp = timeBetween(achDtemp_startTimer, achDtemp_endTimer);
    double elapsed_kernelTime = timeBetween(kernel_startTimer, kernel_endTimer);

    std::cout << "Time taken by achsDtemp_kernel = " << elapsed_achsDtemp << " secs" << std::endl;
    std::cout << "Time taken by asxDtemp_kernel = " << elapsed_achsDtemp << " secs" << std::endl;
    std::cout << "Time taken by achDtemp_kernel = " << elapsed_achDtemp << " secs" << std::endl;
    std::cout << "Time taken by kernel = " << elapsed_kernelTime << " secs" << std::endl;

    cout << "achsDtemp = " << achsDtemp << endl;
    cout << "asxDtemp = " << asxDtemp[0] << endl;
    cout << "achDtemp_cor = " << achDtemp_cor[0] << endl;

//Free the allocated memory since you are a good programmer :D
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
