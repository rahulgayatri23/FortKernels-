#include <iostream>
#include <chrono>
#include <complex>

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

    cout << " number_bands = " << number_bands << \
        "\t nvband = " << nvband << \
        "\t ncouls = " << ncouls << \
        "\t ngpown = " << ngpown << \
        "\t nFreq = " << nFreq << \
        "\t nfreqeval = " << nfreqeval << endl;

    std::complex<double> expr0( 0.0 , 0.0);
    std::complex<double> expr( 0.5 , 0.5);
    std::complex<double> expR( 0.5 , 0.5);
    std::complex<double> expA( 0.5 , -0.5);

//Start to allocate the data structures;
    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls];
    double *vcoul = new double[ncouls];
    double *ekq = new double[number_bands];
    double *dFreqGrid = new double[nFreq];

    std::complex<double> aqsntemp[number_bands][ncouls];
    std::complex<double> aqsmtemp[number_bands][number_bands];
    std::complex<double> I_epsR_array[ncouls][ngpown][nFreq];
    std::complex<double> I_epsA_array[ncouls][ngpown][nFreq];
    std::complex<double> ssxDi[nfreqeval];
    std::complex<double> asxDtemp[nfreqeval];


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
    }

    for(int i=0; i<ncouls; ++i)
    {
        vcoul[i] = 1.00;
        for(int j=0; j<ngpown; ++j)
        {
            for(int k=0; k<number_bands; ++k)
            {
                I_epsR_array[i][j][k] = expR;
                I_epsA_array[i][j][k] = expA;
            }
        }
    }

    dw = 0.00;
    for(int ijk = 0; ijk < nFreq; ++ijk)
    {
        dFreqGrid[ijk] = dw;
        dw += 2.00;
    }

    for(int i = 0; i < nfreqeval; ++i)
    {
        ssxDi[i] = expr0;
        asxDtemp[i] = expr0;
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

            int igmax = ncouls;
            std::complex<double> schsDtemp = expr0;

            for(int ig = 0; ig < igmax; ++ig)
                schsDtemp -= aqsntemp[n1][ig] * std::conj(aqsmtemp[n1][ig]) * I_epsR_array[1][my_igp][ig];

            achsDtemp += schsDtemp * vcoul[igp] * 0.5;
        }

        std::complex<double> ssxDit = expr0;
        std::complex<double> ssxDitt = expr0;
        std::complex<double> ssxDittt = expr0;

        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep;
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
                    if(ifreq == 0) ifreq = nFreq + 3;
                }
                else
                {
                    int ifreq = 0;
                    for(int ijk = 0; ijk < nFreq-1; ++ijk)
                    {
                        if(-wx > dFreqGrid[ijk] && -wx < dFreqGrid[ijk+1])
                            ifreq = ijk;
                    }
                    if(ifreq == 0) ifreq = nFreq + 3;
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

                        for(int ig = 0; ig < igmax; ++ig)
                        {
                            ssxDit = I_epsR_array[ifreq][my_igp][ig] * fact1 + \
                                                         I_epsR_array[ifreq+1][my_igp][ig] * fact2;
        
                            ssxDitt += aqsntemp[n1][ig] * std::conj(aqsmtemp[n1][igp]) * ssxDit;
                        }
                        ssxDittt += ssxDitt * vcoul[igp];
                    }
                    ssxDi[iw] += ssxDittt;
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

                        for(int ig = 0; ig < igmax; ++ig)
                        {
                            ssxDit = I_epsR_array[ifreq][my_igp][ig] * fact1 + \
                                                         I_epsR_array[ifreq+1][my_igp][ig] * fact2;
        
                            ssxDitt += aqsntemp[n1][ig] * std::conj(aqsmtemp[n1][igp]) * ssxDit;
                        }
                    }
                    ssxDi[iw] += ssxDittt;
                }
            }

            if(flag_occ)
            {
                asxDtemp[iw] += ssxDi[iw] * occ;
            }
        } // iw
    } //n1

    cout << "achsDtemp = " << achsDtemp << endl;
    cout << "asxDtemp = " << asxDtemp[0] << endl;

    std::chrono::duration<double> elapsedTime = std::chrono::high_resolution_clock::now() - startTimer;
    cout << "********** Total Time Taken **********= " << elapsedTime.count() << " secs" << endl;

//Free the allocated memory since you are a good programmer :D
    free(inv_igp_index);
    free(indinv);
    free(vcoul);
    free(ekq);
    free(dFreqGrid);

    return 0;
}
