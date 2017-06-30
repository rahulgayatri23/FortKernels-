#include <iostream>
#include <cstdlib>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>

using namespace std;
int debug = 0;

int main(int argc, char** argv)
{

    if (argc != 6)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> <gppsum> " << endl;
        exit (0);
    }
    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int nodes_per_group = atoi(argv[4]);
    int gppsum = atoi(argv[5]);

//    int igmax = ncouls;

    int tid, NTHREADS; // OpenMP related threading variables.


    int npes = 1; //Represents the number of ranks per node
    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

    long double e_lk = 10;
    long double dw = 1;
    int nstart = 0, nend = 3;

    int inv_igp_index[ngpown];
    int indinv[ncouls];


    long double to1 = 1e-6;

    long double gamma = 0.5;
    long double sexcut = 4.0;
    long double limitone = 1.0/(to1*4.0);
    long double limittwo = pow(0.5,2);

    long double e_n1kq= 6.0; //This in the fortran code is derived through the double dimenrsion array ekq whose 2nd dimension is 1 and all the elements in the array have the same value


    //Printing out the params passed.
    std::cout << "number_bands = " << number_bands \
        << "\t nvband = " << nvband \
        << "\t ncouls = " << ncouls \
        << "\t nodes_per_group  = " << nodes_per_group \
        << "\t gppsum = " << gppsum \
        << "\t ngpown = " << ngpown \
        << "\t nend = " << nend \
        << "\t nstart = " << nstart \
        << "\t gamma = " << gamma \
        << "\t sexcut = " << sexcut \
        << "\t limitone = " << limitone \
        << "\t limittwo = " << limittwo << endl;


    //ALLOCATE statements from fortran gppkernel.

    std::complex<long double> expr0 = 0.0 + 0.0i;
    std::complex<long double> expr ( 0.5 , 0.5);
    std::complex<long double> aqsmtemp[ncouls][number_bands];
    std::complex<long double> aqsntemp[ncouls][number_bands];
    std::complex<long double> I_eps_array[ncouls][ngpown];
    std::complex<long double> achtemp[nend-nstart];
    std::complex<long double> asxtemp[nend-nstart];
    std::complex<long double> acht_n1_loc[number_bands];

    long double vcoul[ncouls];
    std::complex<long double> wtilde_array[ncouls][ngpown];
    long double wx_array[3];

    std::complex<long double> schstemp = std::complex<long double>(0.0, 0.0);
    std::complex<long double> schs = std::complex<long double>(0.0, 0.0);
    std::complex<long double> matngmatmgp = std::complex<long double>(0.0, 0.0);
    std::complex<long double> matngpmatmg = std::complex<long double>(0.0, 0.0);

    std::complex<long double> achstemp = std::complex<long double>(0.0, 0.0);
    std::complex<long double> wtilde2, Omega2;
    std::complex<long double> halfinvwtilde, delw, ssx, sch, wdiff, cden , eden, mygpvar1, mygpvar2;
    long double ssxcutoff;
    std::complex<long double> ssx_array[3], \
        sch_array[3], \
        ssxa[ncouls], \
        scha[ncouls], \
        scht, ssxt, wtilde;

    long double wxt, delw2, delwr, wdiffr, scha_mult, rden;
    long double occ=1.0;
    bool flag_occ;


   for(int i=0; i<ncouls; i++)
   {
       for(int j=0; j<number_bands; j++)
       {
           aqsmtemp[i][j] = expr;
           aqsntemp[i][j] = expr;
       }

       for(int j=0; j<ngpown; j++)
       {
           I_eps_array[i][j] = expr;
           wtilde_array[i][j] = expr;
       }

       vcoul[i] = 1.0;
   }

    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;

    //For MPI Work distribution
    int tmp;
    for(int ig=0, tmp=1; ig < ngpown; ++ig,tmp++)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0, tmp=1; ig<ncouls; ++ig,tmp++)
        indinv[ig] = ig;


    long double start_time = omp_get_wtime(); //Start timing here.

    for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
    {
        flag_occ = n1 < nvband;

//#pragma omp parallel for private(mygpvar1, mygpvar2, schs, matngpmatmg, matngmatmgp,schstemp) shared(achstemp) schedule(static) 
        for(int my_igp = 0; my_igp< ngpown; my_igp++)
        {
            schstemp = 0+0i;;
            int igmax;
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            if(indigp == ncouls)
                igp = ncouls;

            if(!(igp > ncouls || igp < 0)){

            if(gppsum == 1)
                igmax = igp;
            else
                igmax = ncouls;

            mygpvar1 = std::conj(aqsmtemp[igp][n1]);
            mygpvar2 = aqsntemp[igp][n1];

            if(gppsum == 1)
            {

                //Aggregating results in schstemp
                for(int ig=0; ig<igmax; ++ig)
                {
                    schs = -I_eps_array[ig][my_igp];
                    matngmatmgp = aqsntemp[ig][n1] * mygpvar1;
                    matngpmatmg = std::conj(aqsmtemp[ig][n1]) * mygpvar2;
                    schstemp = schstemp + matngmatmgp*schs + matngpmatmg*(std::conj(schs));

                }
                //ig = igp ;
                schs = -I_eps_array[igp][my_igp];
                matngmatmgp = aqsntemp[igp][n1] * mygpvar1;

                if(abs(schs) > to1)
                    schstemp = schstemp + matngmatmgp * schs;
            }
            else
            {
                for(int ig=1; ig<igmax; ++ig)
                    schstemp = schstemp - aqsntemp[ig][n1] * I_eps_array[ig][my_igp] * mygpvar1;
            }

            }
//#pragma omp critical
            achstemp += schstemp * vcoul[igp] *(long double) 0.5;
        }

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(abs(wx_array[iw]) < to1) wx_array[iw] = to1;
        }

//#pragma omp parallel for shared(wtilde_array, aqsntemp, aqsmtemp, I_eps_array, scha,wx_array)  firstprivate(ssx_array, sch_array, halfinvwtilde, ssxcutoff, sch, ssx, \
        Omega2, scht, ssxt, wxt, eden, cden) schedule(dynamic) \
        private(scha_mult, mygpvar1, mygpvar2, wtilde, matngmatmgp, matngpmatmg, wtilde2, wdiff, delw, delw2, delwr, wdiffr)
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            if(indigp == ncouls)
                igp = ncouls-1;
            int igmax;

            if(!(igp > ncouls || igp < 0)) {
            if(gppsum == 1)
                igmax = igp;
            else
                igmax = ncouls;

            for(int i=0; i<3; i++)
            {
                ssx_array[i] = expr0;
                sch_array[i] = expr0;
            }

            mygpvar1 = std::conj(aqsmtemp[igp][n1]);
            mygpvar2 = aqsmtemp[igp][n1];

            if(flag_occ)
            {
                for(int iw=nstart; iw<nend; iw++)
                {
                        scht = ssxt = expr0;
                        wxt = wx_array[iw];

                        if(gppsum == 1)
                        {
                            for(int ig=0; ig<igmax; ++ig)
                            {
                                wtilde = wtilde_array[ig][my_igp];
                                wtilde2 = std::pow(wtilde,2);
                                Omega2 = wtilde2*I_eps_array[ig][my_igp];

                                if(std::abs(Omega2) < to1) break; //ask jack about omega2 comparison with to1

                                matngmatmgp = aqsntemp[ig][n1] * mygpvar1;
                                if(ig != igp) matngpmatmg = std::conj(aqsmtemp[ig][n1]) * mygpvar2;

                                halfinvwtilde = ((long double)0.5)/wtilde;
                                delw = (wxt - wtilde) * halfinvwtilde;
                                delw2 = std::pow(abs(delw),2);


                                if((abs(wxt - wtilde) < gamma) || (delw2 < to1))
                                {
                                    sch = expr0;
                                    if(abs(wtilde) > to1)
                                        ssx = -Omega2 / ((long double)4.0 * wtilde2 * ((long double)1.0 + delw));
                                    else
                                        ssx = expr0;
                                }
                                else
                                {
                                    sch = wtilde * I_eps_array[ig][my_igp] / (wxt - wtilde);
                                    ssx = Omega2 / (pow(wxt,2) - wtilde2);
                                }


                                ssxcutoff = sexcut*abs(I_eps_array[ig][my_igp]);
                                if((abs(ssx) > ssxcutoff) &&(abs(wxt) < 0.0)) ssx = 0.0 + 0.0i;

                                if(ig != igp-1)
                                {
                                    ssxa[ig] = matngmatmgp*ssx + matngpmatmg*conj(ssx);
                                    scha[ig] = matngmatmgp*sch + matngpmatmg*conj(sch);
                                }
                                else
                                {
                                    ssxa[ig] = matngmatmgp*ssx;
                                    scha[ig] = matngmatmgp*sch;
                                }

                                ssxt += ssxa[ig];
                                scht += scha[ig];
                            }
                        }
                        else
                        {
                            //344-394
                            for(int ig=0; ig<igmax; ++ig)
                            {
                                wtilde = wtilde_array[ig][my_igp];
                                wtilde2 = pow(wtilde, 2);
                                Omega2 = wtilde * I_eps_array[ig][my_igp];

                                matngmatmgp = aqsntemp[ig][n1] * mygpvar1;
                                wdiff = wxt - wtilde;

                                cden = wdiff;
                                rden = real(cden * conj(cden));
                                rden = 1.00 / rden;
                                delw = wtilde * conj(cden) * rden;
                                delwr = real(delw * conj(delw));
                                wdiffr = real(wdiff * conj(wdiff));

                                if((wdiffr > limittwo) && (delwr < limitone))
                                {
                                    sch = delw * I_eps_array[ig][my_igp];
                                    cden = pow(wxt,2);
                                    rden = real(cden * conj(cden));
                                    rden = 1.00 / rden;
                                    ssx = Omega2 * conj(cden) * rden;
                                }
                                else if (delwr > to1)
                                {
                                    sch = expr0;
                                    cden = (long double) 4.00 * wtilde2 * (delw + (long double)0.50);
                                    rden = real(cden * conj(cden));
                                    rden = 1.00/rden;
                                    ssx = -Omega2 * conj(cden) * rden * delw;
                                }
                                else
                                {
                                    sch = expr0;
                                    ssx = expr0;
                                }

                                ssxcutoff = sexcut*abs(I_eps_array[ig][my_igp]);
                                if((abs(ssx) > ssxcutoff) && (abs(wxt) < 0.00)) ssx = 0.00;

                                ssxa[ig] = matngmatmgp*ssx;
                                scha[ig] = matngmatmgp*sch;

                                ssxt += ssxa[ig];
                                scht += scha[ig];
                            }
                        }
                        ssx_array[iw] += ssxt;
                        sch_array[iw] +=(long double) 0.5*scht;
                    }
                }
                else
                {
                    int igblk = 512;
                    //403 - 479
                    for(int igbeg=0; igbeg<igmax; igbeg+=igblk)
                    {
                        int igend = min(igbeg+igblk-1, igmax);
                        for(int iw=nstart; iw<nend; ++iw)
                        {
                            scht = ssxt = expr0;
                            wxt = wx_array[iw];

                            int tmpVal = 0;

                            if(gppsum == 1)
                            {
                                for(int ig= igbeg; ig<min(igend,igmax-1); ++ig)
                                {
                                    wtilde = wtilde_array[ig][my_igp];
                                    matngmatmgp = aqsntemp[ig][n1] * mygpvar1;
                                    wdiff = wxt - wtilde;
                                    delw = wtilde / wdiff;
                                    delw2 = real(delw * conj(delw));
                                    wdiffr = real(wdiff * conj(wdiff));
                                    if((abs(wdiffr) < limittwo) || (delw2 > limitone))
                                        scha_mult = 1.0;
                                    else 
                                        scha_mult = 0.0;

                                    sch = delw * I_eps_array[ig][my_igp] * scha_mult;
                                    scha[ig] = matngmatmgp*sch + conj(aqsmtemp[ig][n1]) * mygpvar2 * conj(sch);
                                    scht += scha[ig];
                                }
                                if(igend == (igmax-1))
                                {
                                    int ig = igmax;
                                    wtilde = wtilde_array[ig][my_igp];
                                    matngmatmgp = aqsntemp[ig][n1] * mygpvar1;
                                    wdiff = wxt - wtilde;
                                    delw = wtilde / wdiff;
                                    delw2 = real(delw * conj(delw));
                                    wdiffr = real(wdiff * conj(wdiff));
                                    if((abs(wdiffr) < limittwo) || (delw2 > limitone))
                                        scha_mult = 1.0;
                                    else scha_mult = 0.0;

                                    sch = delw * I_eps_array[ig][my_igp] * scha_mult;
                                    scha[ig] = matngmatmgp * sch;
                                    scht = scht + scha[ig];
                                }
                            }
                            else
                            {
                                for(int ig = igbeg; ig<min(igend,igmax); ++ig)
                                {
                                    wdiff = wxt - wtilde_array[ig][my_igp];
                                    cden = wdiff;
                                    rden = real(cden * conj(cden));
                                    rden = 1.00/rden;
                                    delw = wtilde_array[ig][my_igp] * conj(cden) * rden;
                                    delwr = real(delw * conj(delw));
                                    wdiffr = real(wdiff * conj(wdiff));

                                    scha[ig] = mygpvar1 * aqsntemp[ig][n1] * delw * I_eps_array[ig][my_igp];

                                    if((wdiffr > limittwo) && (delwr < limitone)) 
                                        scht += scha[ig];
                                }
                            }
                            sch_array[iw] +=(long double) 0.5*scht;
                        }
                    }
                }

            if(flag_occ)
            {
                for(int iw=nstart; iw<nend; ++iw)
                {
//#pragma omp critical
                    asxtemp[iw] += ssx_array[iw] * occ * vcoul[igp]; //occ does not change and is 1.00 so why not remove it.
                }
            }

//#pragma omp critical
    {
            for(int iw=nstart; iw<nend; ++iw)
                achtemp[iw] += sch_array[iw] * vcoul[igp];

            acht_n1_loc[n1] += sch_array[2] * vcoul[igp];
    }


            } //for the if-loop to avoid break inside an openmp pragma statment
        }
    }

    double end_time = omp_get_wtime(); //End timing here

    for(int iw=nstart; iw<nend; ++iw)
        cout << "achtemp[" << iw << "] = " << std::setprecision(15) << achtemp[iw] << endl;

    cout << "********** Time Taken **********= " << end_time - start_time << " secs" << endl;

    return 0;
}

//Almost done code
