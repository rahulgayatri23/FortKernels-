#include <iostream>
#include <cstdlib>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <math.h>

#include </ccs/home/rgayatri/Kokkos/kokkos/core/src/Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
using namespace std;


KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> kokkos_pow(Kokkos::complex<double> compl_num, int n)
{
//    std::cout << "*********************************************" << std::endl;
//    double tmp_real = (std::pow((std::pow(Kokkos::real(compl_num),2) + std::pow(Kokkos::imag(compl_num),2)),(n/2) )) * cos(n*atan(Kokkos::imag(compl_num)/Kokkos::real(compl_num)));
//    double tmp_imag = (std::pow((std::pow(Kokkos::real(compl_num),2) + std::pow(Kokkos::imag(compl_num),2)),(n/2) )) * sin(n*atan(Kokkos::imag(compl_num)/Kokkos::real(compl_num)));

//    return Kokkos::complex<double> (tmp_real, tmp_imag);

    double compl_real = Kokkos::real(compl_num);
    double compl_imag = Kokkos::imag(compl_num);

    if(n == 2)
    {
        compl_real = compl_real*compl_real - compl_imag*compl_imag;
        compl_imag = 2*(Kokkos::real(compl_num))*compl_imag;

    }

    Kokkos::complex<double> powerExpr(compl_real,compl_imag);
//    std::cout << "Original Expression = " << compl_num << "\t Powered expression = " << powerExpr << std::endl;

//    std::complex<double> expr(Kokkos::real(compl_num),Kokkos::imag(compl_num));
//    std::cout << "orig expr = " << expr << " to the power of " << n << std::endl;
//    std::complex<double> expr2 = std::pow(expr,n);
//    std::cout << "power expr = " << expr2 << std::endl;
//
//    std::complex<double> tmp( 0.5,0.5);
//    std::cout << "tmp = " << tmp << "\t pow(tmp) = " << std::pow(tmp,2) << std::endl;
//
//    return (Kokkos::complex<double> (std::real(expr), std::imag(expr)));
//
    return powerExpr;
}

struct complex2Reduce
{

    Kokkos::complex<double> first;
    Kokkos::complex<double> second;

    complex2Reduce operator +=(complex2Reduce obj2);

};

KOKKOS_INLINE_FUNCTION
complex2Reduce complex2Reduce::operator +=(complex2Reduce obj2)
{
    this->first+=obj2.first;
    this->second+=obj2.second;

    return *this;
}

//KOKKOS_INLINE_FUNCTION
//complex2Reduce complex2Reduce::operator +=(complex2Reduce obj, Kokkos::complex<double> up1, Kokkos::complex<double> up2 )
//{
//    obj.first += up1;
//    obj.second += up2;
//
//    return obj;
//}

int main(int argc, char** argv)
{
  Kokkos::initialize( argc, argv );
  {
//    double start_time = omp_get_wtime(); //start timing here

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

    int igmax = ncouls;

//    int tid, NTHREADS; // OpenMP related threading variables.


    int npes = 1; //Represents the number of ranks per node
    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

    double e_lk = 10;
    double dw = 1;
    int nstart = 0, nend = 3;

    Kokkos::View<int*> inv_igp_index ("inv_igp_index", ngpown);
    Kokkos::View<int*> indinv ("indinv", ncouls);


    double to1 = 1e-6;
//    std::cout << setprecision(16) << "to1 = " << to1 << endl;

    double gamma = 0.5;
    double sexcut = 4.0;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);

    double e_n1kq= 6.0; //This in the fortran code is derived through the double dimenrsion array ekq whose 2nd dimension is 1 and all the elements in the array have the same value


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

//Starting to assign viewspaces to change inside the lambda
//      Kokkos::View<double> wx_array("wx_array",3);
      double wx_array[3];

    Kokkos::complex<double> expr(0.5, 0.5);
    Kokkos::complex<double> expr0(0.0, 0.0);
    Kokkos::View<Kokkos::complex<double>* > asxtemp("asxtemp", nend-nstart);
    Kokkos::View<Kokkos::complex<double>* > acht_n1_loc("asxtemp", number_bands);
    Kokkos::View<Kokkos::complex<double>* > ssx_array("ssx_array", 3), sch_array("sch_array", 3);


    Kokkos::View<Kokkos::complex<double>** > wtilde_array ("wtilde_array", ncouls, ngpown);
    Kokkos::View<Kokkos::complex<double>* > achtemp ("achtemp", nend-nstart);
    Kokkos::View<Kokkos::complex<double>* > vcoul ("vcoul", ncouls);
    Kokkos::View<Kokkos::complex<double>** > aqsntemp ("aqsntemp", ncouls, number_bands);
    Kokkos::View<Kokkos::complex<double>** > aqsmtemp ("aqsmtemp", ncouls, number_bands);
    Kokkos::View<Kokkos::complex<double>** > I_eps_array ("I_eps_array", ncouls, ngpown);


    Kokkos::View<Kokkos::complex<double> > mygpvar1("mygpvar1"), mygpvar2("mygpvar2"), \
        schs("schs"), matngpmatmg("matngpmatmg"), matngmatmgp("matngmatmgp"), \
        achstemp("achstemp"), schstemp("schstemp"), \
        ssxa("ssxa", ncouls), scha("scha", ncouls);

    double occ=1.0;
    bool flag_occ;


   for(int i=0; i<ncouls; i++)
   {
       for(int j=0; j<number_bands; j++)
       {
           aqsmtemp(i,j) = expr;
           aqsntemp(i,j) = expr;
       }

       for(int j=0; j<ngpown; j++)
       {
           I_eps_array(i,j) = expr;
           wtilde_array(i,j) = expr;
//           I_eps_array[i][j] = 0.5 + 0.5i;
//           wtilde_array[i][j] = 0.50 + 0.5i;
       }

       vcoul[i] = 1.0;
   }

    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;

    //For MPI Work distribution
    for(int ig=0; ig < ngpown; ++ig)
        inv_igp_index[ig] = ig * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0; ig<ncouls; ++ig)
        indinv[ig] =ig;


    for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
    {
        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(abs(wx_array[iw]) < to1) wx_array[iw] = to1;

//            std::cout << "1 : wx_array[" << iw << "] = " << wx_array[iw] << std::endl;
        }

        flag_occ = n1 < nvband;

        Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, Kokkos::complex<double> &achstempK) {

            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            if(!(igp > ncouls || igp < 0))
            {

//                igmax = ncouls;
                mygpvar1() = Kokkos::conj(aqsmtemp(igp,n1));
                mygpvar2() = aqsntemp(igp,n1);

                if(gppsum == 1)
                {
                    //Aggregating results in schstemp
                    for(int ig=0; ig<igmax-1; ++ig)
                    {
                        schs() = I_eps_array(ig,my_igp);
                        matngmatmgp() = aqsntemp(ig,n1) * mygpvar1();
                        matngpmatmg() = Kokkos::conj(aqsmtemp(ig,n1)) * mygpvar2();
                        schstemp() += matngmatmgp() * schs() + matngpmatmg() * (Kokkos::conj(schs()));
                    }
                    schs() = I_eps_array(igp,my_igp);
                    matngmatmgp() = aqsntemp(igp,n1) * mygpvar1();

                    if(Kokkos::abs(schs()) > to1)
                        schstemp() += matngmatmgp() * schs();
                }
                else
                {
                    for(int ig=1; ig<igmax; ++ig)
                        schstemp() -= aqsntemp(ig,n1) * I_eps_array(ig,my_igp) * mygpvar1();
                }

            }
            achstempK += 0.5 * schstemp() * vcoul(igp) ;

            }, achstemp());

//*********************************************************************************************
//Rahul - check if there is something in between the 2 for-loops

        Kokkos::parallel_for(Kokkos::TeamPolicy<>(ngpown, Kokkos::AUTO), KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>:: member_type teamMember)
 //       for(int my_igp=0; my_igp<ngpown; my_igp++)
        {
                const int my_igp = teamMember.league_rank();


            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            if(!(igp > ncouls || igp < 0))
            {
                for(int i=0; i<3; i++)
                {
                    ssx_array[i] = expr0;
                    sch_array[i] = 0.00;
                }

                mygpvar1() = Kokkos::conj(aqsmtemp(igp,n1));
                mygpvar2() = aqsmtemp(igp,n1);

                Kokkos::complex<double> wtilde, wtilde2, Omega2, ssx, sch, \
                delw, ssxt, scht, wdiff, cden, delwr;
                double delw2;

                double ssxcutoff, rden, wdiffr ;

                if(flag_occ)
                {
                   for(int iw=nstart; iw<nend; ++iw)
                   {
//                           ssxt = expr;
                           double wxt = wx_array[iw];

                           if(gppsum == 1)
                           {
                               complex2Reduce reduce_2complex;
                               reduce_2complex.first = ssxt;
                               reduce_2complex.first = scht;

                             Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, igmax), [&] (int ig, Kokkos::complex<double> &ssxtUpdate)
 //                              Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, igmax), [&] (int ig, complex2Reduce &reduce_2complexUpdate)
                                       // Rahul - you have to update two variables in this reduction step. For that you have to create the structure of these and then overload the operator +=
                               {
                                   wtilde = wtilde_array(ig,igp);
                                   wtilde2 = kokkos_pow(wtilde, 2);
                                   Omega2 = wtilde2*I_eps_array(ig,my_igp);

//                                    if(Kokkos::abs(Omega2) < to1) break; //ask jack about omega2 comparison with to1

                                   matngmatmgp() = aqsntemp(ig,n1) * mygpvar1();
                                   if(ig != igp) matngpmatmg() = Kokkos::conj(aqsmtemp(ig,n1)) * mygpvar2();

                                   Kokkos::complex<double> wxt_wtilde(wxt - Kokkos::real(wtilde), (0 - Kokkos::imag(wtilde)));
                                   delw = kokkos_pow(wxt_wtilde,2);
                                   delw2 = Kokkos::real(kokkos_pow(Kokkos::abs(delw),2));

                                   if((Kokkos::abs(wxt_wtilde) < gamma)|| ( delw2 < to1))
                                   {
                                       if(Kokkos::abs(wtilde) > to1)
                                           ssx = -Omega2 / ((1.0 + Kokkos::real(delw)) * 4.0 * wtilde2 );
                                       else
                                           ssx = 0.00;
                                   }
                                   else
                                   {
                                       sch = wtilde * I_eps_array(ig,my_igp) / wxt_wtilde;
                                       ssx = Omega2 / ((kokkos_pow(wxt,2)) - wtilde2);
                                   }

                                   ssxcutoff = sexcut*abs(I_eps_array(ig,my_igp));
                                   if((abs(ssx) > ssxcutoff) &&(abs(wxt) < 0.0))
                                       ssx=expr;

                                   if(ig != igp)
                                   {
                                       ssxa(ig) = matngmatmgp()*ssx + matngpmatmg()*conj(ssx);
                                       scha(ig) = matngmatmgp()*sch + matngpmatmg()*conj(sch);
                                   }
                                   else
                                   {
                                       ssxa(ig) = matngmatmgp()*ssx;
                                       scha(ig) = matngmatmgp()*sch;
                                   }

                                   complex2Reduce tmp2Reduce;
                                   tmp2Reduce.first = ssxa(ig);
                                   tmp2Reduce.second = scha(ig);

                                   reduce_2complex +=(tmp2Reduce);

                                   ssxtUpdate += ssxa(ig);
                                   scht += scha(ig);
                               }, ssxt);

                           }
                           else
                           {
                               //344-394
                               Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, igmax), [&] (int ig, Kokkos::complex<double> &ssxtUpdate)
                                       // Rahul - you have to update two variables in this reduction step. For that you have to create the structure of these and then overload the operator +=
//                                for(int ig=0; ig<igmax; ++ig)
                               {
                                   wtilde = wtilde_array(ig,my_igp);
                                   wtilde2 = kokkos_pow(wtilde, 2);
                                   Omega2 = wtilde * I_eps_array(ig,my_igp);

                                   matngmatmgp() = aqsntemp(ig,n1) * mygpvar1();
//                                   wdiff = wxt - wtilde;
                                   Kokkos::complex<double> wdiff(wxt - Kokkos::real(wtilde), (0 - Kokkos::imag(wtilde)));
//
                                   cden = wdiff;
                                   rden = Kokkos::real(cden * Kokkos::conj(cden));
                                   rden = 1.00 / rden;
                                   delw = rden * wtilde * Kokkos::conj(cden);
                                   delwr = Kokkos::real(delw * Kokkos::conj(delw));
                                   wdiffr = Kokkos::real(wdiff * Kokkos::conj(wdiff));
//
////                                    if((wdiffr > limittwo) && (delwr < limitone)) //Rahul - check this
                                   if((wdiffr > limittwo) )
                                   {
                                       sch = delw * I_eps_array(ig,my_igp);
                                       cden = kokkos_pow(wxt,2);
                                       rden = Kokkos::real(cden * Kokkos::conj(cden));
                                       rden = 1.00 / rden;
                                       ssx = rden * Omega2 * Kokkos::conj(cden);
                                   }
                                   else if (Kokkos::real(delwr) > to1)
                                   {
                                       sch = 0.00;
                                       cden = 4.00 * (Kokkos::real(delw) + 0.50);
                                       cden *= wtilde2;
                                       rden = real(cden * Kokkos::conj(cden));
                                       rden = 1.00/rden;
                                       ssx = rden * -Omega2 * Kokkos::conj(cden) * delw;
                                   }
                                   else
                                   {
                                       sch = 0.00;
                                       ssx = 0.00;
                                   }

                                   ssxcutoff = sexcut*Kokkos::abs(I_eps_array(ig,my_igp));
                                   if((Kokkos::abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = 0.00;

                                   ssxa(ig) = matngmatmgp()*ssx;
                                   scha(ig) = matngmatmgp()*sch;

                                   ssxtUpdate += ssxa(ig);
                                   scht += scha(ig);
                               }, ssxt);
                           }
//                           std::cout << "1 : ssx_array(" << iw << ") = " << ssx_array(iw) << "\t sch_array( " << iw << ") = " << sch_array(iw) << std::endl;
                           ssx_array(iw) += ssxt;
                           sch_array(iw) += 0.5*scht;
//                           std::cout << "2 : ssx_array(" << iw << ") = " << ssx_array(iw) << "\t sch_array( " << iw << ") = " << sch_array(iw) << std::endl;
                       }

                    }
                    else
                    {
                        int igblk = 512;
                        //403 - 479
//                        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, igmax), [&] (int igbeg)
                        for(int igbeg=0; igbeg<igmax; igbeg+=igblk)
                        {
                            double scha_mult;
                            int igend = min(igbeg+igblk-1, igmax);


                            for(int iw=nstart; iw<nend; ++iw)
                            {

                                scht = ssxt = 0.00;
                                double wxt = wx_array[iw];

                                if(gppsum == 1)
                                {
                                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, igmax), [&] (int ig, Kokkos::complex<double> &schtUpdate)
////                                    for(int ig= igbeg; ig<min(igend,igmax-1); ++ig)
                                    {
                                        wtilde = wtilde_array(ig,my_igp);
                                        matngmatmgp() = aqsntemp(ig,n1) * mygpvar1();
//                                        wdiff = wxt - wtilde;
                                         Kokkos::complex<double> wdiff(wxt - Kokkos::real(wtilde), (0 - Kokkos::imag(wtilde)));
                                        delw = wtilde / wdiff;
                                        delw2 = Kokkos::real(delw * Kokkos::conj(delw));
                                        wdiffr = Kokkos::real(wdiff * Kokkos::conj(wdiff));
                                        if((wdiffr < limittwo)) //Rahul - check this, so many things need to be set up...
                                            scha_mult = 1.0;
                                        else
                                            scha_mult = 0.0;

                                        sch = delw * I_eps_array(ig,my_igp) ;
                                        scha(ig) = matngmatmgp()*sch + Kokkos::conj(aqsmtemp(ig,n1)) * mygpvar2() * Kokkos::conj(sch);
                                        schtUpdate += scha(ig);
////                                        scht += scha(ig);
                                    }, scht);
                                    if(igend == (igmax-1))
                                    {
                                        int ig = igmax;
                                        wtilde = wtilde_array(ig,my_igp);
                                        matngmatmgp() = aqsntemp(ig,n1) * mygpvar1();
                                        Kokkos::complex<double> wdiff(wxt - Kokkos::real(wtilde), (0 - Kokkos::imag(wtilde)));
//                                        wdiff = wxt - wtilde;
                                        delw = wtilde / wdiff;
                                        delw2 = Kokkos::real(delw * Kokkos::conj(delw));
                                        wdiffr = Kokkos::real(wdiff * Kokkos::conj(wdiff));
//                                        if((Kokkos::abs(wdiffr) < limittwo) || (delw2 > limitone))
                                        if(wdiffr < limittwo) //Rahul - check this
                                            scha_mult = 1.0;
                                        else scha_mult = 0.0;

                                        sch = scha_mult * delw * I_eps_array(ig,my_igp) ;
                                        scha(ig) = matngmatmgp() * sch;
                                        scht += scha(ig);
                                    }
                                }
                                else
                                {
                                    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, igmax), [&] (int ig)
//                                    for(int ig = igbeg; ig<min(igend,igmax); ++ig)
                                    {
                                        wtilde = wtilde_array(ig,my_igp);
                                        Kokkos::complex<double> wdiff(wxt - Kokkos::real(wtilde), (0 - Kokkos::imag(wtilde)));
                                        cden = wdiff;
                                        rden = Kokkos::real(cden * Kokkos::conj(cden));
                                        rden = 1.00/rden;
                                        delw = rden * wtilde_array(ig,my_igp) * Kokkos::conj(cden);
                                        delwr = Kokkos::real(delw * Kokkos::conj(delw));
                                        wdiffr = Kokkos::real(wdiff * Kokkos::conj(wdiff));

                                        scha(ig) = mygpvar1() * aqsntemp(ig,n1) * delw * I_eps_array(ig,my_igp);

                                        if((wdiffr > limittwo)) scht = scht + scha(ig); //Rahul - same as ealier delw2 > limitone does not work
//
                                        });
                                }
                                sch_array(iw) += 0.5*scht;
////                                schArrayUpdate += 0.5*scht;
                            }
                        }//);
                    }
//
                if(flag_occ)
                {
                    for(int iw=nstart; iw<nend; ++iw)
                    {
//#pragma omp critical
                        asxtemp(iw) += occ * ssx_array(iw) * vcoul(igp); //occ does not change and is 1.00 so why not remove it.
                    }
                }

//#pragma omp critical
            {
                for(int iw=nstart; iw<nend; ++iw)
                    achtemp(iw) = achtemp(iw) + sch_array(iw) * vcoul(igp);

                acht_n1_loc(n1) += sch_array(2) * vcoul(igp);
            }

            } //for the if-loop to avoid break inside an openmp pragma statment
        }); //2nd for loop
    } //n1 band
//
//  //  double end_time = omp_get_wtime(); //End timing here
//
    for(int iw=nstart; iw<nend; ++iw)
        cout << "achtemp[" << iw << "] = " << achtemp[iw] << endl;
//
////    cout << "********** Time Taken **********= " << end_time - start_time << " secs" << endl;

  }
    Kokkos::finalize();

    return 0;
}

