#include <iostream>
#include <cstdlib>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
using namespace std;

KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> kokkos_square(Kokkos::complex<double> compl_num, int n)
{
    std::complex<double> powerExpr(Kokkos::real(compl_num), Kokkos::imag(compl_num));
    powerExpr = std::pow(powerExpr,n);

    return powerExpr;
}

KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> doubleMinusKokkosComplex(double op1, Kokkos::complex<double> op2)
{
    Kokkos::complex<double> expr((op1 - Kokkos::real(op2)), (0 - Kokkos::imag(op2)));
    return expr;
}

KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> doublePlusKokkosComplex(double op1, Kokkos::complex<double> op2)
{
    Kokkos::complex<double> expr((op1 + Kokkos::real(op2)), (0 + Kokkos::imag(op2)));
    return expr;
}

KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> doubleMultKokkosComplex(double op1, Kokkos::complex<double> op2)
{
    Kokkos::complex<double> expr((op1 * Kokkos::real(op2)), (0 * Kokkos::imag(op2)));
    return expr;
}

KOKKOS_INLINE_FUNCTION
void reduce_achstemp(int n1, Kokkos::View<int*> inv_igp_index, int ncouls, Kokkos::View<Kokkos::complex<double>** > aqsmtemp, Kokkos::View<Kokkos::complex<double>** > aqsntemp, Kokkos::View<Kokkos::complex<double>** > I_eps_array, Kokkos::complex<double>& achstemp, Kokkos::View<int*> indinv, int ngpown, Kokkos::View<double*> vcoul)
{
    Kokkos::complex<double> expr(0.5 , 0.5);
    Kokkos::complex<double> expr0(0.0 , 0.0);
    Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, Kokkos::complex<double> &achstempUpdate)
    {
        Kokkos::complex<double> mygpvar1, mygpvar2;
        Kokkos::complex<double> schstemp = expr0;
        int igmax = ncouls;

        int indigp = inv_igp_index(my_igp);
        int igp = indinv(indigp);
        if(indigp == ncouls)
            igp = ncouls-1;

        if(!(igp > ncouls || igp < 0)){

          igmax = ncouls;

        mygpvar1 = Kokkos::conj(aqsmtemp(n1,igp));
        mygpvar2 = aqsntemp(n1,igp);

            for(int ig=1; ig<igmax; ++ig)
            {
                schstemp = schstemp - aqsmtemp(n1,ig);
                schstemp *= I_eps_array(my_igp, ig);
                schstemp *= mygpvar1;
            }

        }
        achstempUpdate += 0.5 * vcoul(igp) * schstemp;
    },achstemp);
}

KOKKOS_INLINE_FUNCTION
void flagOCC_solver(double wxt, Kokkos::View<Kokkos::complex<double>** > wtilde_array, int my_igp, int n1, Kokkos::View<Kokkos::complex<double>** > aqsmtemp, Kokkos::View<Kokkos::complex<double>** > aqsntemp, Kokkos::View<Kokkos::complex<double>** > I_eps_array, Kokkos::complex<double> &ssxt, Kokkos::complex<double> &scht, int ncouls, int igp)
{
    Kokkos::complex<double> ssxa[ncouls], scha[ncouls];
    int igmax = ncouls;

    for(int ig=0; ig<igmax; ++ig)
    {
        Kokkos::complex<double> wtilde = wtilde_array(my_igp,ig);
        Kokkos::complex<double> wtilde2 = kokkos_square(wtilde,2);
        Kokkos::complex<double> Omega2 = wtilde2*I_eps_array(my_igp,ig);
        Kokkos::complex<double> mygpvar1 = Kokkos::conj(aqsmtemp(n1,igp));
        Kokkos::complex<double> matngmatmgp = aqsntemp(n1,ig) * mygpvar1;
        Kokkos::complex<double> expr0( 0.0 , 0.0);
        double to1 = 1e-6;
        double sexcut = 4.0;
        double gamma = 0.5;
        double limitone = 1.0/(to1*4.0);
        double limittwo = pow(0.5,2);
        Kokkos::complex<double> sch, ssx;

        Kokkos::complex<double> wdiff = doubleMinusKokkosComplex(wxt , wtilde);

        Kokkos::complex<double> cden = wdiff;
        double rden = Kokkos::real(cden * Kokkos::conj(cden));
        rden = 1.00 / rden;
        Kokkos::complex<double> delw = rden * wtilde * Kokkos::conj(cden);
        double delwr = Kokkos::real(delw * Kokkos::conj(delw)); //This is diff from original ...
        double wdiffr = Kokkos::real(wdiff * Kokkos::conj(wdiff));

        if((wdiffr > limittwo) && (delwr < limitone))
        {
            sch = delw;
            sch *= I_eps_array(my_igp, ig);
            cden = kokkos_square(wxt,2);
            rden = Kokkos::real(cden * Kokkos::conj(cden));
            rden = 1.00 / rden;
            ssx = rden * Omega2 * Kokkos::conj(cden);
        }
        else if (delwr > to1)
        {
            sch = 0.00;
            cden = 4.00 * wtilde2 * doublePlusKokkosComplex(0.5, delw );
            rden = Kokkos::real(cden * Kokkos::conj(cden));
            rden = 1.00/rden;
            ssx = rden * -Omega2 * conj(cden) * delw;
        }
        else
        {
            sch = expr0;
            ssx = expr0;
        }

        double ssxcutoff = sexcut*abs(I_eps_array(my_igp, ig));
        if((Kokkos::abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;

        ssxa[ig] = matngmatmgp*ssx;
        scha[ig] = matngmatmgp*sch;

        ssxt += ssxa[ig];
        scht += scha[ig];
    }
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {

    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> " << endl;
        exit (0);
    }
    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int nodes_per_group = atoi(argv[4]);

    int igmax = ncouls;

    int npes = 1; //Represents the number of ranks per node
    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

    double e_lk = 10;
    double dw = 1;
    int nstart = 0, nend = 3;

//    int inv_igp_index[ngpown];
//    int indinv[ncouls];


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
        << "\t ngpown = " << ngpown \
        << "\t nend = " << nend \
        << "\t nstart = " << nstart \
        << "\t gamma = " << gamma \
        << "\t sexcut = " << sexcut \
        << "\t limitone = " << limitone \
        << "\t limittwo = " << limittwo << endl;


    double ssxcutoff;
    double wxt, delw2, delwr, wdiffr, scha_mult;
    double occ=1.0;
    bool flag_occ;

    /********************KOKKOS RELATED VARS AND VIEWS ***************************/
    Kokkos::complex<double> expr(0.5 , 0.5);
    Kokkos::complex<double> expr0(0.0 , 0.0);
    Kokkos::complex<double> achstemp = expr0;
    Kokkos::complex<double> achstemp_tmp = expr0;

    Kokkos::View<int*> inv_igp_index("inv_igp_index", ngpown);
    Kokkos::View<int*> indinv("indinv", ncouls);
    Kokkos::View<double*> vcoul ("vcoul", ncouls);

    Kokkos::View<Kokkos::complex<double>** > aqsmtemp ("aqsntemp", number_bands, ncouls);
    Kokkos::View<Kokkos::complex<double>** > aqsntemp ("aqsntemp", number_bands, ncouls);
    Kokkos::View<Kokkos::complex<double>** > I_eps_array ("I_eps_array", ngpown, ncouls);
    Kokkos::View<Kokkos::complex<double>** > wtilde_array ("wtilde_array", ngpown, ncouls);


    Kokkos::View<complex<double> *> asxtemp ("asxtemp", nend-nstart);
    Kokkos::View<complex<double> *> acht_n1_loc(" acht_n1_loc", number_bands);
//    Kokkos::View<complex<double> *> ssxa(" ssxa", ncouls);

    for(int i = 0; i< number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsmtemp(i,j) = expr;
           aqsntemp(i,j) = expr;
       }

    for(int i = 0; i< ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array(i,j) = expr;
           wtilde_array(i,j) = expr;
       }
    
   for(int i=0; i<ncouls; i++)
       vcoul(i) = 1.0;

    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;

    //For MPI Work distribution
    for(int ig=0; ig < ngpown; ++ig)
        inv_igp_index(ig) = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0; ig<ncouls; ++ig)
        indinv(ig) =ig;

//********************** Structures to update arrays inside Kokkos parallel calls ************************************************
//****** achtemp **********
    Kokkos::complex<double>  achtemp[nend-nstart];
    struct achtempStruct 
    {
         Kokkos::complex<double> value[3];
        void operator+=(achtempStruct const& other) 
        {
            for (int i = 0; i < 3; ++i) 
                value[i] += other.value[i];
        }
        void operator+=(achtempStruct const volatile& other) volatile 
        {
            for (int i = 0; i < 3; ++i) 
                value[i] += other.value[i];
        }
    };
    achtempStruct achtempVar = {{achtemp[0],achtemp[1],achtemp[2]}}; 


//**********************************************************************************************************************************

    double start_time = omp_get_wtime(); //Start timing here.

//    Kokkos::parallel_reduce(Kokkos::TeamPolicy<>(number_bands, Kokkos::AUTO), KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>:: member_type teamMember, achtempStruct& achtempVarUpdate)
    for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
    {
 //       const int n1 = teamMember.league_rank();
        double wx_array[3];

        reduce_achstemp(n1, inv_igp_index, ncouls, aqsmtemp, aqsntemp, I_eps_array, achstemp,  indinv, ngpown, vcoul);

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(wx_array[iw] < to1) wx_array[iw] = to1;
        }

//      Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, achtempStruct& achtempVarUpdate)
      Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, achtempStruct& achtempVarUpdate)
//          for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            Kokkos::complex<double> wtilde2, Omega2;
            bool flag_occ = n1 < nvband;
            double wxt, delw2, delwr, wdiffr, scha_mult, ssxcutoff;
//            Kokkos::complex<double> mygpvar1, mygpvar2;
            Kokkos::complex<double> schstemp = expr0;
            Kokkos::complex<double> matngmatmgp = expr;
            Kokkos::complex<double> matngpmatmg = expr;
            Kokkos::complex<double> schs = expr0;
            Kokkos::complex<double> wtilde;
            Kokkos::complex<double> scht, ssxt;
            int indigp = inv_igp_index(my_igp);
            int igp = indinv(indigp);
            if(indigp == ncouls)
                igp = ncouls-1;
            Kokkos::complex<double> scha[ncouls], ssx_array[3], sch_array[3];

            if(!(igp > ncouls || igp < 0)){
            
            for(int i=0; i<3; i++)
            {
                ssx_array[i] = expr0;
                sch_array[i] = expr0;
            }

            if(flag_occ)
            {
                for(int iw=nstart; iw<nend; ++iw)
                {
                    scht = ssxt = expr0;
                    wxt = wx_array[iw];
                    flagOCC_solver(wxt, wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp);

                    ssx_array[iw] += ssxt;
                    sch_array[iw] += 0.5*scht;
                }
            }
            else
            {
                int igblk = 512;
                int igmax = ncouls;
                Kokkos::complex<double> delw, sch, wdiff, rden;
                Kokkos::complex<double> mygpvar1 = Kokkos::conj(aqsmtemp(n1,igp));

                for(int igbeg=0; igbeg<igmax; igbeg+=igblk)
                {
                    int igend = min(igbeg+igblk-1, igmax);
                    for(int iw=nstart; iw<nend; ++iw)
                    {
                        scht = ssxt = expr0;
                        wxt = wx_array[iw];

                        for(int ig = igbeg; ig<min(igend,igmax); ++ig)
                        {
                            wdiff = doubleMinusKokkosComplex(wxt , wtilde_array(my_igp, ig));
                            rden = (Kokkos::complex<double>) 1.00/(wdiff * Kokkos::conj(wdiff));
                            delw = rden * wtilde_array(my_igp, ig) * Kokkos::conj(wdiff);
                            scha[ig] = mygpvar1 * delw * aqsntemp(n1,ig)* I_eps_array(my_igp, ig);
                        } // for iw-blockSize

                        for(int ig = igbeg; ig<min(igend,igmax); ++ig)
                            scht+=scha[ig];

                        sch_array[iw] += 0.5*scht;
                    } // for nstart - nend
                } //for - ig-Block
            } //else-loop
        } // if-condition

            if(flag_occ)
            {
                for(int iw=nstart; iw<nend; ++iw)
                {
                    Kokkos::View<complex<double>> addVal("addVal");
                    addVal() = occ * ssx_array[iw];
                    asxtemp(iw) += addVal() ; //occ does not change and is 1.00 so why not remove it.
                }
            }

            for(int iw=nstart; iw<nend; ++iw)
            {
                Kokkos::complex<double> addVal = vcoul(igp) * sch_array[iw];
//                achtempVar.value[iw] += addVal;
                achtempVarUpdate.value[iw] += addVal;
            }
            Kokkos::View<complex<double>> addVal("addVal");
            addVal() = sch_array[2];
            acht_n1_loc(n1) += addVal() * vcoul(igp);
//            acht_n1_loc(n1) += sch_array[2] * vcoul(igp);
        },achtempVar); // for - ngpown 

        //Rahul - have to copy it into a diff buffer not related to kokkos-views so that the value is not modified at the start of each iteration.
        for(int iw=nstart; iw<nend; ++iw)
            achtemp[iw] += achtempVar.value[iw];
    } // for - number_bands

   
    double end_time = omp_get_wtime(); //End timing here

    for(int iw=nstart; iw<nend; ++iw)
        cout << "Final achtemp[" << iw << "] = " << achtemp[iw] << endl;
//        cout << "achtemp[" << iw << "] = " << achtemp[iw] << endl;

    cout << "********** Time Taken **********= " << end_time - start_time << " secs" << endl;

    }
    Kokkos::finalize();

    return 0;
}

//Almost done code
