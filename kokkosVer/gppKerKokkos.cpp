#include <iostream>
#include <cstdlib>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
<<<<<<< HEAD
=======
//#include </ccs/home/rgayatri/Kokkos/kokkos/core/src/Kokkos_Complex.hpp>

>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
using namespace std;
int debug = 0;

KOKKOS_INLINE_FUNCTION
<<<<<<< HEAD
Kokkos::complex<double> kokkos_square(Kokkos::complex<double> compl_num, int n)
{
    std::complex<double> powerExpr(Kokkos::real(compl_num), Kokkos::imag(compl_num));
    powerExpr = std::pow(powerExpr,n);

=======
Kokkos::complex<double> kokkos_square(Kokkos::complex<double> compl_num)
{

    double compl_real = Kokkos::real(compl_num);
    double compl_imag = Kokkos::imag(compl_num);

    compl_real = compl_real*compl_real - compl_imag*compl_imag;
    compl_imag = 2*(Kokkos::real(compl_num))*compl_imag;

    Kokkos::complex<double> powerExpr(compl_real,compl_imag);
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
    return powerExpr;
}

KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> doubleMinusKokkosComplex(double op1, Kokkos::complex<double> op2)
{
<<<<<<< HEAD
    Kokkos::complex<double> expr((op1 - Kokkos::real(op2)), (0 - Kokkos::imag(op2)));
=======
    Kokkos::complex<double> expr((op1 - Kokkos::real(op2)), (op1 - Kokkos::imag(op2)));
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
    return expr;
}

KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> doublePlusKokkosComplex(double op1, Kokkos::complex<double> op2)
{
<<<<<<< HEAD
    Kokkos::complex<double> expr((op1 + Kokkos::real(op2)), (0 + Kokkos::imag(op2)));
=======
    Kokkos::complex<double> expr((op1 + Kokkos::real(op2)), (op1 + Kokkos::imag(op2)));
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
    return expr;
}

KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> doubleMultKokkosComplex(double op1, Kokkos::complex<double> op2)
{
<<<<<<< HEAD
    Kokkos::complex<double> expr((op1 * Kokkos::real(op2)), (0 * Kokkos::imag(op2)));
    return expr;
}
int main(int argc, char** argv)
{

    Kokkos::initialize(argc, argv);
    {

=======
    Kokkos::complex<double> expr((op1 * Kokkos::real(op2)), (op1 * Kokkos::imag(op2)));
    return expr;
}


int main(int argc, char** argv)
{

>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
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

<<<<<<< HEAD
    int igmax = ncouls;
=======
//    int igmax = ncouls;
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2

    int tid, NTHREADS; // OpenMP related threading variables.


<<<<<<< HEAD
    //    The below 3 params have to be changed for the final version. Currently using small numbers hence setting them to smaller ones...memory constraints on the local machine.
//    int npes = 8; //Represents the number of ranks per node
//    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task
//    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

=======
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
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
        << "\t gppsum = " << gppsum \
        << "\t ngpown = " << ngpown \
        << "\t nend = " << nend \
        << "\t nstart = " << nstart \
        << "\t gamma = " << gamma \
        << "\t sexcut = " << sexcut \
        << "\t limitone = " << limitone \
        << "\t limittwo = " << limittwo << endl;


    //ALLOCATE statements from fortran gppkernel.

<<<<<<< HEAD
    std::complex<double> expr(0.5 , 0.5);
    std::complex<double> expr0(0.0 , 0.0);
//    std::complex<double> aqsmtemp[ncouls][number_bands];
//    std::complex<double> aqsntemp[ncouls][number_bands];
//    std::complex<double> I_eps_array[ncouls][ngpown];
//    std::complex<double> achtemp[nend-nstart];
//    std::complex<double> asxtemp[nend-nstart];
//    std::complex<double> acht_n1_loc[number_bands];

//    double vcoul[ncouls];
    double wx_array[3];

//    std::complex<double> wtilde_array[ncouls][ngpown];
//    std::complex<double> schstemp = expr0;
//    std::complex<double> schs = expr0;
//    std::complex<double> matngmatmgp = expr0;
//    std::complex<double> matngpmatmg = expr0;
//
//    std::complex<double> achstemp = expr0;
//    std::complex<double> wtilde2, Omega2;
//    std::complex<double> halfinvwtilde, delw, ssx, sch, wdiff, cden , eden, mygpvar1, mygpvar2;
//    std::complex<double> ssx_array[3], \
//        sch_array[3], \
        ssxa[ncouls], \
        scha[ncouls], \
        scht, ssxt, wtilde;

    double ssxcutoff;
    double wxt, delw2, delwr, wdiffr, scha_mult, rden;
    double occ=1.0;
    bool flag_occ;

//    //Kokkos::Complex
//    Kokkos::complex<double> expr(0.5 , 0.5);
//    Kokkos::complex<double> expr0(0.0 , 0.0);
//    Kokkos::complex<double> aqsmtemp[ncouls][number_bands];
//    Kokkos::complex<double> aqsntemp[ncouls][number_bands];
//    Kokkos::complex<double> I_eps_array[ncouls][ngpown];
//    Kokkos::complex<double> achtemp[nend-nstart];
//    Kokkos::complex<double> asxtemp[nend-nstart];
//    Kokkos::complex<double> acht_n1_loc[number_bands];
//    Kokkos::complex<double> wtilde_array[ncouls][ngpown];
//    Kokkos::complex<double> schstemp = expr0;
//    Kokkos::complex<double> schs = expr0;
//    Kokkos::complex<double> matngmatmgp = expr0;
//    Kokkos::complex<double> matngpmatmg = expr0;
//
//    Kokkos::complex<double> achstemp = expr0;
//    Kokkos::complex<double> wtilde2, Omega2;
//    Kokkos::complex<double> halfinvwtilde, delw, ssx, sch, wdiff, cden , eden, mygpvar1, mygpvar2;
//    Kokkos::complex<double> ssx_array[3], \
//        sch_array[3], \
//        ssxa[ncouls], \
//        scha[ncouls], \
//        scht, ssxt, wtilde;

    //Kokkos Views 
=======
//    std::complex<double> expr = 0.5 + 0.5i;




    Kokkos::initialize(argc, argv);
    {
    if(Kokkos::OpenMP::is_initialized())
        std::cout << "OpenMP is initialized" << std::endl;


    /********************KOKKOS RELATED VARS AND VIEWS ***************************/
    Kokkos::complex<double> expr(0.5 , 0.5);
    Kokkos::complex<double> expr0(0.0 , 0.0);
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2

    Kokkos::View<int*> inv_igp_index("inv_igp_index", ngpown);
    Kokkos::View<int*> indinv("indinv", ncouls);
    Kokkos::View<double*> vcoul ("vcoul", ncouls);

<<<<<<< HEAD
    Kokkos::View<Kokkos::complex<double>** > aqsntemp ("aqsntemp", ncouls, number_bands);
=======
    Kokkos::View<Kokkos::complex<double>** > aqsntemp ("aqsmtemp", ncouls, number_bands);
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
    Kokkos::View<Kokkos::complex<double>** > aqsmtemp ("aqsmtemp", ncouls, number_bands);
    Kokkos::View<complex<double>** > I_eps_array ("I_eps_array", ncouls, ngpown);
    Kokkos::View<complex<double>** > wtilde_array ("wtilde_array", ncouls, ngpown);

    Kokkos::View<complex<double> *> achtemp ("achtemp", nend-nstart);
<<<<<<< HEAD
    Kokkos::View<complex<double> *> asxtemp ("asxtemp", nend-nstart);
    Kokkos::View<complex<double> *> acht_n1_loc(" acht_n1_loc", number_bands);
    Kokkos::View<complex<double> *> sch_array(" sch_array", 3);
    Kokkos::View<complex<double> *> ssx_array(" ssx_array", 3);
    Kokkos::View<complex<double> *> ssxa(" ssxa", ncouls);
    Kokkos::View<complex<double> *> scha(" scha", ncouls);
    
    Kokkos::View<complex<double>> scht("scht");
    Kokkos::View<complex<double>> ssxt("ssxt");
=======
    Kokkos::View<complex<double> *> asxtemp ("achtemp", nend-nstart);
    Kokkos::View<complex<double> *> acht_n1_loc(" acht_n1_loc", number_bands);
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2

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
<<<<<<< HEAD
           wtilde_array(i,j) = expr;
=======
           wtilde_array(i,j) = expr0;
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
       }

       vcoul(i) = 1.0;
   }

    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;

    //For MPI Work distribution
    for(int ig=0; ig < ngpown; ++ig)
        inv_igp_index(ig) = ig * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0; ig<ncouls; ++ig)
        indinv(ig) =ig;


<<<<<<< HEAD
    double start_time = omp_get_wtime(); //Start timing here.

 //   for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(number_bands, Kokkos::AUTO), KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>:: member_type teamMember)
=======
//    double start_time = omp_get_wtime(); //Start timing here.


//    Kokkos::parallel_for(number_bands, KOKKOS_LAMBDA( int n1)
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(number_bands, Kokkos::AUTO), KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>:: member_type teamMember)
  //  for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
    {
        const int n1 = teamMember.league_rank();
        Kokkos::complex<double> matngmatmgp = expr;
        Kokkos::complex<double> matngpmatmg = expr;
        Kokkos::complex<double> achstemp = expr0;
        Kokkos::complex<double> schs = expr0;
        Kokkos::complex<double> schstemp = expr0;
<<<<<<< HEAD
        Kokkos::complex<double> wtilde;
=======
        Kokkos::complex<double> ssx_array[3], \
            sch_array[3], \
            ssxa[ncouls], \
            scha[ncouls], \
            scht, ssxt, wtilde;
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2

        Kokkos::complex<double> halfinvwtilde, delw, ssx, sch, wdiff, cden , eden, mygpvar1, mygpvar2;
        Kokkos::complex<double> wtilde2, Omega2;
        bool flag_occ = n1 < nvband;
        double wx_array[3];
        double wxt, delw2, delwr, wdiffr, scha_mult, rden, \
        ssxcutoff;
        int igmax = ncouls;

<<<<<<< HEAD
        flag_occ = n1 < nvband;

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, ngpown), [&] (int my_igp, Kokkos::complex<double> &achstempUpdate)
 //       for(int my_igp = 0; my_igp< ngpown; my_igp++)
=======
        double occ=1.0;

//        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, ngpown), [&] (int my_igp, Kokkos::complex<double> &achstempUpdate)
        for(int my_igp = 0; my_igp< ngpown; my_igp++)
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
        {
            int indigp = inv_igp_index(my_igp);
            int igp = indinv(indigp);

            if(!(igp > ncouls || igp < 0)){
<<<<<<< HEAD

            if(gppsum == 1)
                igmax = igp;
            else
              igmax = ncouls;



            mygpvar1 = Kokkos::conj(aqsmtemp(igp,n1));
=======
            if(gppsum == 1)
                igmax = igp;
            else
                igmax=ncouls;

            Kokkos::complex<double> mygpvar1 = Kokkos::conj(aqsmtemp(igp,n1));
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
            mygpvar2 = aqsntemp(igp,n1);

            if(gppsum == 1)
            {
<<<<<<< HEAD

=======
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
                //Aggregating results in schstemp
                for(int ig=0; ig<igmax-1; ++ig)
                {
                    //std::complex<double> schs = I_eps_array[ig][my_igp];
                    schs = I_eps_array(ig,my_igp);
                    matngmatmgp = aqsntemp(ig,n1) * mygpvar1;
                    matngpmatmg = Kokkos::conj(aqsmtemp(ig,n1)) * mygpvar2;
<<<<<<< HEAD
                    schstemp = schstemp + matngmatmgp*schs + matngpmatmg*(Kokkos::conj(schs));
=======
                    schstemp += matngmatmgp*schs + matngpmatmg*(Kokkos::conj(schs));
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2

                }
                //ig = igp ;
                schs = I_eps_array(igp,my_igp);
                matngmatmgp = aqsntemp(igp,n1) * mygpvar1;

                if(abs(schs) > to1)
<<<<<<< HEAD
                    schstemp = schstemp + matngmatmgp * schs;
=======
                    schstemp += matngmatmgp * schs;
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
            }
            else
            {
                for(int ig=1; ig<igmax; ++ig)
<<<<<<< HEAD
                {
                    schstemp = schstemp - aqsntemp(ig,n1);
                    schstemp *= I_eps_array(ig,my_igp);
                    schstemp *= mygpvar1;
                }
            }

            }
            achstempUpdate += 0.5 * vcoul(igp) * schstemp;
        },achstemp);
=======
                    schstemp -= I_eps_array(ig,my_igp) * mygpvar1  * aqsntemp(ig,n1) ;
            }

            }
//            achstempUpdate +=  0.5 * vcoul(igp) * schstemp ;
            achstemp +=  0.5 * vcoul(igp) * schstemp ;
        }//, achstemp);
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
<<<<<<< HEAD
            if(abs(wx_array[iw]) < to1) wx_array[iw] = to1;
=======
            if(wx_array[iw] < to1) wx_array[iw] = to1;
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
        }

        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
<<<<<<< HEAD
            int indigp = inv_igp_index(my_igp);
            int igp = indinv(indigp);

            if(!(igp > ncouls || igp < 0)) {
            
            if(gppsum == 1)
                igmax = igp;
            else
                igmax = ncouls;
=======

            int indigp = inv_igp_index(my_igp);
            int igp = indinv(indigp);
            if(gppsum == 1)
                igmax = igp;
            else
                igmax=ncouls;

            if(!(igp > ncouls || igp < 0))
            {
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2


            for(int i=0; i<3; i++)
            {
<<<<<<< HEAD
                ssx_array(i) = expr0;
                sch_array(i) = expr0;
=======
                ssx_array[i] = expr;
                sch_array[i] = expr0;
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
            }

            mygpvar1 = Kokkos::conj(aqsmtemp(igp,n1));
            mygpvar2 = aqsmtemp(igp,n1);

<<<<<<< HEAD
=======

>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
            if(flag_occ)
            {
                for(int iw=nstart; iw<nend; ++iw)
                {
<<<<<<< HEAD
                        scht() = ssxt() = expr0;
                        wxt = wx_array[iw];

                        if(gppsum == 1)
                        {
                            for(int ig=0; ig<igmax; ++ig)
                            {
                                wtilde = wtilde_array(ig,igp);
                                wtilde2 = kokkos_square(wtilde,2);
                                Omega2 = wtilde2;
                                Omega2 *= I_eps_array(ig,my_igp);

                                if(Kokkos::abs(Omega2) < to1) break; //ask jack about omega2 comparison with to1

                                matngmatmgp = aqsntemp(ig,n1) * mygpvar1;
                                if(ig != igp) matngpmatmg = Kokkos::conj(aqsmtemp(ig,n1)) * mygpvar2;

                                halfinvwtilde = (Kokkos::complex<double>) 0.5/wtilde;
                                delw = kokkos_square(doubleMinusKokkosComplex(wxt, wtilde),2);
                                delw2 = std::pow(Kokkos::abs(delw),2);

                                if((abs(doubleMinusKokkosComplex(wxt, wtilde)) < gamma) || (delw2 < to1))
                                {
                                    sch = expr0;
                                    if(abs(wtilde) > to1)
                                        ssx = -Omega2 / (4.0 * (doublePlusKokkosComplex(1.0 , delw)) * wtilde2 );
                                    else
                                        ssx = 0.00;
                                }
                                else
                                {
                                    sch = wtilde / doubleMinusKokkosComplex(wxt , wtilde);
                                    sch *= I_eps_array(ig,my_igp);
                                    ssx = Omega2 / doubleMinusKokkosComplex(pow(wxt,2) , wtilde2);
                                }

                                ssxcutoff = sexcut*abs(I_eps_array(ig,my_igp));
                                if((abs(ssx) > ssxcutoff) &&(abs(wxt) < 0.0)) ssx = expr0;

                                if(ig != igp)
                                {
                                    ssxa(ig) = matngmatmgp*ssx + matngpmatmg*Kokkos::conj(ssx);
                                    scha(ig) = matngmatmgp*sch + matngpmatmg*Kokkos::conj(sch);
                                }
                                else
                                {
                                    ssxa(ig) = matngmatmgp*ssx;
                                    scha(ig) = matngmatmgp*sch;
                                }

                                ssxt() += ssxa(ig);
                                scht() += scha(ig);
                            }
=======
                        wxt = wx_array[iw];

                        scht = expr0;
                        ssxt = expr0;
                        if(gppsum == 1)
                        {
                            std::cout << "igmax = " << igmax << std::endl;
                            for(int ig=0; ig<igmax; ++ig)
                            {
                                wtilde = wtilde_array(ig, igp);
//                                std::cout << "wtilde[" << ig << "][" << igp << "] = " << wtilde_array(ig,igp) <<  "\t ncouls = " << ncouls << "\t ngpown = " << ngpown << std::endl;
                                wtilde2 = kokkos_square(wtilde);
                                Omega2 = I_eps_array(ig,my_igp) * wtilde2 ;

                                if(Kokkos::abs(Omega2) > to1) 
                                {
                                    matngmatmgp = aqsntemp(ig,n1) * mygpvar1;
                                    if(ig != igp) matngpmatmg = Kokkos::conj(aqsmtemp(ig,n1)) * mygpvar2;
//                                    std::cout << "matngmatmgp = " << matngmatmgp << "\t mygpvar1 = " << mygpvar1 << "\t mygpvar2 = " << mygpvar2 << std::endl;

                                    halfinvwtilde = (Kokkos::complex<double>) 0.5/wtilde; //Rahul - had to typecast, for somereason, even though both rhs and lhs are Kokkos::complex<double>
                                    Kokkos::complex<double> wxt_wtilde = doubleMinusKokkosComplex(wxt , wtilde);
                                    delw = kokkos_square(wxt_wtilde);
                                    delw2 = std::pow(abs(delw),2);

                                    if((Kokkos::abs(wxt_wtilde) < gamma) || (delw2 < to1))
                                    {
                                        sch = expr0;
                                        if(Kokkos::abs(wtilde) > to1)
                                            ssx = -Omega2 / (4.0 * doublePlusKokkosComplex(1.0 , delw) * wtilde2);
                                        else
                                            ssx = 0.00;
                                    }
                                    else
                                    {
                                        sch = I_eps_array(ig,my_igp) * wtilde / wxt_wtilde;
                                        ssx = Omega2 / doubleMinusKokkosComplex(std::pow(wxt,2) , wtilde2);
                                    }

                                    Kokkos::complex<double> ssxcutoffComplex = I_eps_array(ig,my_igp);
                                    ssxcutoff = sexcut*Kokkos::abs(ssxcutoffComplex);
                                    if((Kokkos::abs(ssx) > ssxcutoff) && (wxt < 0.0)) ssx = expr0;

                                    if(ig != igp)
                                    {
                                        ssxa[ig] = matngmatmgp*ssx + matngpmatmg * Kokkos::conj(ssx);
                                        scha[ig] = matngmatmgp*sch + matngpmatmg * Kokkos::conj(sch);
                                    }
                                    else
                                    {
                                        ssxa[ig] = matngmatmgp * ssx;
                                        scha[ig] = matngmatmgp * sch;
                                    }

                                    ssxt += ssxa[ig];
                                    scht += scha[ig];
                                }

                            }

>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
                        }
                        else
                        {
                            //344-394
                            for(int ig=0; ig<igmax; ++ig)
                            {
<<<<<<< HEAD
                                wtilde = wtilde_array(ig, my_igp);
                                wtilde2 = kokkos_square(wtilde,2);
=======
                                wtilde = wtilde_array(ig,my_igp);
                                wtilde2 = kokkos_square(wtilde);
                                Omega2 = I_eps_array(ig,my_igp) * wtilde;
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2

                                matngmatmgp = aqsntemp(ig,n1) * mygpvar1;
                                wdiff = doubleMinusKokkosComplex(wxt , wtilde);

                                cden = wdiff;
                                rden = Kokkos::real(cden * Kokkos::conj(cden));
                                rden = 1.00 / rden;
                                delw = rden * wtilde * Kokkos::conj(cden);
<<<<<<< HEAD
                                delwr = Kokkos::real(delw * Kokkos::conj(delw)); //This is diff from original ...
=======
                                delwr = Kokkos::real(delw * Kokkos::conj(delw));
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
                                wdiffr = Kokkos::real(wdiff * Kokkos::conj(wdiff));

                                if((wdiffr > limittwo) && (delwr < limitone))
                                {
<<<<<<< HEAD
                                    sch = delw;
                                    sch *= I_eps_array(ig,my_igp);
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

                                ssxcutoff = sexcut*abs(I_eps_array(ig,my_igp));
                                if((Kokkos::abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;

                                ssxa(ig) = matngmatmgp*ssx;
                                scha(ig) = matngmatmgp*sch;

                                ssxt() += ssxa(ig);
                                scht() += scha(ig);
                          }
                        }
                        ssx_array(iw) += ssxt();
                        sch_array(iw) += 0.5*scht();
                    }
//                for(int iw=nstart; iw<nend; ++iw)
//                    std::cout <<"sch_array[" << iw << "] = " << sch_array[iw] << "\t ssx_array[" << iw << "] = " << ssx_array[iw] << std::endl;
=======
                                    sch = I_eps_array( ig, my_igp) * delw;
                                    cden = kokkos_square(wxt);
                                    rden = Kokkos::real(cden * Kokkos::conj(cden));
                                    rden = 1.00 / rden;
                                    ssx = rden * Omega2 * Kokkos::conj(cden);
                              }
                                else if (delwr > to1)
                                {
                                    sch = 0.00;
                                    cden = 4.00 * wtilde2 * doublePlusKokkosComplex(0.50, delw);
                                    rden = Kokkos::real(cden * Kokkos::conj(cden));
                                    rden = 1.00/rden;
                                    ssx = rden * -Omega2 * Kokkos::conj(cden) * delw;
                                }
                                else
                                {
                                    sch = 0.00;
                                    ssx = 0.00;
                                }

                                Kokkos::complex<double> ssxcutoffComplex = I_eps_array(ig,my_igp);
                                ssxcutoff = sexcut*Kokkos::abs(ssxcutoffComplex);
                                if((abs(ssx) > ssxcutoff) && (abs(wxt) < 0.00)) ssx = 0.00;

                                ssxa[ig] = matngmatmgp*ssx;
                                scha[ig] = matngmatmgp*sch;

                                ssxt += ssxa[ig];
                                scht += scha[ig];
                            }
                        }
                        ssx_array[iw] += ssxt;
                        sch_array[iw] += 0.5*scht;
//                        std::cout << "scht - " <<  scht << std::endl;
                    }
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
                }
                else
                {
                    int igblk = 512;
<<<<<<< HEAD
//                    //403 - 479
=======
                    //403 - 479
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
                    for(int igbeg=0; igbeg<igmax; igbeg+=igblk)
                    {
                        int igend = min(igbeg+igblk-1, igmax);
                        for(int iw=nstart; iw<nend; ++iw)
                        {
<<<<<<< HEAD
                            scht() = ssxt() = expr0;
                            wxt = wx_array[iw];

                            int tmpVal = 0;

=======
                            scht = ssxt = 0.00;
                            wxt = wx_array[iw];

>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
                            if(gppsum == 1)
                            {
                                for(int ig= igbeg; ig<min(igend,igmax-1); ++ig)
                                {
                                    wtilde = wtilde_array(ig,my_igp);
<<<<<<< HEAD
                                    matngmatmgp = aqsntemp(ig,n1) * mygpvar1;
                                    wdiff = doubleMinusKokkosComplex(wxt , wtilde);
                                    delw = wtilde / wdiff;
                                    delw2 = Kokkos::real(delw * Kokkos::conj(delw));
                                    wdiffr = Kokkos::real(wdiff * Kokkos::conj(wdiff));
                                    if((abs(wdiffr) < limittwo) || (delw2 > limitone))
                                        scha_mult = 1.0;
                                    else 
                                        scha_mult = 0.0;

                                    sch = scha_mult * delw ;
                                    sch *= I_eps_array(ig, my_igp);
                                    scha(ig) = matngmatmgp*sch + Kokkos::conj(aqsmtemp(ig,n1)) * mygpvar2 * Kokkos::conj(sch);
                                    scht() += scha(ig);
=======
                                    matngmatmgp = mygpvar1 * aqsntemp(ig,n1) ;
                                    wdiff = doubleMinusKokkosComplex(wxt , wtilde);
                                    delw = wtilde / wdiff;
                                    delw2 = Kokkos::real(delw * conj(delw));
                                    wdiffr = Kokkos::real(wdiff * Kokkos::conj(wdiff));
                                    if((wdiffr < limittwo) || (delw2 > limitone))
                                        scha_mult = 1.0;
                                    else
                                        scha_mult = 0.0;

                                    Kokkos::complex<double> I_eps_temp = I_eps_array(ig, my_igp);
                                    sch = scha_mult * delw * I_eps_temp ;
                                    scha[ig] = matngmatmgp*sch + Kokkos::conj(aqsmtemp(ig,n1)) * mygpvar2 * Kokkos::conj(sch);
                                    scht += scha[ig];
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
                                }
                                if(igend == (igmax-1))
                                {
                                    int ig = igmax;
                                    wtilde = wtilde_array(ig,my_igp);
<<<<<<< HEAD
                                    matngmatmgp = aqsntemp(ig,n1) * mygpvar1;
                                    wdiff = doubleMinusKokkosComplex(wxt,wtilde);
                                    delw = wtilde / wdiff;
                                    delw2 = Kokkos::real(delw * Kokkos::conj(delw));
                                    wdiffr = Kokkos::real(wdiff * Kokkos::conj(wdiff));
=======
                                    matngmatmgp = mygpvar1 * aqsntemp(ig,n1) ;
                                    wdiff = doubleMinusKokkosComplex(wxt , wtilde);
                                    delw = wtilde / wdiff;
                                    delw2 = Kokkos::real(delw * conj(delw));
                                    wdiffr = Kokkos::real(wdiff * conj(wdiff));
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
                                    if((wdiffr < limittwo) || (delw2 > limitone))
                                        scha_mult = 1.0;
                                    else scha_mult = 0.0;

<<<<<<< HEAD
                                    sch = scha_mult * delw ;
                                    sch *= I_eps_array(ig, my_igp);
                                    scha(ig) = matngmatmgp * sch;
                                    scht() += scha(ig);
=======
                                    Kokkos::complex<double> I_eps_temp = I_eps_array(ig, my_igp);
                                    sch = scha_mult * delw * I_eps_temp ;
//                                    sch = scha_mult * delw * I_eps_array(ig,my_igp);
                                    scha[ig] = matngmatmgp * sch;
                                    scht += scha[ig];
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
                                }
                            }
                            else
                            {
                                for(int ig = igbeg; ig<min(igend,igmax); ++ig)
                                {
                                    wdiff = doubleMinusKokkosComplex(wxt , wtilde_array(ig,my_igp));
                                    cden = wdiff;
                                    rden = Kokkos::real(cden * Kokkos::conj(cden));
                                    rden = 1.00/rden;
<<<<<<< HEAD
                                    delw = rden * wtilde_array(ig,my_igp) * Kokkos::conj(cden);
                                    delwr = Kokkos::real(delw * conj(delw));
                                    wdiffr = Kokkos::real(wdiff * conj(wdiff));

                                    scha(ig) = mygpvar1 * delw * aqsntemp(ig,n1);
                                    scha(ig) *= I_eps_array(ig, my_igp);

                                    if((wdiffr > limittwo) && (delwr < limitone)) scht() += scha(ig);
                                }
                            }
                            sch_array(iw) += 0.5*scht();
                        }
                    }
                }
//                for(int iw=nstart; iw<nend; ++iw)
//                    std::cout << "iw = " << iw << "\t sch_array " << sch_array[iw] << "\t n1 = " << n1 << "\t igp " << igp << std::endl;


            if(flag_occ)
            {
                for(int iw=nstart; iw<nend; ++iw)
                {
//#pragma omp critical
                    asxtemp(iw) += occ * ssx_array(iw) * vcoul(igp); //occ does not change and is 1.00 so why not remove it.
                }
            }
//
//#pragma omp critical
//    {
            for(int iw=nstart; iw<nend; ++iw)
                achtemp(iw) += vcoul(igp) * sch_array(iw);

            acht_n1_loc(n1) += sch_array(2) * vcoul(igp);
//    }

            } //for the if-loop to avoid break inside an openmp pragma statment
        }
    });

    double end_time = omp_get_wtime(); //End timing here

    for(int iw=nstart; iw<nend; ++iw)
        cout << "achtemp[" << iw << "] = " << achtemp(iw) << endl;

    cout << "********** Time Taken **********= " << end_time - start_time << " secs" << endl;

    }
    Kokkos::finalize();

=======
                                    delw = rden * wtilde_array(ig,my_igp) * conj(cden);
                                    delwr = Kokkos::real(delw * Kokkos::conj(delw));
                                    wdiffr = Kokkos::real(wdiff * Kokkos::conj(wdiff));

//                                    scha[ig] = mygpvar1 * aqsntemp(ig,n1) * delw * I_eps_array(ig,my_igp);
                                    Kokkos::complex<double> I_eps_temp = I_eps_array(ig, my_igp);
                                    scha[ig] = mygpvar1 * delw * aqsntemp(ig,n1) *  I_eps_temp;

                                    if((wdiffr > limittwo) && (delwr < limitone)) 
                                        scht += scha[ig];

                                }
                            }
                            sch_array[iw] += 0.5*scht;
                        }
                    }

                    if(flag_occ)
                    {
                        for(int iw=nstart; iw<nend; ++iw)
                        {
                            Kokkos::View<complex<double>> ssx_array_view ("ssx_array_view");
                            ssx_array_view() = ssx_array[iw];
                            asxtemp(iw) += occ * vcoul(igp) * ssx_array_view() ; //occ does not change and is 1.00 so why not remove it.

                        }
                    }
            
                    for(int iw=nstart; iw<nend; ++iw)
                    {
                        Kokkos::View<complex<double>> sch_array_iw ("sch_array_iw");
                        sch_array_iw() = sch_array[iw];
                        Kokkos::View<complex<double>> achtemp_back ("achtemp_back");
                        achtemp_back() = achtemp(iw);
                        achtemp(iw) += sch_array_iw() * vcoul(igp);
//                        std::cout << "sch_array[" << iw << "][" << sch_array[iw] << "\t vcouls[ " << igp <<  "= " << vcoul(igp) << std::endl;
                    }

                    Kokkos::View<complex<double>> sch_array_2 ("sch_array_2");
                    sch_array_2() = sch_array[2];
                    acht_n1_loc(n1) += sch_array_2() * vcoul(igp);
                } 
            }//for the if-loop to avoid break inside an openmp pragma statment
        }
    });

//    double end_time = omp_get_wtime(); //End timing here

    for(int iw=nstart; iw<nend; ++iw)
        std::cout << "achtemp[" << iw << "] = " << achtemp(iw) << std::endl;

//    cout << "********** Time Taken **********= " << end_time - start_time << " secs" << endl;

//        Kokkos::OpenMP::finalize();

    }
    Kokkos::finalize();
>>>>>>> 4bdcc9aa5d2975ca11f83268ad68d2f58ce5ada2
    return 0;
}

//Almost done code
