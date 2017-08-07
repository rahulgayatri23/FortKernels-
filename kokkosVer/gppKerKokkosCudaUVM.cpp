/*Kokkos code for OpenMP version*/
#include "gppKerKokkos.h"

KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> kokkos_square(Kokkos::complex<double> compl_num, int n)
{
    double re = Kokkos::real(compl_num);
    double im = Kokkos::imag(compl_num);

    Kokkos::complex<double> result(re*re - im*im, 2*re*im);
    return result;
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
void reduce_achstemp(ViewScalarTypeComplex mygpvar1, ViewScalarTypeComplex schstemp, int n1, ViewVectorTypeInt inv_igp_index, int ncouls, ViewMatrixTypeComplex aqsmtemp, ViewMatrixTypeComplex aqsntemp, ViewMatrixTypeComplex I_eps_array, Kokkos::complex<double>& achstemp, ViewVectorTypeInt indinv, int ngpown, Kokkos::View<double*> vcoul)
{
    double to1 = 1e-6;
    Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, Kokkos::complex<double> &achstempUpdate)
    {
        int indigp = inv_igp_index(my_igp);
        int igp = indinv(indigp);
        if(indigp == ncouls)
            igp = ncouls-1;

        if(!(igp > ncouls || igp < 0)){

            mygpvar1() = Kokkos::conj(aqsmtemp(n1,igp));

            if(Kokkos::abs(-I_eps_array(my_igp,igp)) > to1)
                schstemp() += (aqsntemp(n1,igp) * mygpvar1()) * (-I_eps_array(my_igp,igp));
        }
        else
        {
            for(int ig=1; ig<ncouls; ++ig)
                schstemp() = schstemp() - aqsntemp(n1,igp) * I_eps_array(my_igp,ig) * mygpvar1();
        }

        achstempUpdate += 0.5 * vcoul(igp) * schstemp();
    },achstemp);
}

KOKKOS_INLINE_FUNCTION
void flagOCC_solver(ViewScalarTypeComplex mygpvar1, double wxt, ViewMatrixTypeComplex wtilde_array, int my_igp, int n1, ViewMatrixTypeComplex aqsmtemp, ViewMatrixTypeComplex aqsntemp, ViewMatrixTypeComplex I_eps_array, Kokkos::complex<double> &ssxt, Kokkos::complex<double> &scht, int ncouls, int igp)
{

    for(int ig=0; ig<ncouls; ++ig)
    {
        Kokkos::complex<double> wtilde = wtilde_array(my_igp,ig);
        Kokkos::complex<double> wtilde2 = kokkos_square(wtilde,2);
        Kokkos::complex<double> Omega2 = wtilde2*I_eps_array(my_igp,ig);
        Kokkos::complex<double> matngmatmgp = aqsntemp(n1,ig) * mygpvar1();
        Kokkos::complex<double> expr0( 0.0 , 0.0);
        double to1 = 1e-6;
        double sexcut = 4.0;
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

        ssxt += matngmatmgp*ssx;
        scht += matngmatmgp*sch;
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
    int ngpown = ncouls / nodes_per_group ; //Number of gvectors per mpi task
    double e_lk = 10;
    double e_n1kq= 6.0; 
    double dw = 1;
    double to1 = 1e-6;
    int nstart = 0, nend = 3;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);
    double sexcut = 4.0;

    Kokkos::complex<double> expr0( 0.0 , 0.0);
    Kokkos::complex<double> expr( 0.5 , 0.5);

    //Printing out the params passed.
    std::cout << "number_bands = " << number_bands \
        << "\t nvband = " << nvband \
        << "\t ncouls = " << ncouls \
        << "\t nodes_per_group  = " << nodes_per_group \
        << "\t ngpown = " << ngpown \
        << "\t nend = " << nend \
        << "\t nstart = " << nstart \
        << "\t sexcut = " << sexcut \
        << "\t limitone = " << limitone \
        << "\t limittwo = " << limittwo << endl;


    double occ=1.0;

    /********************KOKKOS RELATED VARS AND VIEWS ***************************/
    ViewVectorTypeInt inv_igp_index("inv_igp_index", ngpown);
    ViewVectorTypeInt indinv("indinv", ncouls);
    ViewVectorTypeDouble vcoul("vcoul", ncouls);

    ViewMatrixTypeComplex aqsmtemp("aqsmtemp", number_bands, ncouls);
    ViewMatrixTypeComplex aqsntemp("aqsntemp", number_bands, ncouls);
    ViewMatrixTypeComplex I_eps_array("I_eps_array", ngpown, ncouls);
    ViewMatrixTypeComplex wtilde_array("wtilde_array", ngpown, ncouls);

    ViewVectorTypeComplex asxtemp("asxtemp", nend-nstart);
    ViewVectorTypeComplex acht_n1_loc("acht_n1_loc", number_bands);
    ViewVectorTypeComplex scha("scha", ncouls);

    ViewVectorTypeDouble wx_array("wx_array",3);

    ViewScalarTypeComplex mygpvar1("mygpvar1");
    ViewScalarTypeComplex schstemp("schstemp");

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

//**********************************************************************************************************************************

    auto start_chrono = std::chrono::high_resolution_clock::now();

    for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
    {
        reduce_achstemp(mygpvar1, schstemp, n1, inv_igp_index, ncouls, aqsmtemp, aqsntemp, I_eps_array, achstemp,  indinv, ngpown, vcoul);

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array(iw) = e_lk - e_n1kq + dw*((iw+1)-2);
            if(wx_array(iw) < to1) wx_array(iw) = to1;
        }
//         cout << "achstemp = " << achstemp << endl;

      Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, achtempStruct& achtempVarUpdate)
//          for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            bool flag_occ = n1 < nvband;
            Kokkos::complex<double> ssx_array[3], sch_array[3];

            Kokkos::complex<double> scht, ssxt;
            int indigp = inv_igp_index(my_igp);
            int igp = indinv(indigp);
            if(indigp == ncouls)
                igp = ncouls-1;

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
                    flagOCC_solver(mygpvar1, wx_array(iw), wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp);

                    ssx_array[iw] += ssxt;
                    sch_array[iw] += 0.5*scht;
                }
            }
            else
            {
                int igblk = 512;
                mygpvar1() = Kokkos::conj(aqsmtemp(n1,igp));
                Kokkos::complex<double> delw, wdiff;
                for(int igbeg=0; igbeg<ncouls; igbeg+=igblk)
                {
                    int igend = min(igbeg+igblk-1, ncouls);
                    for(int iw=nstart; iw<nend; ++iw)
                    {
                        scht = ssxt = expr0;

                        for(int ig = igbeg; ig<min(igend,ncouls); ++ig)
                        {
                            wdiff = doubleMinusKokkosComplex(wx_array[iw] , wtilde_array(my_igp, ig));
                            double rden = Kokkos::real(wdiff * Kokkos::conj(wdiff));
                            rden = 1.00/rden;
                            delw = rden * wtilde_array(my_igp, ig) * Kokkos::conj(wdiff);
                            scha(ig) = mygpvar1() * delw * aqsntemp(n1,ig) * I_eps_array(my_igp,ig);
                            scht += scha(ig);

                        } // for iw-blockSize

                            sch_array[iw] += 0.5*scht;

                    } // for nstart - nend
                } //for - ig-Block
            } //else-loop
      } // if-condition

            if(flag_occ)
            {
                for(int iw=nstart; iw<nend; ++iw)
                    asxtemp(iw) += occ * ssx_array[iw];
            }

            for(int iw=nstart; iw<nend; ++iw)
                achtempVarUpdate.value[iw] += vcoul(igp) * sch_array[iw];

            acht_n1_loc(n1) += sch_array[2] * vcoul(igp);
        },achtempVar); // for - ngpown 

        //Rahul - have to copy it into a diff buffer not related to kokkos-views so that the value is not modified at the start of each iteration.
        for(int iw=nstart; iw<nend; ++iw)
            achtemp[iw] += achtempVar.value[iw];
    } // for - number_bands

   
        auto end_chrono = std::chrono::high_resolution_clock::now();

        for(int iw=nstart; iw<nend; ++iw)
            cout << "Final achtemp[" << iw << "] = " << achtemp[iw] << endl;

        std::chrono::duration<double> elapsed_chrono = end_chrono - start_chrono;

        cout << "********** Chrono Time Taken **********= " << elapsed_chrono.count() << " secs" << endl;

    }
    Kokkos::finalize();

    return 0;
}
