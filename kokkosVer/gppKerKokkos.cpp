#include <gppKerKokkos.h>  

Kokkos::complex<double> kokkos_square(Kokkos::complex<double> compl_num, int n)
{
    double re = Kokkos::real(compl_num);
    double im = Kokkos::imag(compl_num);

    Kokkos::complex<double> result(re*re - im*im, 2*re*im);
    return result;
}

Kokkos::complex<double> doubleMinusKokkosComplex(double op1, Kokkos::complex<double> op2)
{
    Kokkos::complex<double> expr((op1 - Kokkos::real(op2)), (0 - Kokkos::imag(op2)));
    return expr;
}

Kokkos::complex<double> doublePlusKokkosComplex(double op1, Kokkos::complex<double> op2)
{
    Kokkos::complex<double> expr((op1 + Kokkos::real(op2)), (0 + Kokkos::imag(op2)));
    return expr;
}

Kokkos::complex<double> doubleMultKokkosComplex(double op1, Kokkos::complex<double> op2)
{
    Kokkos::complex<double> expr((op1 * Kokkos::real(op2)), (0 * Kokkos::imag(op2)));
    return expr;
}

void reduce_achstemp(int n1, Kokkos::View<int*> inv_igp_index, int ncouls, Kokkos::View<Kokkos::complex<double>** > aqsmtemp, Kokkos::View<Kokkos::complex<double>** > aqsntemp, Kokkos::View<Kokkos::complex<double>** > I_eps_array, Kokkos::complex<double>& achstemp, Kokkos::View<int*> indinv, int ngpown, Kokkos::View<double*> vcoul)
{
    Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, Kokkos::complex<double> &achstempUpdate)
    {
        Kokkos::complex<double> mygpvar1, mygpvar2;
        Kokkos::complex<double> schstemp(0.0, 0.0);
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

void flagOCC_solver(double wxt, Kokkos::View<Kokkos::complex<double>** > wtilde_array, int my_igp, int n1, Kokkos::View<Kokkos::complex<double>** > aqsmtemp, Kokkos::View<Kokkos::complex<double>** > aqsntemp, Kokkos::View<Kokkos::complex<double>** > I_eps_array, Kokkos::complex<double> &ssxt, Kokkos::complex<double> &scht, int ncouls, int igp)
{
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
        int nstart = 0, nend = 3;
        double occ=1.0;

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

        int npes = 1; //Represents the number of ranks per node
        int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

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


        Kokkos::complex<double> expr(0.5 , 0.5);
        Kokkos::complex<double> expr0(0.0 , 0.0);
        Kokkos::complex<double> achstemp(0.0 , 0.0);
        Kokkos::View<complex<double>> addVal("addVal");

        /********************KOKKOS RELATED VARS AND VIEWS ***************************/

        ViewVectorTypeInt inv_igp_index("inv_igp_index", ngpown);
        ViewVectorTypeInt indinv("indinv", ncouls);
        ViewVectorTypeDouble vcoul ("vcoul", ncouls);
        ViewVectorTypeDouble wx_array ("wx_array", 3);

        ViewMatrixTypeComplex aqsmtemp ("aqsmtemp", number_bands, ncouls);
        ViewMatrixTypeComplex aqsntemp ("aqsntemp", number_bands, ncouls);
        ViewMatrixTypeComplex I_eps_array ("I_eps_array", ngpown, ncouls);
        ViewMatrixTypeComplex wtilde_array ("wtilde_array", ngpown, ncouls);

        ViewVectorTypeComplex asxtemp ("asxtemp", nend-nstart);
        ViewVectorTypeComplex acht_n1_loc(" acht_n1_loc", number_bands);
        ViewVectorTypeComplex ssx_array(" ssx_array", 3);
        ViewVectorTypeComplex sch_array(" sch_array", 3);
        ViewVectorTypeComplex scha ("scha", ncouls);

        ViewScalarTypeComplex delw("delw");
        ViewScalarTypeComplex sch("sch");
        ViewScalarTypeComplex wdiff("wdiff");
        ViewScalarTypeComplex rden("rden");
        ViewScalarTypeComplex mygpvar1("mygpvar1");

        //Create host mirrors of device views
        ViewMatrixTypeComplex::HostMirror host_aqsntemp = Kokkos::create_mirror_view(aqsntemp);
        ViewMatrixTypeComplex::HostMirror host_aqsmtemp = Kokkos::create_mirror_view(aqsmtemp);
        ViewMatrixTypeComplex::HostMirror host_I_eps_array = Kokkos::create_mirror_view(I_eps_array);
        ViewMatrixTypeComplex::HostMirror host_wtilde_array= Kokkos::create_mirror_view(wtilde_array); 
        ViewVectorTypeInt::HostMirror host_inv_igp_index = Kokkos::create_mirror_view(inv_igp_index);
        ViewVectorTypeInt::HostMirror host_indinv = Kokkos::create_mirror_view(indinv);
        ViewVectorTypeDouble::HostMirror host_vcoul = Kokkos::create_mirror_view(vcoul);
        ViewVectorTypeDouble::HostMirror host_wx_array = Kokkos::create_mirror_view(wx_array);

        for(int i = 0; i< number_bands; i++)
           for(int j=0; j<ncouls; j++)
           {
               host_aqsmtemp(i,j) = expr;
               host_aqsntemp(i,j) = expr;
           }

        for(int i = 0; i< ngpown; i++)
           for(int j=0; j<ncouls; j++)
           {
               host_I_eps_array(i,j) = expr;
               host_wtilde_array(i,j) = expr;
           }
        
       for(int i=0; i<ncouls; i++)
           host_vcoul(i) = 1.0;



        cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
        cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
        cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;

        //For MPI Work distribution
        for(int ig=0; ig < ngpown; ++ig)
            host_inv_igp_index(ig) = (ig+1) * ncouls / ngpown;

        //Do not know yet what this array represents
        for(int ig=0; ig<ncouls; ++ig)
            host_indinv(ig) =ig;

        Kokkos::deep_copy(aqsmtemp, host_aqsmtemp);
        Kokkos::deep_copy(aqsntemp, host_aqsntemp);
        Kokkos::deep_copy(I_eps_array, host_I_eps_array);
        Kokkos::deep_copy(wtilde_array, host_wtilde_array);
        Kokkos::deep_copy(inv_igp_index, host_inv_igp_index);
        Kokkos::deep_copy(indinv, host_indinv);
        Kokkos::deep_copy(vcoul, host_vcoul);

    //********************** Structures to update arrays inside Kokkos parallel calls ************************************************
    //****** achtemp **********
//        Kokkos::complex<double>  achtemp[nend-nstart];
//        achtempStruct achtempVar = {{achtemp[0],achtemp[1],achtemp[2]}}; 


    //**********************************************************************************************************************************

        auto start_chrono = std::chrono::high_resolution_clock::now();

        for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
        {
            reduce_achstemp(n1, inv_igp_index, ncouls, aqsmtemp, aqsntemp, I_eps_array, achstemp,  indinv, ngpown, vcoul);

            for(int iw=nstart; iw<nend; ++iw)
            {
                host_wx_array(iw) = e_lk - e_n1kq + dw*((iw+1)-2);
                if(host_wx_array(iw) < to1) host_wx_array(iw) = to1;
            }

          Kokkos::parallel_reduce(range_policy(0, ngpown), KOKKOS_LAMBDA (int my_igp, achtempStruct& achtempVarUpdate)
//              for(int my_igp=0; my_igp<ngpown; ++my_igp)
            {
                bool flag_occ = n1 < nvband;
                double wxt;
                Kokkos::complex<double> scht, ssxt;
                int indigp = inv_igp_index(my_igp);
                int igp = indinv(indigp);
                if(indigp == ncouls)
                    igp = ncouls-1;

                if(!(igp > ncouls || igp < 0)){
                
                for(int i=0; i<3; i++)
                {
                    ssx_array(i) = expr0;
                    sch_array(i) = expr0;
                }

                if(flag_occ)
                {
                    for(int iw=nstart; iw<nend; ++iw)
                    {
                        scht = ssxt = expr0;
                        wxt = wx_array(iw);
                        flagOCC_solver(wxt, wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp);

                        ssx_array(iw) += ssxt;
                        sch_array(iw) += 0.5*scht;
                    }
                }
                else
                {
                    int igmax = ncouls;
                    mygpvar1() = Kokkos::conj(aqsmtemp(n1,igp));

                    for(int iw=nstart; iw<nend; ++iw)
                    {
                        scht = ssxt = expr0;
                        wxt = wx_array(iw);

                        for(int ig = 0; ig<igmax; ++ig)
                        {
                            wdiff() = doubleMinusKokkosComplex(wxt , wtilde_array(my_igp, ig));
                            rden() = (Kokkos::complex<double>) 1.00/(wdiff() * Kokkos::conj(wdiff()));
                            delw() = rden() * wtilde_array(my_igp, ig) * Kokkos::conj(wdiff());
                            scha(ig) = mygpvar1() * delw() * aqsntemp(n1,ig)* I_eps_array(my_igp, ig);
                        } // for ig

                        for(int ig = 0; ig<igmax; ++ig)
                            scht+=scha[ig];

                        sch_array(iw) += 0.5*scht;
                    } // for nstart - nend
                } //else-loop
              } // if-condition

                if(flag_occ)
                {
                    for(int iw=nstart; iw<nend; ++iw)
                        asxtemp(iw) += occ * ssx_array(iw) ; 
                }

                for(int iw=nstart; iw<nend; ++iw)
                    achtempVarUpdate.value[iw] += vcoul(igp) * sch_array(iw);

                    acht_n1_loc(n1) += vcoul(igp) * sch_array(2);


            },achtempVar); // for - ngpown 

            //copy it into a diff buffer not related to kokkos-views so that the value is not modified at the start of each iteration.
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

//Almost done code
