#include "gppKerKokkos.h"

Kokkos::complex<double> kokkos_square(Kokkos::complex<double> compl_num, int n)
{
    std::complex<double> powerExpr(Kokkos::real(compl_num), Kokkos::imag(compl_num));
    powerExpr = std::pow(powerExpr,n);

    return powerExpr;
}

Kokkos::complex<double> doubleMinusKokkosComplex(double op1, Kokkos::complex<double> op2)
{
    Kokkos::complex<double> result((op1 - Kokkos::real(op2)), (0 - Kokkos::imag(op2)));
    return result;
}

Kokkos::complex<double> doublePlusKokkosComplex(double op1, Kokkos::complex<double> op2)
{
    Kokkos::complex<double> result((op1 + Kokkos::real(op2)), (0 + Kokkos::imag(op2)));
    return result;
}

Kokkos::complex<double> doubleMultKokkosComplex(double op1, Kokkos::complex<double> op2)
{
    Kokkos::complex<double> result((op1 * Kokkos::real(op2)), (0 * Kokkos::imag(op2)));
    return result;
}

void reduce_achstemp(int n1, ViewVectorTypeInt inv_igp_index, int ncouls, ViewMatrixTypeComplex aqsmtemp, ViewMatrixTypeComplex aqsntemp, ViewMatrixTypeComplex I_eps_array, Kokkos::complex<double>& achstemp, Kokkos::View<int*> indinv, int ngpown, ViewVectorTypeDouble vcoul)
{
    Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, Kokkos::complex<double> &achstempUpdate)
    {
        int indigp = inv_igp_index(my_igp);
        int igp = indinv(indigp);
        if(indigp == ncouls)
            igp = ncouls-1;

        if(!(igp > ncouls || igp < 0))
        {
            mygpvar1 = Kokkos::conj(aqsmtemp(n1,igp));
            mygpvar2 = aqsntemp(n1,igp);

            schs = -I_eps_array(my_igp,igp);
            matngmatmgp = aqsntemp(n1,igp) * mygpvar1;

            if(abs(schs) > to1)
                schstemp = schstemp + matngmatmgp * schs;

        }
        else
        {
            for(int ig=1; ig<ncouls; ++ig)
                schstemp = schstemp - aqsntemp(n1,igp) * I_eps_array(my_igp,ig) * mygpvar1;
        }
        achstempUpdate += 0.5 * vcoul(igp) * schstemp;
    },achstemp);
}

void flagOCC_solver(double wxt, ViewMatrixTypeComplex wtilde_array, int my_igp, int n1, ViewMatrixTypeComplex aqsmtemp, ViewMatrixTypeComplex aqsntemp, ViewMatrixTypeComplex I_eps_array, Kokkos::complex<double> &ssxt, Kokkos::complex<double> &scht, int ncouls, int igp)
{
    Kokkos::complex<double> ssxa[ncouls], scha[ncouls];
    int igmax = ncouls;

    for(int ig=0; ig<igmax; ++ig)
    {
        wtilde = wtilde_array(my_igp,ig);
        wtilde2 = kokkos_square(wtilde,2);
        Omega2 = wtilde2*I_eps_array(my_igp,ig);
        mygpvar1 = Kokkos::conj(aqsmtemp(n1,igp));
        matngmatmgp = aqsntemp(n1,ig) * mygpvar1;
        Kokkos::complex<double> sch, ssx;

        wdiff = doubleMinusKokkosComplex(wxt , wtilde);

        Kokkos::complex<double> cden = wdiff;
        double rden = Kokkos::real(cden * Kokkos::conj(cden));
        rden = 1.00 / rden;
        delw = rden * wtilde * Kokkos::conj(cden);
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

    int igmax = ncouls;

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

//    ViewScalarTypeComplex scht("scht");
//    ViewScalarTypeComplex scht("ssxt");

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
//    Kokkos::complex<double>  achtemp[nend-nstart];
//    struct achtempStruct 
//    {
//         Kokkos::complex<double> value[3];
//        void operator+=(achtempStruct const& other) 
//        {
//            for (int i = 0; i < 3; ++i) 
//                value[i] += other.value[i];
//        }
//        void operator+=(achtempStruct const volatile& other) volatile 
//        {
//            for (int i = 0; i < 3; ++i) 
//                value[i] += other.value[i];
//        }
//    };
//    achtempStruct achtempVar = {{achtemp[0],achtemp[1],achtemp[2]}}; 


//**********************************************************************************************************************************

    double start_time = omp_get_wtime(); //Start timing here.

    for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
    {
        double wx_array[3];
        bool flag_occ = n1 < nvband;

        reduce_achstemp(n1, inv_igp_index, ncouls, aqsmtemp, aqsntemp, I_eps_array, achstemp,  indinv, ngpown, vcoul);

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(wx_array[iw] < to1) wx_array[iw] = to1;
        }

        Kokkos::parallel_reduce(ngpown, KOKKOS_LAMBDA (int my_igp, achtempStruct& achtempVarUpdate)       //for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            Kokkos::complex<double> ssx_array[3], sch_array[3];
            Kokkos::complex<double> scht, ssxt;

            double wxt;
            int indigp = inv_igp_index(my_igp);
            int igp = indinv(indigp);
            if(indigp == ncouls)
                igp = ncouls-1;

            if(!(igp > ncouls || igp < 0))
            {
            
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
                    Kokkos::complex<double> scha[ncouls];

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
                    asxtemp(iw) +=occ * ssx_array[iw]  ; //occ does not change and is 1.00 so why not remove it.
            }

            for(int iw=nstart; iw<nend; ++iw)
                achtempVarUpdate.value[iw] += vcoul(igp) * sch_array[iw];

                acht_n1_loc(n1) += vcoul(igp) * sch_array[2] ;
        },achtempVar); // for - ngpown 

        //Rahul - have to copy it into a diff buffer not related to kokkos-views so that the value is not modified at the start of each iteration.
        for(int iw=nstart; iw<nend; ++iw)
            achtemp[iw] += achtempVar.value[iw];
    } // for - number_bands

   
    double end_time = omp_get_wtime(); //End timing here

    for(int iw=nstart; iw<nend; ++iw)
        cout << "Final achtemp[" << iw << "] = " << achtemp[iw] << endl;

    cout << "********** Time Taken **********= " << end_time - start_time << " secs" << endl;

    }
    Kokkos::finalize();

    return 0;
}

//Almost done code
