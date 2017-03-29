#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <complex>

//template <class T>
//class complex;
using namespace std;

int main(int argc, char** argv)
{

    if (argc != 6)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> <gppsum> " << endl;
    }
    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int nodes_per_group = atoi(argv[4]);
    int gppsum = atoi(argv[5]);

    int igmax = ncouls;

    //    The below 3 params have to be changed for the final version. Currently using small numbers hence setting them to smaller ones...memory constraints on the local machine.
//    int npes = 8; //Represents the number of ranks per node
//    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task
//    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

    int npes = 8; //Represents the number of ranks per node
    int ngpown = ncouls / npes; //Number of gvectors per mpi task

    double e_lk = 10;
    double dw = 1;
    int nstart = 1, nend = 3;

    int inv_igp_index[ngpown];
    int indinv[ncouls];


    double to1 = 1e-6;
    std::cout << setprecision(16) << "to1 = " << to1 << endl;

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
        << "\t gamma = " << gamma \
        << "\t sexcut = " << sexcut \
        << "\t limitone = " << limitone \
        << "\t limittwo = " << limittwo << endl;


    //ALLOCATE statements from fortran gppkernel.

    std::complex<double> expr = 0.5 + 0.5i;
    std::complex<double> aqsmtemp[ncouls][number_bands];
    std::complex<double> aqsntemp[ncouls][number_bands];
    std::complex<double> I_eps_array[ncouls][ngpown];
    std::complex<double> wx_array[3];

    double vcoul[ncouls];



   for(int i=0; i<ncouls; i++)
   {
       for(int j=0; j<number_bands; j++)
       {
           aqsmtemp[i][j] = expr;
           aqsntemp[i][j] = expr;
       }

       for(int j=0; j<ngpown; j++)
           I_eps_array[i][j] = expr;

       vcoul[i] = 1.0;
   }

    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) << " bytes" << endl;

//    cout << "aqsmtemp[0][0].real = " << aqsmtemp[2][1].real() << "\t aqsmtemp[0][0].imag = " << aqsmtemp[2][2].imag() << endl;

    //For MPI Work distribution
    for(int ig=0; ig < ngpown; ++ig)
        inv_igp_index[ig] = ig * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0; ig<ncouls; ++ig)
        indinv[ig] =ig;


    for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
    {
        double flag_occ, occ=1.0;
        if(n1 < nvband)
            flag_occ = limittwo;

        std::complex<double> achstemp = std::complex<double>(0.0, 0.0);

        for(int my_igp = 0; my_igp< ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            if(igp > ncouls || igp < 0)
                break;

            if(gppsum == 1)
                igmax = igp;
            else
                igmax = ncouls;


            std::complex<double> mygpvar1 = std::conj(aqsmtemp[igp][n1]);
            std::complex<double> mygpvar2 = aqsntemp[igp][n1];
            std::complex<double> schstemp = std::complex<double>(0.0, 0.0);
            std::complex<double> schs = std::complex<double>(0.0, 0.0);
            std::complex<double> matngmatmgp = std::complex<double>(0.0, 0.0);
            std::complex<double> matngpmatmg = std::complex<double>(0.0, 0.0);

            if(gppsum == 1)
            {

                //Aggregating results in schstemp
                for(int ig=0; ig<igmax-1; ++ig)
                {
                    //std::complex<double> schs = I_eps_array[ig][my_igp];
                    schs = I_eps_array[ig][my_igp];
                    matngmatmgp = aqsntemp[ig][n1] * mygpvar1;
                    matngpmatmg = std::conj(aqsmtemp[ig][n1]) * mygpvar2;
                    schstemp = schstemp + matngmatmgp*schs + matngpmatmg*(std::conj(schs));

                }
                //ig = igp ;
                schs = I_eps_array[igp][my_igp];
                matngmatmgp = aqsntemp[igp][n1] * mygpvar1;

                if(abs(schs) > to1)
                    schstemp = schstemp + matngmatmgp * schs;
            }
            else
            {
                for(int ig=1; ig<igmax; ++ig)
                    schstemp = schstemp - aqsntemp[ig][n1] * I_eps_array[ig][my_igp] * mygpvar1;
            }

            achstemp = achstemp + schstemp*vcoul[igp]*0.5;
        }

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*(iw-2);
            if(abs(wx_array[iw]) < to1) wx_array[iw] = to1;

        }
    }


    cout << "EXIT EXIT EXIT" << endl;
    return 0;
}


//This code is till 250 lines of the original fortran code.
