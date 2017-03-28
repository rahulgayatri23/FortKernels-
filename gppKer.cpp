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

    int npes = 8; //Represents the number of ranks per node
    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

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

   for(int i=0; i<ncouls; i++)
   {
       for(int j=0; j<number_bands; j++)
       {
           aqsmtemp[i][j] = expr;
           aqsntemp[i][j] = expr;
       }

       for(int j=0; j<ngpown; j++)
           I_eps_array[i][j] = expr;
   }

    cout << "Size of I_eps_array array = " << (ngpown) << " bytes" << endl;

//    cout << "aqsmtemp[0][0].real = " << aqsmtemp[2][1].real() << "\t aqsmtemp[0][0].imag = " << aqsmtemp[2][2].imag() << endl;

    //For MPI Work distribution
//    for(int ig=0; ig < ngpown; ++ig)
//        inv_igp_index[ig] = ig * ncouls / ngpown;
//
//    //Do not know yet what this array represents
//    for(int ig=0; ig<ncouls; ++ig)
//        indinv[ig] =ig;


//    for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
//    {
//        double flag_occ, occ=1.0;
//        if(n1 < nvband)
//            flag_occ = limittwo;
//
//        for(int my_igp = 0; my_igp< ngpown; ++my_igp)
//        {
//            int indigp = inv_igp_index[my_igp];
//            int igp = indinv[indigp];
//
//            if(igp > ncouls || igp < 0)
//                break;
//
//            if(gppsum == 1)
//                igmax = igp;
//            else
//                igmax = ncouls;
//
//
//            std::complex<double> mygpvar1 = std::conj(aqsmtemp[igp][n1]);
//            std::complex<double> mygpvar2 = aqsntemp[igp][n1];
//            double schstemp = 0.00;
//
//            if(gppsum == 1)
//            {
//                for(int ig=0; ig<igmax-1; ++ig)
//                {
//
//                }
//            }
//
//
//        }
//    }


    cout << "EXIT EXIT EXIT" << endl;
    return 0;
}
