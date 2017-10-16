#include <iostream>
#include <cstdlib>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
using namespace std;


int main(int argc, char** argv)
{
    typedef Kokkos::Cuda     ExecSpace;
    typedef Kokkos::CudaUVMSpace  MemSpace;
    typedef Kokkos::LayoutLeft   Layout;
    typedef Kokkos::RangePolicy<ExecSpace>  range_policy;
    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;

    typedef Kokkos::View<Kokkos::complex<double>, Layout, MemSpace>   ViewScalarTypeComplex;
    typedef Kokkos::View<Kokkos::complex<double>*, Layout, MemSpace>   ViewVectorTypeComplex;
    typedef Kokkos::View<Kokkos::complex<double>**, Layout, MemSpace>  ViewMatrixTypeComplex;

    typedef Kokkos::View<int*, Layout, MemSpace>   ViewVectorTypeInt;
    typedef Kokkos::View<double*, Layout, MemSpace>   ViewVectorTypeDouble;

    typedef Kokkos::View<int, Layout, MemSpace>   ViewScalarTypeInt;
    typedef Kokkos::View<double, Layout, MemSpace>   ViewScalarTypeDouble;

    Kokkos::initialize(argc, argv);
    {

        int number_bands = 32, ngpown = 32, ncouls = 32;

        struct achtempStruct 
        {
            Kokkos::complex<double> value[3];
        KOKKOS_INLINE_FUNCTION
            void operator+=(achtempStruct const& other) 
            {
                for (int i = 0; i < 3; ++i) 
                    value[i] += other.value[i];
            }
        KOKKOS_INLINE_FUNCTION
            void operator+=(achtempStruct const volatile& other) volatile 
            {
                for (int i = 0; i < 3; ++i) 
                    value[i] += other.value[i];
            }
        };
//**********************************************************************************************************************************

    auto start_chrono = std::chrono::high_resolution_clock::now();

    Kokkos::parallel_for(team_policy(number_bands, Kokkos::AUTO, 32) , KOKKOS_LAMBDA (const member_type& teamMember) 
    {
        const int n1 = teamMember.league_rank();
        Kokkos::complex<double>  achtemp[3];
        achtempStruct achtempVar = {{achtemp[0],achtemp[1],achtemp[2]}}; 

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, ngpown), [&] (const int my_igp, achtempStruct& achtempVarUpdate)
        {
            for(int iw=nstart; iw<nend; ++iw)
                achtempVarUpdate.value[iw] += expr;

        },achtempVar); // for - ngpown 

    }); // for - number_bands

        auto end_chrono = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed_chrono = end_chrono - start_chrono;

        cout << "********** Chrono Time Taken **********= " << elapsed_chrono.count() << " secs" << endl;

    }
    Kokkos::finalize();

    return 0;
}
