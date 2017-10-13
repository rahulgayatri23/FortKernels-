#include <iostream>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
using namespace std;

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
    int number_bands = 32;
    int ncouls = 32;
    int ngpown = 32;
    typedef Kokkos::Cuda     ExecSpace;
    typedef Kokkos::CudaUVMSpace  MemSpace;
    typedef Kokkos::LayoutLeft   Layout;
    typedef Kokkos::RangePolicy<ExecSpace>  range_policy;
    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;

        double timePass1 = 0.00;
        Kokkos::parallel_reduce(team_policy(number_bands, Kokkos::AUTO, 32) , KOKKOS_LAMBDA (const member_type& teamMember, double& timePassUpdate1) 
        {
            const int n1 = teamMember.league_rank();
            double timePass2 = 0.00;

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, ngpown), [&] (const int my_igp, double& timePassUpdate2)
            {
                double timePass3 = 0.00;
                Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( teamMember, ncouls ), [&] (const int ig, double &timePassUpdate3 ) 
                {
                    timePassUpdate3 += 0.5;

                }, timePass3);

                timePassUpdate2 += timePass3;

            },timePass2); // for - ngpown 

            timePassUpdate1 += timePass2;

        },timePass1); // for - number_bands
    }
    Kokkos::finalize();

    return 0;
}
