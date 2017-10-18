#include <iostream>
#include <cstdlib>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
using namespace std;


#define CUDASPACE 0
#define OPENMPSPACE 0
#define CUDAUVM 1
#define SERIAL 0
#define THREADS 0

#if OPENMPSPACE
        typedef Kokkos::OpenMP   ExecSpace;
        typedef Kokkos::OpenMP        MemSpace;
        typedef Kokkos::LayoutRight  Layout;
#endif

#if CUDASPACE
        typedef Kokkos::Cuda     ExecSpace;
        typedef Kokkos::CudaSpace     MemSpace;
        typedef Kokkos::LayoutLeft   Layout;
#endif

#if SERIAL
        typedef Kokkos::Serial   ExecSpace;
        typedef Kokkos::HostSpace     MemSpace;
#endif

#if THREADS
        typedef Kokkos::Threads  ExecSpace;
        typedef Kokkos::HostSpace     MemSpace;
#endif

#if CUDAUVM
        typedef Kokkos::Cuda     ExecSpace;
        typedef Kokkos::CudaUVMSpace  MemSpace;
        typedef Kokkos::LayoutLeft   Layout;
#endif

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

//KOKKOS_INLINE_FUNCTION
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

//KOKKOS_INLINE_FUNCTION
//void flagOCC_solver(ViewScalarTypeComplex mygpvar1, double wxt, ViewMatrixTypeComplex wtilde_array, int my_igp, int n1, ViewMatrixTypeComplex aqsmtemp, ViewMatrixTypeComplex aqsntemp, ViewMatrixTypeComplex I_eps_array, Kokkos::complex<double> &ssxt, ViewScalarTypeComplex scht, int ncouls, int igp);
//
//
//KOKKOS_INLINE_FUNCTION
//void reduce_achstemp(ViewScalarTypeComplex mygpvar1,ViewScalarTypeComplex schstemp,  int n1, ViewVectorTypeInt inv_igp_index, int ncouls, ViewMatrixTypeComplex aqsmtemp, ViewMatrixTypeComplex aqsntemp, ViewMatrixTypeComplex I_eps_array, Kokkos::complex<double>& achstemp, ViewVectorTypeInt indinv, int ngpown, Kokkos::View<double*> vcoul);
//
//KOKKOS_INLINE_FUNCTION
//Kokkos::complex<double> doubleMultKokkosComplex(double op1, Kokkos::complex<double> op2);
//
//KOKKOS_INLINE_FUNCTION
//Kokkos::complex<double> doublePlusKokkosComplex(double op1, Kokkos::complex<double> op2);
//
//KOKKOS_INLINE_FUNCTION
//Kokkos::complex<double> doubleMinusKokkosComplex(double op1, Kokkos::complex<double> op2);
//
//KOKKOS_INLINE_FUNCTION
//Kokkos::complex<double> kokkos_square(Kokkos::complex<double> compl_num, int n);

Kokkos::complex<double>  achtemp[3];
achtempStruct achtempVar = {{achtemp[0],achtemp[1],achtemp[2]}}; 

Kokkos::complex<double> achstemp(0.00, 0.00);
