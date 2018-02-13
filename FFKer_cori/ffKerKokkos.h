#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
using namespace std;


#define CUDASPACE 0
#define OPENMPSPACE 1
#define CUDAUVM 0
#define SERIAL 0
#define THREADS 0

#if OPENMPSPACE
#include "GPUComplex.h"
        typedef Kokkos::OpenMP   ExecSpace;
        typedef Kokkos::OpenMP        MemSpace;
        typedef Kokkos::LayoutRight  Layout;
#endif

#if CUDASPACE
#include "GPUComplex.h"
        typedef Kokkos::Cuda     ExecSpace;
        typedef Kokkos::CudaSpace     MemSpace;
        typedef Kokkos::LayoutLeft   Layout;
#endif

#if CUDAUVM
#include "GPUComplex.h"
        typedef Kokkos::Cuda     ExecSpace;
        typedef Kokkos::CudaUVMSpace  MemSpace;
        typedef Kokkos::LayoutLeft   Layout;
#endif

#if SERIAL
#include "Complex.h"
        typedef Kokkos::Serial   ExecSpace;
        typedef Kokkos::HostSpace     MemSpace;
#endif

#if THREADS
#include "Complex.h"
        typedef Kokkos::Threads  ExecSpace;
        typedef Kokkos::HostSpace     MemSpace;
#endif

typedef Kokkos::RangePolicy<ExecSpace>  range_policy;
typedef Kokkos::TeamPolicy<> team_policy;
typedef Kokkos::TeamPolicy<>::member_type member_type;

typedef Kokkos::View<GPUComplex, Layout, MemSpace> ViewScalarTypeComplex;
typedef Kokkos::View<GPUComplex*, Layout, MemSpace> ViewVectorTypeComplex;
typedef Kokkos::View<GPUComplex**, Layout, MemSpace> ViewMatrixTypeComplex;

typedef Kokkos::View<int*, Layout, MemSpace>   ViewVectorTypeInt;
typedef Kokkos::View<double*, Layout, MemSpace>   ViewVectorTypeDouble;

typedef Kokkos::View<int, Layout, MemSpace>   ViewScalarTypeInt;
typedef Kokkos::View<double, Layout, MemSpace>   ViewScalarTypeDouble;

struct GPUComplStruct 
{
    double re;
    double im;
KOKKOS_INLINE_FUNCTION
    void operator+=(GPUComplStruct const& other) 
    {
        re += other.re;
        im += other.im;
    }
KOKKOS_INLINE_FUNCTION
    void operator+=(GPUComplStruct const volatile& other) volatile 
    {
        re += other.re;
        im += other.im;
    }
};

