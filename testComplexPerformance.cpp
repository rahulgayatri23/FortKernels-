#include "Complex.h" 

using namespace std;

#define N 150000
#define RAND_LOCALMAX 100

void initComplex(std::complex<double>* stdComplexDouble, GPUComplex* GPUComplexDouble)
{

    double rand_re = rand() % RAND_LOCALMAX;
    double rand_im = rand() % RAND_LOCALMAX;

    std::complex<double> stdExpr(rand_re, rand_im);
    GPUComplex GPUExpr(rand_re, rand_im);

    for(int i = 0; i < N+1; i++)
    {
        stdComplexDouble[i] = stdExpr; 
        GPUComplexDouble[i] = GPUExpr;
    }
}

int main(int argc, char** argv)
{
    //OpenMP variables
    int tid, numThreads;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;

    std::complex<double> stdExpr0(0.0, 0.0);
    std::complex<double> stdExpr(0.5, 0.5);
    GPUComplex GPUExpr0(0.0, 0.0);
    GPUComplex GPUExpr(0.5, 0.5);

    std::complex<double> *stdComplexDouble = new std::complex<double>[N+1];
    GPUComplex *GPUComplexDouble = new GPUComplex[N+1];


    initComplex(stdComplexDouble, GPUComplexDouble);

//std::complex<double> computations
    auto start_chrono_stdComplex = std::chrono::high_resolution_clock::now();
    std::complex<double> stdComplexSquare(0.00, 0.00),\
        stdComplexMult1(0.00, 0.00), stdComplexMult2,\
        stdComplexConj(0.00, 0.00), stdComplexProduct(0.00, 0.00),\
        stdComplexFMA(0.00, 0.00), stdComplexFMS(0.00, 0.00);

   GPUComplex GPUComplexSquare(0.00, 0.00), GPUComplexConj(0.00, 0.00),\
                GPUComplexMult1(0.00, 0.00), GPUComplexMult2(0.00, 0.00),\
                GPUComplexProduct(0.00, 0.00),\
                GPUComplexFMA(0.00, 0.00), GPUComplexFMS(0.00, 0.00);

    double stdComplexAbs = 0.00, GPUComplexAbs = 0.00;

    for(int i = 0; i < N; i++)
    {
        stdComplexSquare += std::pow(stdComplexDouble[i], 2);
        stdComplexConj += std::conj(stdComplexDouble[i]);
        stdComplexMult1 += stdComplexDouble[i] * 5.0 * 10.0;
        stdComplexMult2 += stdComplexDouble[i] * 3.75;
        stdComplexAbs += std::abs(stdComplexDouble[i]);
        stdComplexProduct += stdComplexDouble[i] * stdComplexDouble[i+1];
        stdComplexFMA += stdComplexDouble[i] * stdComplexDouble[i+1];
        stdComplexFMS -= stdComplexDouble[i] * stdComplexDouble[i+1];
    }

    std::chrono::duration<double> elapsed_chrono_stdComplex = std::chrono::high_resolution_clock::now() - start_chrono_stdComplex;

//GPUComplex computations
    auto start_chrono_GPUComplex = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < N; i++)
    {
        GPUComplexSquare += GPUComplex_square(GPUComplexDouble[0]);
        GPUComplexConj += GPUComplex_conj(GPUComplexDouble[0]);
        GPUComplexMult1 += GPUComplex_mult(GPUComplexDouble[0], 5, 10);
        GPUComplexMult2 += GPUComplex_mult(GPUComplexDouble[0], 3.75);
        GPUComplexAbs += GPUComplex_abs(GPUComplexDouble[0]);
        GPUComplexProduct += GPUComplex_product(GPUComplexDouble[i] , GPUComplexDouble[i+1]);
        GPUComplex_fma(GPUComplexFMA, GPUComplexDouble[i], GPUComplexDouble[i+1]);
        GPUComplex_fms(GPUComplexFMS, GPUComplexDouble[i], GPUComplexDouble[i+1]);
    }

    std::chrono::duration<double> elapsed_chrono_GPUComplex = std::chrono::high_resolution_clock::now() - start_chrono_GPUComplex;

    printf("Square Results : \n");
    printf("stdComplexSquare = \(%f, %f)\n", std::real(stdComplexSquare), std::imag(stdComplexSquare));
    printf("GPUComplexSquare = "); GPUComplexSquare.print();
    printf("\n");

    printf("Conjugate Results : \n");
    printf("stdComplexConj = \(%f, %f)\n", std::real(stdComplexConj), std::imag(stdComplexConj));
    printf("GPUComplexConj = "); GPUComplexConj.print();
    printf("\n");

    printf("Absolute Results : \n");
    printf("stdComplexAbs = %f\n", stdComplexAbs);
    printf("GPUComplexAbs = %f\n", GPUComplexAbs);
    printf("\n");

    printf("Mult1 Results result = complex * double * double: \n");
    printf("stdComplexMult1 = \(%f, %f)\n", std::real(stdComplexMult1), std::imag(stdComplexMult1));
    printf("GPUComplexMult1 = "); GPUComplexMult1.print();
    printf("\n");

    printf("Mult2 Results result = complex * double  \n");
    printf("stdComplexMult2 = \(%f, %f)\n", std::real(stdComplexMult2), std::imag(stdComplexMult2));
    printf("GPUComplexMult2 = "); GPUComplexMult2.print();
    printf("\n");

    printf("Product Results : \n");
    printf("stdComplexProduct = \(%f, %f)\n", std::real(stdComplexProduct), std::imag(stdComplexProduct));
    printf("GPUComplexProduct = "); GPUComplexProduct.print();
    printf("\n");

    printf("FMA Results : \n");
    printf("stdComplexFMA = \(%f, %f)\n", std::real(stdComplexFMA), std::imag(stdComplexFMA));
    printf("GPUComplexFMA = "); GPUComplexFMA.print();
    printf("\n");

    printf("FMS Results : \n");
    printf("stdComplexFMS = \(%f, %f)\n", std::real(stdComplexFMS), std::imag(stdComplexFMS));
    printf("GPUComplexFMS = "); GPUComplexFMS.print();
    printf("\n");

    cout << "********** stdComplex Chrono Time Taken **********= " << elapsed_chrono_stdComplex.count() << " secs" << endl;
    cout << "********** GPUComplex Chrono Time Taken **********= " << elapsed_chrono_GPUComplex.count() << " secs" << endl;


    free(stdComplexDouble);
    free(GPUComplexDouble);

    return 0;
}
