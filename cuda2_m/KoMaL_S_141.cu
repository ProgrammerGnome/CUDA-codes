#include <iostream>
#include <stdio.h>
#include <chrono>

#define THREADS_PER_BLOCK 384

// Atomic add for double (available from CUDA 11.2)
__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void normal_kernel(uint64_t M, uint64_t N, uint64_t P, double* result) {
    double Nd = (double)N;
    double sum = 0;

    for (uint64_t i = 0; i < M; ++i)
    {
        sum = sum + Nd / (P + i);
    }

    *result = sum;
}

__global__ void parallel_kernel(uint64_t M, uint64_t N, uint64_t P, double* result) {
    double Nd = (double)N;
    double local_sum = 0;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
        local_sum += Nd / (P + i);
    }

    // Szálak közötti eredmények összegzése atomikus művelettel
    __shared__ double shared_sum;
    if (threadIdx.x == 0) {
        shared_sum = 0;
    }
    __syncthreads();

    atomicAdd_double(&shared_sum, local_sum);

    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd_double(result, shared_sum);
    }
}

int main()
{
    uint64_t M, N, P;

    printf("N: "); scanf("%lu", &N);
    printf("M: "); scanf("%lu", &M);
    printf("P: "); scanf("%lu", &P);

    printf("\t\tSum\t\tTime\n");

    // Normal version
    {
        auto start = std::chrono::high_resolution_clock::now();

        double* dev_result;
        cudaMalloc((void**)&dev_result, sizeof(double));

        normal_kernel<<<1, 1>>>(M, N, P, dev_result);

        double result;
        cudaMemcpy(&result, dev_result, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(dev_result);

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        printf("Normal\t\t%f\t%10.6f\n", result, elapsed.count());
    }

    // Parallel version
    {
        auto start = std::chrono::high_resolution_clock::now();

        double* dev_result;
        cudaMalloc((void**)&dev_result, sizeof(double));
        cudaMemset(dev_result, 0, sizeof(double));

        int num_blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        parallel_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(M, N, P, dev_result);

        double result;
        cudaMemcpy(&result, dev_result, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(dev_result);

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        printf("Parallel\t%f\t%10.6f\n", result, elapsed.count());
    }

    return 0;
}
