#include <cuda.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

__global__ void gpuParallelMatrixMultiplication(int *a, int *b, int *c, int rows, int cols, int width)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < rows; i += stride)
    {
        for (int j = 0; j < cols; ++j)
        {
            int sum = 0;
            for (int k = 0; k < width; ++k)
            {
                sum += a[i * width + k] * b[k * cols + j];
            }
            c[i * cols + j] = sum;
        }
    }
}

void cpuMatrixMultiplication(int *a, int *b, int *c, int rows, int cols, int width)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            int sum = 0;
            for (int k = 0; k < width; ++k)
            {
                sum += a[i * width + k] * b[k * cols + j];
            }
            c[i * cols + j] = sum;
        }
    }
}

// CUDA számítási kapacitás verziójának Cuda Core értékké alakítása
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
int ConvertSMVer2Cores(int major, int minor) {
    // SM 2.x eszközök
    if (major == 2) {
        switch (minor) {
            case 0: return 32; // Tesla
            case 1: return 48; // Fermi
            default: return 0;
        }
    }
    // SM 3.x eszközök
    else if (major == 3) {
        switch (minor) {
            case 0: return 192; // Kepler
            case 2: return 192; // Kepler
            case 5: return 128; // Kepler
            default: return 0;
        }
    }
    // SM 5.x eszközök
    else if (major == 5) {
        switch (minor) {
            case 0: return 128; // Maxwell
            case 2: return 128; // Maxwell
            case 3: return 192; // Maxwell
            case 5: return 128; // Maxwell
            case 6: return 128; // Maxwell
            default: return 0;
        }
    }
    // SM 6.x eszközök
    else if (major == 6) {
        switch (minor) {
            case 0: return 64;  // Pascal
            case 1: return 128; // Pascal
            case 2: return 128; // Pascal
            case 5: return 64;  // Pascal
            default: return 0;
        }
    }
    // SM 7.x eszközök
    else if (major == 7) {
        switch (minor) {
            case 0: return 64;  // Volta
            case 2: return 64;  // Volta
            case 5: return 64;  // Turing
            case 7: return 64;  // Turing
            default: return 0;
        }
    }
    // Alapeset
    else {
        return 0;
    }
}

int main(int argc, char **argv)
{
    // Beolvasás matrix_A.txt-ből
    FILE *fileA = fopen("matrix_A.txt", "r");
    int rowsA, colsA;
    fscanf(fileA, "%d %d", &rowsA, &colsA);

    int *h_matrixA = (int *)malloc(rowsA * colsA * sizeof(int));
    for (int i = 0; i < rowsA * colsA; ++i)
    {
        fscanf(fileA, "%d", &h_matrixA[i]);
    }
    fclose(fileA);

    // Beolvasás matrix_B.txt-ből
    FILE *fileB = fopen("matrix_B.txt", "r");
    int rowsB, colsB;
    fscanf(fileB, "%d %d", &rowsB, &colsB);

    int *h_matrixB = (int *)malloc(rowsB * colsB * sizeof(int));
    for (int i = 0; i < rowsB * colsB; ++i)
    {
        fscanf(fileB, "%d", &h_matrixB[i]);
    }
    fclose(fileB);

    // A kimeneti GPU mátrix inicializálása
    int *h_result = (int *)malloc(rowsA * colsB * sizeof(int));

    // GPU memórialefoglalás
    int *d_matrixA, *d_matrixB, *d_result;
    cudaMalloc((void **)&d_matrixA, rowsA * colsA * sizeof(int));
    cudaMalloc((void **)&d_matrixB, rowsB * colsB * sizeof(int));
    cudaMalloc((void **)&d_result, rowsA * colsB * sizeof(int));

    // GPU-ra másolás Host-ról
    cudaMemcpy(d_matrixA, h_matrixA, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice);

    // Lekérdezzük a GPU CUDA magjainak számát
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int THREADS_PER_BLOCK = prop.multiProcessorCount * ConvertSMVer2Cores(prop.major, prop.minor);
    std::cout << "Number of CUDA cores: " << THREADS_PER_BLOCK << std::endl;

    // Kiszámoljuk a blokkok és szálak számát
    int num_blocks = (rowsA + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Kernel futtatása párhuzamosan GPU-n + IDŐ kiiratása
    auto start = std::chrono::high_resolution_clock::now();
    gpuParallelMatrixMultiplication<<<num_blocks, THREADS_PER_BLOCK>>>(d_matrixA, d_matrixB, d_result, rowsA, colsB, colsA);
    cudaDeviceSynchronize();  // Szekvenciális lefutás miatt kell.
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    printf("Parallel GPU kernel time: %lf sec\n", elapsed.count());

    // Eredmény visszamásolása a Host-ra
    cudaMemcpy(h_result, d_result, rowsA * colsB * sizeof(int), cudaMemcpyDeviceToHost);

    // Eredmény kiírása a result.txt-be
    FILE *fileResult = fopen("result.txt", "w");
    fprintf(fileResult, "%d %d\n", rowsA, colsB);
    for (int i = 0; i < rowsA; ++i)
    {
        for (int j = 0; j < colsB; ++j)
        {
            fprintf(fileResult, "%d ", h_result[i * colsB + j]);
        }
        fprintf(fileResult, "\n");
    }
    fclose(fileResult);

    // Inicializálás CPU-n
    int *h_result_cpu = (int *)malloc(rowsA * colsB * sizeof(int));

    // Kernel futtatása CPU-n, párhuzamosítás nélkül + IDŐ kiiratása
    auto start2 = std::chrono::high_resolution_clock::now();
    cpuMatrixMultiplication(h_matrixA, h_matrixB, h_result_cpu, rowsA, colsB, colsA);
    auto finish2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = finish2 - start2;
    printf("CPU kernel time: %lf sec\n", elapsed2.count());
    free(h_result_cpu);

    // RAM felszabadítása a Host-on és GPU-n (VRAM-on) egyaránt.
    free(h_matrixA);
    free(h_matrixB);
    free(h_result);
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_result);

    printf("Acceleration: %lf \n", (double)(elapsed2/elapsed));

    return 0;
}
