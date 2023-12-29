#include <cuda.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256 // Reméljük ennyi CUDA mag elérhető...

__global__ void add(int a, int b, int* maxi)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < *maxi; i += stride)
    {
        printf("a valtozo erteke: %d\n", i);

        // Szinkronizálás a szálak között (szekvencialitás miatt)
        __syncthreads();
    }
}

int main(int argc, char** argv)
{
    // Memóriafoglalás és másolás a GPU-ra
    int* maxi;
    cudaMalloc((void**)&maxi, sizeof(int));
    int max_value = 1000000;
    cudaMemcpy(maxi, &max_value, sizeof(int), cudaMemcpyHostToDevice);

    // Kiszámoljuk a blokkok és szálak számát
    int num_blocks = (max_value + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Kernel futtatása párhuzamosan
    add<<<num_blocks, THREADS_PER_BLOCK>>>(1, 2, maxi);

    // Memóriafelszabadítás GPU-n
    cudaFree(maxi);

    return 0;
}

