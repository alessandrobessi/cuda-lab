/*
nvcc add.cu -o add_cuda
nvprof ./add_cuda
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* 
GPU kernel
    threadIdx.x   index of the current thread within its block
    blockDim.x    number of threads in the block
    gridDim.x     number of thread blocks
    blockIdx.x    index of current block within the grid
*/
__global__ void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+= stride)
    {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20; // 1M elements

    // allocate unified memory -- accessible from CPU or GPU
    float *x, *y;
    cudaMallocManaged(&x, sizeof(float) * N);
    cudaMallocManaged(&y, sizeof(float) * N);

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // run kernel on 1M elements on the CPU
    // CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    add<<<num_blocks, threads_per_block>>>(N, x, y);

    // wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    printf("Max error: %f\n", maxError);

    // free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}