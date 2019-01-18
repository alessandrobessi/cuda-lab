/*
!nvcc pi.cu -o pi -Wno-deprecated-gpu-targets -lcurand
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#define NUM_BLOCKS 1024
#define THREADS_PER_BLOCK 256
#define ITER_PER_THREAD 2048

#define PI 3.14159265359

__global__ void kernel(int *count)
{
    double x, y, z;

    // find the overall ID of the thread
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    count[index] = 0;
    curandState state;
    curand_init((unsigned long long)clock() + index, 0, 0, &state);
    for (int i = 0; i < ITER_PER_THREAD; i++)
    {
        x = curand_uniform_double(&state);
        y = curand_uniform_double(&state);
        z =  x * x + y * y;
 
        if (z <= 1)
            count[index] += 1;
    }
}

void CUDAErrorCheck()
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error : %s (%d)\n", cudaGetErrorString(error), error);
        exit(EXIT_FAILURE);
    }
}

int main()
{   
    long unsigned int n = NUM_BLOCKS * THREADS_PER_BLOCK;
    int *count;
    cudaMallocManaged(&count, n * sizeof(int));
    CUDAErrorCheck();
        
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(count);
        
    cudaDeviceSynchronize();
    CUDAErrorCheck();
    
    long unsigned int reduced_count = 0;
    for(int i = 0; i < n; i++)
        reduced_count += count[i];
 
    cudaFree(count);
 
    // find the ratio
    long unsigned int total_iter = n * ITER_PER_THREAD;
    double pi = ((double)reduced_count / total_iter) * 4.0;
    printf("PI [%lu iterations] = %.10g\n", total_iter, pi);
    printf("Error = %.10g\n", pi - PI);

    return 0;
}