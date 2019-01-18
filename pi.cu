#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>

#define N 100000

__global__ void kernel(int* count, float* random_vals)
{
    int i;
    double x,y,z;
    
    // find the overall ID of the thread
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    i = tid;
    int xidx = 0, yidx = 0;
 
    xidx = i + i;
    yidx = xidx + 1;
 
    // get the random x,y points
    x = random_vals[xidx];
    y = random_vals[yidx];
    z = ((x*x)+(y*y));
 
    if (z<=1)
        count[tid] = 1;
    else
        count[tid] = 0;
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
    float *random_vals;
    double pi;
    
    // Allocate in the unified memory the array for the random numbers
    cudaMallocManaged(&random_vals, 2 * N * sizeof(float));
        
    // Use CuRand to generate an array of random numbers on the device
    int status;
    curandGenerator_t gen;
    status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
    status |= curandSetPseudoRandomGeneratorSeed(gen, 4294967296ULL^time(NULL));
    status |= curandGenerateUniform(gen, random_vals, 2 * N);
    status |= curandDestroyGenerator(gen);
        
    // Check to see if there was any problem launching the CURAND kernels and generating
    // the random numbers on the device
    if (status != CURAND_STATUS_SUCCESS)
    {
        printf("CuRand Failure\n");
        exit(EXIT_FAILURE);
    }
 
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
        
    int *count;
    cudaMallocManaged(&count, num_blocks * threads_per_block * sizeof(int));
    CUDAErrorCheck();
        
    kernel <<<num_blocks, threads_per_block>>> (count, random_vals);
        
    cudaDeviceSynchronize();
    CUDAErrorCheck();
    
    unsigned int reduced_count = 0;
    for(int i = 0; i < N; i++)
        reduced_count += count[i];
 
    cudaFree(random_vals);
    cudaFree(count);
 
    // find the ratio
    pi = ((double)reduced_count / N) * 4.0;
    printf("PI = %g\n", pi);
 
    return 0;
}