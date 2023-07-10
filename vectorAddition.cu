#include <stdio.h>
#include <time.h>
#include <math.h>

//Perform Vector Addition Utilizing the CPU Only
double cpuAdd(int N, double *output){ 
    // Number of bytes to allocate for N doubles
    size_t bytes = N*sizeof(double);
    // Allocate memory for arrays A, B, and C on host
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);
    // Fill host arrays A and B
    for(int i = 0; i < N; i ++) {
        A[i] = sin(i);
        B[i] = cos(i);
    }

    //Start Timer
    clock_t t;
    t = clock();

    //Perform For Loop Addition of Vectors
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    //End Timer
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;

    //Copy C to Output Array
    memcpy(output, C, bytes);
    //Free Memory
    free(A);
    free(B);
    free(C);
    //Return the amount of time taken
    return time_taken;
}

// Kernel
__global__ void add_vectors(double *a, double *b, double *c, int N) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N){
       c[index] = a[index] + b[index];
    }
}

//Perform Vector Addition Utilizing the GPU Only
double gpuAdd(int N, double *output){
    // Number of bytes to allocate for N doubles
    size_t bytes = N*sizeof(double);
    // Allocate memory for arrays A, B, and C on host
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);
    // Fill host arrays A and B
    for(int i = 0; i < N; i ++) {
        A[i] = sin(i);
        B[i] = cos(i);
    }

    //Start Timer
    clock_t t;
    t = clock();

    // Allocate memory for arrays d_A, d_B, and d_C on device
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    int thr_per_blk = 1024;
    int blk_in_grid = ceil(double(N)/double(thr_per_blk));

    // Launch kernel
    add_vectors<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, N);

    // Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    //End Timer
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //Copy C to Output Array
    memcpy(output, C, bytes);
    //Free Memory
    free(A);
    free(B);
    free(C);
    //Return the amount of time taken
    return time_taken;
}

// Main program
int main() {
    int power = 50;
    int vectorLengths[power];
    //Fill vectorLengths with vector lengths we would like to test
    for(int i = 0; i < power; i ++) {
        vectorLengths[i] = (int)(pow(1.5,i) + 0.5);
    }
    int N;
    for(int x = 0; x < power; x ++) {
        N = vectorLengths[x];
        printf("Running Comparison for Vector Size of %d \n", N);
        // Number of bytes to allocate for N doubles
        size_t bytes = N*sizeof(double);

        // Allocate memory for output arrays
        double *CPU = (double*)malloc(bytes);
        double *GPU = (double*)malloc(bytes);

        //Execute CPU Addition
        double cpuTime = cpuAdd(N, CPU);
        printf("CPU Vector Addition took %f seconds to execute \n", cpuTime);

        //Execute GPU Addition
        double gpuTime = gpuAdd(N, GPU);
        printf("GPU Vector Addition took %f seconds to execute \n", gpuTime);


        double err = 0;
        //Compare the results of both executions; sum up total of discrepancies
        for (int ROW=0; ROW < N; ROW++){
            err += CPU[ROW] - GPU[ROW];
        }
        printf("Difference Between Result Vectors: %f \n", abs(err));
        printf("\n");

        //Free Utilized Memory
        free(CPU);
        free(GPU);
    }
    return 0;
}
