#include <stdio.h>
#include <time.h>
#include <math.h>

//Perform Matrix Multiplication Utilizing the CPU Only
double cpuMult(int N, double *output) {
    // Number of bytes to allocate for N doubles
    size_t bytes = N * N * sizeof(double);
    // Allocate memory for arrays A, B, and C
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);
    // Fill input arrays A and B
    for(int i = 0; i < N; i ++) {
        for(int j = 0; j < N; j ++) {
            A[i* N + j] = sin(i);
            B[i * N + j] = cos(j);
        }
    }
    //Start Clock
    clock_t t;
    t = clock();
    //Perform For Loop Multiplication of Matrices
    double sum;
    for (int i = 0; i < N; i++) {
        for(int j = 0; j < N; j ++) {
            sum = 0.f;
            for(int k = 0; k < N; k ++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    //End Clock
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
__global__ void matrix_multiplication(double *A, double *B, double *C, int N) {
    int r = blockIdx.y*blockDim.y+threadIdx.y;
    int c= blockIdx.x*blockDim.x+threadIdx.x;

    double sum = 0;
    if ((r < N) && (c < N)) {
        for (int i = 0; i < N; i++) {
            sum += A[r * N + i] * B[i * N + c];
        }
        C[r * N + c] = sum;
    }
}

//Perform Matrix Multiplication Utilizing the GPU Only
double gpuMult(int N, double *output) {
    // Number of bytes to allocate for N doubles
    size_t bytes = N * N * sizeof(double);
    // Allocate memory for arrays A, B, and C
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);
    // Fill input arrays A and B
    for(int i = 0; i < N; i ++) {
        for(int j = 0; j < N; j ++) {
            A[i* N + j] = sin(i);
            B[i * N + j] = cos(j);
        }
    }

    //Start Clock
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
    dim3 thr_per_blk(N, N, 1);
    dim3 blk_in_grid(1, 1, 1);
    if (N*N > 1024){
        thr_per_blk.x = 32;
        thr_per_blk.y = 32;
        blk_in_grid.x = ceil(double(N + 31)/double(thr_per_blk.x));
        blk_in_grid.y = ceil(double(N + 31)/double(thr_per_blk.y));
    }
    // Launch kernel
    matrix_multiplication<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, N);
    //Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    //End Clock
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;

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

//Main program
int main(){
    int power = 20; //number of different N's to try
    int vectorLengths[power];
    //Fill vectorLengths with vector lengths we would like to test (Values are (1.5^x) where x is in the range of [0:power])
    for(int i = 0; i < power; i ++){
        vectorLengths[i] = (int)(pow(1.5,i) + 0.5);
    }
    int N;
    for(int x = 0; x < power; x ++){
        N = vectorLengths[x];
        printf("Running Comparison for NxN Matrix with N = %d \n", N);

        // Number of bytes to allocate for N doubles
        size_t bytes = N * N * sizeof(double);
        // Allocate memory for output arrays
        double *CPU = (double*)malloc(bytes);
        double *GPU = (double*)malloc(bytes);

        //Execute CPU Multiplication
        double cpuTime = cpuMult(N, CPU);
        printf("CPU Vector Multiplication took %f seconds to execute \n", cpuTime);
        //Execute GPU Multiplication
        double gpuTime = gpuMult(N, GPU);
        printf("GPU Vector Multiplication took %f seconds to execute \n", gpuTime);

        double err = 0;
        //Compare the results of both executions; sum up total of discrepancies
        for (int ROW=0; ROW < N; ROW++){
            for (int COL=0; COL < N; COL++){
                err += CPU[ROW * N + COL] - GPU[ROW * N + COL];
            }
        }
        printf("Difference Between Results Matrices: %f \n", abs(err));
        printf("\n");

        free(CPU);
        free(GPU);
    }
    return 0;
}
