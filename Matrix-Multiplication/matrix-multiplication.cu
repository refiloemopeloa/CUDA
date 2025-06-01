#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>
#include "assert.h"

using namespace std;

void init_matrices(int *a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = rand() % 100;
        }
    }
}

__global__ void matrix_multiplication(int *a, int *b, int *c, int n) {
    
    // if (row < n && col < n && row * n < n)
    //     c[row * n + col] = a[row * n + col] + b[row * n + col];
    
    // Compute each thread's row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute each thread's column
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int temp_sum = 0;

    // Boundary protection
    if ((row < n) && (col < n)) {
        // Iterate over row, and down column
        for (int k = 0; k < n; k++) {
            // Accumulate result for a single element
            temp_sum += a[row * n + k] * b[k * n + col]; 
        }

        // Assign result
        c[row * n + col] = temp_sum;
    }
}


void verify_result(int *a, int *b, int *c, int n) {
    int *verify_c;
    verify_c = (int*)malloc(n * n * sizeof(int));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                verify_c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            assert(c[i * n + j] == verify_c[i * n + j]);
        }
    }
}

int main() {

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    // matrix of 1024 x 1024
    int n = 1 << 10;

    size_t bytes = n * n * sizeof(int);

    h_a = (int *)malloc(bytes);
    h_b = (int *)malloc(bytes);
    h_c = (int *)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    init_matrices(h_a, n);
    init_matrices(h_b, n);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threads per block
    int BLOCK_SIZE = 16; //16 * 16 = 256

    // Blocks in each dimension
    int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    matrix_multiplication <<< grid, threads >>> (d_a, d_b, d_c, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    verify_result(h_a, h_b, h_c, n);

    cout << "Completed Successfully";

    return 0;

}