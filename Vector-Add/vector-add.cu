#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// CUDA Kernel for vector addition
// This function is executed by each and every thread, individually
__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    // Calculate global thread ID (tid)
              //which block we are in   //which thread we are in (offset in the block)
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
                          //block size
    // Vector boundary guard
    // Ensures that we dont go passed the number of elements in the vector, since it is possible that there are more threads than elements in the vector
    if (tid < n) {
        // Each thread adds a single element
        c[tid] = a[tid] + b[tid];
    }
}

// Initialize vector of size n to int between 0 and 99
void matrix_init(int* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 100;    }
}

// Check vector add result
void error_check(int *a, int* b, int *c, int n) {
    for (int i=0; i < n; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    // 2 ^ 16
    int n = 1 << 16;

    // Host vector pointers
    int *h_a, *h_b, *h_c;

    // Device vector pointers
    int *d_a, *d_b, *d_c;

    // Allocation size for all vectors
    size_t bytes = sizeof(int) * n;

    // Allocate host memory
    h_a = (int *)malloc(bytes);
    h_b = (int *)malloc(bytes);
    h_c = (int *)malloc(bytes);

    // Allocate space on device memory for vectors
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize vectors a and b with random values between 0 and 99
    matrix_init(h_a, n);
    matrix_init(h_b, n);

    // Copy data to from host memory to device memory
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threadblock size
    int NUM_THREADS = 256;

    // Grid size - each grid will compute on n/NUM_THREADS of the vector
    int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

    // Launch kernel on default stream w/o shared memory
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a,d_b,d_c,n);

    // Copy sum vector from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result for errors
    error_check(h_a, h_b, h_c, n);


    

    return 0;
}