#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

void render_cpu(int x, int y, int* pixels) {
    float r;
    float g;
    float b;
    int pixel_index;

    for (int j = y-1; j >= 0; j--) {
        for (int i = 0;  i < x; i++) {
            pixel_index = j * x * 3 + i * 3;
            r = float(i) / float(x);
            g = float(j) / float(y);
            b = 0.2;
            pixels[pixel_index + 0] = int(255.99 * r);
            pixels[pixel_index + 1] = int(255.99 * g);
            pixels[pixel_index + 2] = int(255.99 * b);
        }
    }
}

__global__ void render_gpu(int x, int y, int* pixels) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ((i < x) && (j < y)) {
        int pixel_index = j * x * 3 + i * 3;
        float r = float(i) / float(x);
        float g = float(j) / float(y);
        float b = 0.2;
        pixels[pixel_index + 0] = int(255.99 * r);
        pixels[pixel_index + 1] = int(255.99 * g);
        pixels[pixel_index + 2] = int(255.99 * b);
    }
}

void print_image(int x, int y, int* pixels, ostream& file) {
    file << "P3\n" << x << " " << y << "\n255\n";

    for (int j = y-1; j >= 0; j--) {
        for (int i = 0;  i < x; i++) {
            int pixel_index = j * x * 3 + i * 3;
            file << pixels[pixel_index + 0] << " " << pixels[pixel_index + 1] << " " << pixels[pixel_index + 2] << "\n";
        }
    }
}

bool verify_result(int *pixels_gpu, int *pixels_cpu, int x, int y) {
    int pixel_index, cpu_1, cpu_2, cpu_3, gpu_1, gpu_2, gpu_3;
    for (int j = y-1; j >= 0; j--) {
        for (int i = 0;  i < x; i++) {
            pixel_index = j * x * 3 + i * 3;
            cpu_1 = pixels_cpu[pixel_index + 0];
            cpu_2 = pixels_cpu[pixel_index + 1];
            cpu_3 = pixels_cpu[pixel_index + 2];
            gpu_1 = pixels_gpu[pixel_index + 0];
            gpu_2 = pixels_gpu[pixel_index + 1];
            gpu_3 = pixels_gpu[pixel_index + 2];
            if (cpu_1 != gpu_1 || cpu_2 != gpu_2 || cpu_3 != gpu_3) {
                return false;
            }
        }
    }
    return true;
}

int main() {

    int x = 20000;
    int y = x/2;
    int n = x * y * 3;
    long bytes = n * sizeof(int);

    int *h_pixels = (int*) malloc(bytes);

    int *d_pixels;

    
    cudaMalloc(&d_pixels, bytes);    
    
    // string filename = "out-cpu-array.ppm";
    
    dim3 threads_per_block(16,16);
    dim3 blocks_per_grid(x * 3 / threads_per_block.x, y * 3 / threads_per_block.y);
    
    auto start = high_resolution_clock::now();
    render_gpu<<<blocks_per_grid, threads_per_block>>>(x, y, d_pixels);
    
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
    cout << "Time taken by gpu: " << duration.count()*10e-9 << " seconds" << endl;
    
    float speedup = float(duration.count());
    
    cudaMemcpy(h_pixels, d_pixels, bytes, cudaMemcpyDeviceToHost);
    
    int *verify_pixels = (int*) malloc(bytes);
    start = high_resolution_clock::now();
    render_cpu(x, y, verify_pixels);
    stop = high_resolution_clock::now();
    
    if (verify_result(h_pixels, verify_pixels, x, y)==false)
    {
        cerr << "Verification failed!\n";
        return 0;
    }

    duration = duration_cast<nanoseconds>(stop - start);
    cout << "Time taken by cpu: " << duration.count()*10e-9 << " seconds" << endl;



    speedup = 1/speedup;
    speedup *=  float(duration.count());

    cout << "Speedup: " << speedup << endl;

    // ofstream file(filename);
    // print_image(x, y, h_pixels, file);

    // file.close();

    cudaFree(d_pixels);
    free(h_pixels);
    free(verify_pixels);
    
    return 0;
}