#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#include "rays/ray_gpu.cu"

using namespace std;
using namespace std::chrono;

#define PRINT
// #define VERIFY

__device__ vec3 color(const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = (unit_direction.y() + 1.0f)*0.5f;
    return vec3(1.0, 1.0, 1.0)*(1.0f-t) + vec3(0.5, 0.7, 1.0)*t;
}

__global__ void render(vec3 *pixels, int x, int y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ((i < x) && (j < y)) {
        int pixel_index = j * x + i;
        float u = float(i) / float(x);
        float v = float(j) / float(y);
        ray r(origin, lower_left_corner + horizontal*u + vertical*v);
        pixels[pixel_index] = color(r);
    }
}

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


// __global__ void render_gpu(int x, int y, int* pixels) {

//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     int j = blockDim.y * blockIdx.y + threadIdx.y;

//     if ((i < x) && (j < y)) {
//         int pixel_index = j * x * 3 + i * 3;
//         float r = float(i) / float(x);
//         float g = float(j) / float(y);
//         float b = 0.2;
//         vec3 col(r, g, b);
//         pixels[pixel_index + 0] = int(255.99 * col[0]);
//         pixels[pixel_index + 1] = int(255.99 * col[1]);
//         pixels[pixel_index + 2] = int(255.99 * col[2]);
//     }
// }

void print_image(int x, int y, vec3* pixels, ostream& file) {
    file << "P3\n" << x << " " << y << "\n255\n";

    for (int j = y-1; j >= 0; j--) {
        for (int i = 0;  i < x; i++) {
            int pixel_index = j * x  + i;
            file << int(255.99 * pixels[pixel_index].r()) << " " << int(255.99 * pixels[pixel_index].g()) << " " << int(255.99 * pixels[pixel_index].b()) << "\n";
        }
    }
}

bool verify_result(vec3 *pixels_gpu, int *pixels_cpu, int x, int y) {
    int pixel_index, pixel_gpu, cpu_1, cpu_2, cpu_3;
    vec3 gpu;
    for (int j = y-1; j >= 0; j--) {
        for (int i = 0;  i < x; i++) {
            pixel_index = j * x * 3 + i * 3;
            pixel_gpu = j*x + i;
            gpu = pixels_gpu[pixel_gpu];
            cpu_1 = pixels_cpu[pixel_index + 0];
            cpu_2 = pixels_cpu[pixel_index + 1];
            cpu_3 = pixels_cpu[pixel_index + 2];
            if (cpu_1 != gpu.r() || cpu_2 != gpu.g() || cpu_3 != gpu.b()) {
                return false;
            }
        }
    }
    return true;
}

int main() {

    int x = 200;
    int y = x/2;
    int n = x * y * 3;
    long bytes = n * sizeof(int);
    long vec3_bytes = x * y * sizeof(vec3);

    vec3 *h_pixels = (vec3*) malloc(bytes);

    vec3 *d_pixels;

    
    cudaMalloc((void **)&d_pixels, vec3_bytes);    

    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);
    
    
    dim3 threads_per_block(16,16);
    dim3 blocks_per_grid(x / threads_per_block.x+1, y/ threads_per_block.y+1);
    
    auto start = high_resolution_clock::now();
    render<<<blocks_per_grid, threads_per_block>>>(d_pixels, x, y, lower_left_corner, horizontal, vertical, origin);
    
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
    cout << "Time taken by gpu: " << duration.count()*10e-9 << " seconds" << endl;
    
    float speedup = float(duration.count());
    
    cudaMemcpy(h_pixels, d_pixels, vec3_bytes, cudaMemcpyDeviceToHost);
    
    int *verify_pixels = (int*) malloc(bytes);
    start = high_resolution_clock::now();
    render_cpu(x, y, verify_pixels);
    stop = high_resolution_clock::now();
    
#ifdef VERIFY
    if (verify_result(h_pixels, verify_pixels, x, y)==false)
    {
        cerr << "Verification failed!\n";
        return 0;
    }
#endif
    
    duration = duration_cast<nanoseconds>(stop - start);
    cout << "Time taken by cpu: " << duration.count()*10e-9 << " seconds" << endl;
    
    speedup = 1/speedup;
    speedup *=  float(duration.count());
    
    cout << "Speedup: " << speedup << endl;
    
#ifdef PRINT 
    string filename = "out-ray.ppm";
    ofstream file(filename);
    print_image(x, y, h_pixels, file);

    file.close();
#endif

    cudaFree(d_pixels);
    free(h_pixels);
    free(verify_pixels);
    
    return 0;
}