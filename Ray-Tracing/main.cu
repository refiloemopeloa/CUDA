#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <curand_kernel.h>
#include <omp.h>

#include "include/camera_gpu.cu"

// #include "include/helper_cuda.h"

using namespace std;
using namespace std::chrono;

#define PRINT
// #define VERIFY

// __host__ __device__ float hit_sphere(const vec3& center, float radius, const Ray& r);

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#define RND (curand_uniform(&local_rand_state))

// __device__ vec3 random_in_unit_sphere(curandState *local_rand_state);
// __host__ vec3 random_in_unit_sphere(uniform_real_distribution<float> &local_rand_state, default_random_engine &generator);

__device__ vec3 color(const Ray &r, hitable **world, curandState *local_rand_state)
{
    Ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f); // Initial attenuation set to white

    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            Ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation; // Update attenuation
                cur_ray = scattered;            // Update ray to scattered ray
            }
            else
            {
                return vec3(0, 0, 0); // Return black if no scattering
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f); // Gradient from white to blue
            return cur_attenuation * c;
        }
    }
    return vec3(0, 0, 0); // Return black if max iterations reached
}

__host__ vec3 color(const Ray &r, hitable **world, uniform_real_distribution<float> &local_rand_state, default_random_engine &generator)
{
    Ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f); // Initial attenuation set to white

    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            Ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state, generator))
            {
                cur_attenuation *= attenuation; // Update attenuation
                cur_ray = scattered;            // Update ray to scattered ray
            }
            else
            {
                return vec3(0, 0, 0); // Return black if no scattering
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f); // Gradient from white to blue
            return cur_attenuation * c;
        }
    }
    return vec3(0, 0, 0); // Return black if max iterations reached
}

__device__ void print_element(float element, int index)
{
    printf("[%d]: %f ", index, element);
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_cam, int x, int y, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));

        int i = 1;

        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f)
                {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f)
                {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else
                {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_cam = new camera(lookfrom,
                            lookat,
                            vec3(0, 1, 0),
                            30.0,
                            float(x) / float(y),
                            aperture,
                            dist_to_focus);
    }
}

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int x, int y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= x) || (j >= y))
        return;
    int pixel_index = j * x + i;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *pixels, int x, int y, int s, camera **cam, hitable **world, curandState *rand_state)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ((i >= x) || (j >= y))
        return;
    {
        int pixel_index = j * x + i;
        curandState local_state = rand_state[pixel_index];
        vec3 col(0, 0, 0);
        for (int k = 0; k < s; k++)
        {
            // Random jitter for anti-aliasings
            float r = float(i + curand_uniform(&local_state)) / float(x);
            float g = float(j + curand_uniform(&local_state)) / float(y);
            Ray ray = (*cam)->get_ray(r, g, &local_state); // Assuming camera is defined globally or passed in some way
            col += color(ray, world, &local_state);
        }
        // float u = float(i) / float(x);
        // float v = float(j) / float(y);
        // Ray r(origin, lower_left_corner + horizontal*u + vertical*v);
        rand_state[pixel_index] = local_state; // Save the state back to global memory
        col /= float(s);
        col[0] = sqrt(col[0]);
        col[1] = sqrt(col[1]);
        col[2] = sqrt(col[2]);
        pixels[pixel_index] = col;
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_cam)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int i = 0; i < 22 * 22 + 1 + 3; i++)
        {
            delete ((sphere *)d_list[i])->mat_ptr;
            delete d_list[i];
        }
        delete *d_world;
        delete *d_cam;
    }
}

// __host__ __device__ float hit_sphere(const vec3& center, float radius, const Ray& r) {
//     vec3 oc = r.origin() - center;
//     float a = dot(r.direction(), r.direction());
//     float b = 2.0f * dot(oc, r.direction());
//     float c = dot(oc, oc) - radius*radius;
//     float discriminant = b*b - 4*a*c;
//     if (discriminant < 0) {
//         return -1.0f; // No intersection
//     }
//     else {
//         return (-b - sqrt(discriminant)) / (2.0f * a); // Return the nearest intersection point
//     }
// }

#define RND (local_rand_state(generator))


void create_world_cpu(hitable **h_list, hitable **h_world, camera **h_cam, int x, int y, uniform_real_distribution<float> &local_rand_state, default_random_engine &generator) {
        h_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    h_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    h_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    h_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        h_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        h_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        h_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *h_world  = new hitable_list(h_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *h_cam   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(x)/float(y),
                                 aperture,
                                 dist_to_focus);
}

void free_world_cpu(hitable **h_list, hitable **h_world, camera **h_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)h_list[i])->mat_ptr;
        delete h_list[i];
    }
    delete *h_world;
    delete *h_camera;
}

void render_cpu(vec3 *pixels, hitable **world, camera **cam, int x, int y, int s, uniform_real_distribution<float> &local_rand_state, default_random_engine &generator)
{
    float r;
    float g;
    vec3 b;
    int pixel_index;

#pragma omp parallel for collapse(2) private(r, g, b, pixel_index)
    for (int j = y - 1; j >= 0; j--)
    {
        for (int i = 0; i < x; i++)
        {
            pixel_index = j * x + i;
            vec3 col(0, 0, 0);
            for (int k = 0; k < s; k++)
            {
                // Random jitter for anti-aliasings
                float r = float(i + local_rand_state(generator)) / float(x);
                float g = float(j + local_rand_state(generator)) / float(y);
                Ray ray = (*cam)->get_ray(r, g, local_rand_state, generator); // Assuming camera is defined globally or passed in some way
                col += color(ray, world, local_rand_state, generator);
            }
            col /= float(s);
            col[0] = sqrt(col[0]);
            col[1] = sqrt(col[1]);
            col[2] = sqrt(col[2]);
            pixels[pixel_index] = col;
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

void print_image(int x, int y, vec3 *pixels, ostream &file)
{
    file << "P3\n"
         << x << " " << y << "\n255\n";

    for (int j = y - 1; j >= 0; j--)
    {
        for (int i = 0; i < x; i++)
        {
            int pixel_index = j * x + i;
            file << int(255.99 * pixels[pixel_index].r()) << " " << int(255.99 * pixels[pixel_index].g()) << " " << int(255.99 * pixels[pixel_index].b()) << "\n";
        }
    }
}

bool verify_result(vec3 *pixels_gpu, vec3 *pixels_cpu, int x, int y)
{
    int pixel_index, pixel_gpu;
    vec3 gpu, cpu;
    for (int j = y - 1; j >= 0; j--)
    {
        for (int i = 0; i < x; i++)
        {
            pixel_index = j * x + i;
            gpu = pixels_gpu[pixel_index];
            cpu = pixels_cpu[pixel_index];
            if (cpu.r() != gpu.r() || cpu.g() != gpu.g() || cpu.b() != gpu.b())
            {
                return false;
            }
        }
    }
    return true;
}

void print_array(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        cout << "[" << i << "]: " << arr[i] << " ";
    }
    cout << endl;
}

int main(int argc, char *argv[])
{

    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    default_random_engine generator;
    uniform_real_distribution<float> distribution(0.0, 1.0);

    int x, y, s;

    if (argc != 2)
    {
        x = 200;
    }
    else
    {
        x = atoi(argv[1]);
    }

    y = x / 2;
    s = 10;
    int n = x * y;
    long bytes = n * sizeof(int);
    long vec3_bytes = x * y * sizeof(vec3);

    vec3 *h_pixels = (vec3 *)malloc(vec3_bytes);

    vec3 *d_pixels;
    checkCudaErrors(cudaMalloc((void **)&d_pixels, vec3_bytes));

        // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, n*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));
    
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hitable **d_list;
    int num_hitables = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));

    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));

    camera **d_cam;
    checkCudaErrors(cudaMalloc((void **)&d_cam, sizeof(camera *)));

    create_world<<<1, 1>>>(d_list, d_world, d_cam, x, y, d_rand_state2);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // vec3 lower_left_corner(-2.0, -1.0, -1.0);
    // vec3 horizontal(4.0, 0.0, 0.0);
    // vec3 vertical(0.0, 2.0, 0.0);
    // vec3 origin(0.0, 0.0, 0.0);

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(x / threads_per_block.x + 1, y / threads_per_block.y + 1);

    render_init<<<blocks_per_grid, threads_per_block>>>(x, y, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cout << "Rendering with GPU..." << endl;
    auto start = high_resolution_clock::now();
    render<<<blocks_per_grid, threads_per_block>>>(d_pixels, x, y, s, d_cam, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    auto stop = high_resolution_clock::now();

    checkCudaErrors(cudaMemcpy(h_pixels, d_pixels, vec3_bytes, cudaMemcpyDeviceToHost));
    auto duration = duration_cast<nanoseconds>(stop - start);
    cout << "Time taken by gpu: " << duration.count() / 1e9 << " seconds" << endl;

    float speedup = float(duration.count());

#ifdef PRINT
    string filename_gpu = "out-Ray-gpu.ppm";
    ofstream file_gpu(filename_gpu);
    print_image(x, y, h_pixels, file_gpu);
    file_gpu.close();
    cout << "Image saved to " << filename_gpu << endl;
#endif

    // Free memory

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_pixels));
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(d_rand_state));

    cudaDeviceReset();

    cout << "Continue with CPU rendering? (y/n): ";
    string choice;
    cin >> choice;
    if (choice.compare("y") || choice.compare("Y"))
    {

        vec3 *verify_pixels = (vec3 *)malloc(vec3_bytes);
        hitable **h_list = (hitable **)malloc(num_hitables * sizeof(hitable *));
        hitable **world = (hitable **)malloc(sizeof(hitable *));
        camera **cam = (camera **)malloc(sizeof(camera *));

        create_world_cpu(h_list, world, cam, x, y, distribution, generator);

        cout << "Rendering with CPU..." << endl;
        start = high_resolution_clock::now();
        render_cpu(verify_pixels, world, cam, x, y, s, distribution, generator);
        stop = high_resolution_clock::now();

        // print_array(h_rand_state_r, x * y);

#ifdef VERIFY
        if (verify_result(h_pixels, verify_pixels, x, y) == false)
        {
            cerr << "Verification failed!\n";
            return 0;
        }
#endif

        duration = duration_cast<nanoseconds>(stop - start);
        cout << "Time taken by cpu: " << duration.count() / (1e9) << " seconds" << endl;

        speedup = 1 / speedup;
        speedup *= float(duration.count());

        cout << "Speedup: " << speedup << endl;

#ifdef PRINT
        string filename_cpu = "out-Ray-cpu.ppm";
        ofstream file_cpu(filename_cpu);
        print_image(x, y, h_pixels, file_cpu);
        file_cpu.close();
        cout << "Image saved to " << filename_cpu << endl;
#endif
        free_world_cpu(h_list, world, cam);
        free(h_list);
        free(world);
        free(cam);
        free(h_pixels);
        free(verify_pixels);
    }
    else
    {
        cout << "Skipping CPU rendering." << endl;
    }

    return 0;
}