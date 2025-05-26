cmake_minimum_required(VERSION 3.18)

# RTX 3050 specific: Compute Capability 8.9
set(CMAKE_CUDA_ARCHITECTURES "121")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.9/bin/nvcc")
project(CUDA-Vector-Add LANGUAGES CXX CUDA)

# First ensure CUDA toolkit is found
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda-12.9/")

# Debug/Release configurations
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -lineinfo -Xcompiler -rdynamic")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xcompiler -O3 --use_fast_math")

# Enable separable compilation (recommended for Ampere)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Add executable
add_executable(vector-add
        vector-add.cu  # Your main CUDA file
)

# Modern CUDA configurations
target_compile_features(vector-add PRIVATE cuda_std_17)
target_link_libraries(vector-add PRIVATE
        CUDA::cudart
#        CUDA::cudadevrt  # Required for separable compilation
)

# Generate PTX for forward compatibility
#set_property(TARGET vector-add PROPERTY CUDA_PTX_COMPILATION ON)

## Optional: Enable CUDA Thrust
#find_package(Thrust REQUIRED)
#target_link_libraries(vector-add PRIVATE Thrust::Thrust)