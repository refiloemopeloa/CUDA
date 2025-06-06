COMPILED = main
LINKS = include/vec3_gpu.cu include/ray_gpu.cu include/hitable_gpu.cu include/hitable_list_gpu.cu include/sphere_gpu.cu include/camera_gpu.cu
GENCODE_FLAGS = -gencode arch=compute_86,code=sm_86


main: main.cu $(LINKS)
	nvcc -g -G -m64 -Xcompiler -fopenmp $(GENCODE_FLAGS) -o main main.cu

all:
	make main

clean:
	rm -f $(COMPILED)