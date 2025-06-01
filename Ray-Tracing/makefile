COMPILED = main
LINKS = vectors/vec3_gpu.cu rays/ray_gpu.cu

main: main.cu $(LINKS)
	nvcc -g -G -o main -I $(LINKS) main.cu

all:
	make main

clean:
	rm -f $(COMPILED)