COMPILED = renderer-cpu renderer-cpu-array

renderer-cpu: renderer-cpu.cpp
	g++ renderer-cpu.cpp -o renderer-cpu

renderer-cpu-array: renderer-cpu-array.cpp
	g++ renderer-cpu-array.cpp -o renderer-cpu-array

renderer-gpu-array: renderer-gpu-array.cu
	nvcc renderer-gpu-array.cu -o renderer-gpu-array

clean: 
	rm ${COMPILED}