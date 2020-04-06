all: library main test

library:
	nvcc --lib -o BurrowWheelerTransform.lib bwt_cpu.cpp bwt_cuda_bitonicsort.cu bwt_cuda.cu bwt_debug.cpp string_adapter.cpp
main: main.cpp
	nvcc -o BWT main.cpp -L ./ BurrowWheelerTransform.lib
test: test.cpp
	nvcc -o TestBWT test.cpp -L ./ BurrowWheelerTransform.lib
