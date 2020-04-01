all: library main test

library:
	nvcc --lib -o BurrowWheelerTransform.lib bwt_cpu.cpp bwt_cuda_bitonicsort.cu bwt_cuda.cu bwt_debug.cpp string_adapter.cpp
main: main.cpp
	nvcc -o BWT main.cpp -l BurrowWheelerTransform
test: test.cpp
	nvcc -o TestBWT test.cpp -l BurrowWheelerTransform