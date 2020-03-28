all: main test

main: main.cpp
	nvcc -o BWT main.cpp bwt_cpu.cpp
test: test.cpp
	nvcc -o TestBWT test.cpp bwt_cpu.cpp