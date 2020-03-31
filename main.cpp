#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <time.h>
#include <chrono>
#include "bwt.h"

std::vector<unsigned char> readFileIntoBuffer(std::string filename){
    	FILE* file = fopen(filename.c_str(), "rb");
        if (!file) {
            std::cout << "Could not open " << filename << std::endl;
            exit(0);
        }
        fseek(file, 0, SEEK_END);
		int length = ftell(file)+1;
		fseek(file, 0, SEEK_SET);
        std::vector<unsigned char> buffer(length);
		fread(buffer.data(), buffer.size(), 1, file);
        return std::move(buffer);
}

TransformedData Transform(std::vector<unsigned char> &originaldata, std::string device){
    auto start = std::chrono::steady_clock::now();
    TransformedData transform = (device == "GPU") ? BWT_CUDA(originaldata) : BWT(originaldata);
    auto end = std::chrono::steady_clock::now();
    std::cout << device << " Elapsed time in milliseconds : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    return std::move(transform);
}

struct settings{
    bool print = false;
    std::string device = "GPU";
} settings;

int main(int argc, char** argv){
    if (argc < 2) {
        std::cout << "Incorrect Number of Arguments. Usage: " << std::endl;
        std::cout << "\t" << argv[0] << " [filepath/filename]" << std::endl;  
    }
    for (int i = 0; i < argc; i++){
        if (std::string(argv[i]) == std::string("--cpu"))   settings.device = "CPU";
        if (std::string(argv[i]) == std::string("--print")) settings.print  = true;
    }

    TransformedData d = Transform(readFileIntoBuffer(argv[1]), settings.device);
    
    if (settings.print){
        for (auto b : d.data){
            std::cout << b;
        }
        std::cout << std::endl;
    }
}