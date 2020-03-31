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
        fclose(file);
        return std::move(buffer);
}

void saveBufferToFile(std::string filename, TransformedData& transformed){
        filename+=".transformed";
    	FILE* file = fopen(filename.c_str(), "wb");
        if (!file) {
            std::cout << "Could not open " << filename << std::endl;
            exit(0);
        }
		fseek(file, 0, SEEK_SET);
        std::vector<unsigned char> fileheader(8);
		fileheader[0] = 0xC2; fileheader[1] = 0xB3; fileheader[2] = 0x23; fileheader[3] = 0x12;
		std::copy(static_cast<const unsigned char*>(static_cast<const void*>(&transformed.originalIndex)),
			static_cast<const unsigned char*>(static_cast<const void*>(&transformed.originalIndex)) + sizeof(transformed.originalIndex),
			&fileheader[4]);

        fwrite(fileheader.data(), fileheader.size(), 1, file);
        fwrite(transformed.data.data(), transformed.data.size(), 1, file);
        fclose(file);
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
    saveBufferToFile(argv[1], d);
    
    if (settings.print){
        for (auto b : d.data){
            std::cout << b;
        }
        std::cout << std::endl;
    }
}