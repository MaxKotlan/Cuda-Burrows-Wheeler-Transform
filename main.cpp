#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <time.h>
#include "bwt.h"

std::vector<unsigned char> readFileIntoBuffer(std::string filename){
    	FILE* file = fopen(filename.c_str(), "rb");
        fseek(file, 0, SEEK_END);
		int length = ftell(file)+1;
		fseek(file, 0, SEEK_SET);
        std::vector<unsigned char> buffer(length);
		fread(buffer.data(), buffer.size(), 1, file);
        return std::move(buffer);
}

void CompareCpuAndKernel(std::string testdata){
    std::vector<unsigned char> data;
    for (auto c : testdata)
        if (c != '\0')
            data.push_back(c);
    auto t = BWT(data);
    std::cout << "Cpu Version: " << std::endl;
    for (auto c : t.data)
        std::cout << c;
    std::cout << std::endl;
    std::cout << "Gpu Version: " << std::endl;
    BWT_CUDA(data);
    std::cout << std::endl;
}

int main(int argc, char** argv){
    CompareCpuAndKernel("0123456789ABCDEF");
    CompareCpuAndKernel("SIX.MIXED.PIXIES.SIFT.MIXED.PIXISIX.MIXED.PIXIES.SIFT.MIXED.PIXI");
    CompareCpuAndKernel("There are laboratory tests that can identify the virus that caus");
}