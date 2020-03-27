#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
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

int main(int argc, char** argv){
    std::vector<unsigned char> data;
    for (auto c : "SIX.MIXED.PIXIES.SIFT.SIXTY.PIXIE.DUST.BOXES")
        if (c != '\0')
            data.push_back(c);

    BWT::TransformedData t = BWT::BWT(data);
    std::cout << t.originalIndex << std::endl;
    for (auto c : t.data)
        std::cout << c;

    std::vector<unsigned char> lotr = readFileIntoBuffer("lotr.txt");
    BWT::TransformedData lotr_t = BWT::BWT(lotr);
    for (auto c : lotr_t.data)
        std::cout << c;


}