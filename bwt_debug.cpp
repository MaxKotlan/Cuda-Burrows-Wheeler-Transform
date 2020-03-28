#include "bwt_debug.h"
#include <iostream>
#include <algorithm>

void PrintUnsortedMatrix(std::string input){
    std::vector<unsigned char> data(input.size());
    for (int i = 0; i < input.size(); i++)
        data[i] = input[i];
    return PrintUnsortedMatrix(data);
}


void PrintUnsortedMatrix(const std::vector<unsigned char>& data){
    int k =  data.size();
    for (int i = 0; i < k; i++){
        std::cout << i << ": ";
        for(int j = 0; j < k; j++){
            unsigned int l = j-i+k;
            std::cout << data[l%k];
        }
        std::cout << std::endl;
    }   
}

void PrintSortedMatrix(std::string input){
    std::vector<unsigned char> data(input.size());
    for (int i = 0; i < input.size(); i++)
        data[i] = input[i];
    PrintSortedMatrix(data);
}

void PrintSortedMatrix(const std::vector<unsigned char>& data){
    unsigned int k =  data.size();
    std::vector<unsigned int> sortedindex(k);

    for (unsigned int i = 0; i < k; i++)
        sortedindex[i] = i;

    std::sort(sortedindex.begin(), sortedindex.end(), [&](unsigned int a, unsigned int b){
        unsigned int resa = 0; unsigned int resb = 0;
        for (int i = 0; i < k && resa == resb; i++){
            unsigned int la = i-a+k;
            unsigned int lb = i-b+k;
            resa += data[(la)%(k)];
            resb += data[(lb)%(k)];
        }
        return resa < resb;
    });
    
    for (int i = 0; i < k; i++){
        std::cout << i << ": ";
        for(int j = 0; j < k; j++){
            unsigned int l = j-sortedindex[i]+k;
            std::cout << data[l%k];
        }
        std::cout << std::endl;
    }   
}
