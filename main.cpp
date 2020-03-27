#include <iostream>
#include <vector>
#include <algorithm>
#include "bwt.h"

void printMatrix(std::vector<unsigned char>& data){
    int k =  data.size();
    for (int i = 0; i < k; i++){
        for(int j = 0; j < k; j++){
            unsigned int l = j-i+k;
            std::cout << data[l%k];
        }
        std::cout << std::endl;
    }   
}

void printSortedMatrix(std::vector<int>& indices, std::vector<unsigned char>& data){
    int k =  data.size();
    for (int i = 0; i < k; i++){
        std::cout << i << ": ";
        for(int j = 0; j < k; j++){
            unsigned int l = j-indices[i]+k;
            std::cout << data[l%k];
        }
        std::cout << std::endl;
    }   
}

void printNthColumn(std::vector<unsigned char>& data, unsigned int column){
    for (int i = 0; i < data.size(); i++){
        std::cout << data[(column - i)%(data.size())] << std::endl;
    }
}

std::vector<int> sortIndex(std::vector<unsigned char>& data){
    int k = data.size();
    std::vector<int> indices(k);
    
    for (int i = 0; i < k; i++)
        indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&](int a, int b){
        unsigned int resa = 0; unsigned int resb = 0;
        for (int i = 0; i < k && resa == resb; i++){
            unsigned int la = i-a+k;
            unsigned int lb = i-b+k;
            resa += data[(la)%(k)]+ (k*8*sizeof(unsigned char) - 8*i*sizeof(unsigned char));
            resb += data[(lb)%(k)]+ (k*8*sizeof(unsigned char) - 8*i*sizeof(unsigned char));
        }
        return resa < resb;
    } );

    return std::move(indices);
}

void burrowswheeler(std::vector<unsigned char>& data){
    //std::vector<unsigned int> index(data.size());
    //printMatrix(data);
    //printNthColumn(data, 0);
    std::vector<int> indices = sortIndex(data);
    //for (int i = 0; i < indices.size(); i++)
    //    indices[i] = i;
    std::cout << std::endl;
    printSortedMatrix(indices, data);
    /* for (int i = 0; i < data.size(); i++){
        for(int j = 0; j < data.size(); j++)
            std::cout << data[(j - indices[i])%(data.size())];
        std::cout << std::endl;
    }*/
    std::cout << std::endl << "Transform: ";
    for (int i = 0; i < data.size(); i++)
        std::cout << data[(-1 - indices[i] + data.size())%(data.size())];


}

int main(int argc, char** argv){

    std::vector<unsigned char> data;
    
    for (auto c : "SIX.MIXED.PIXIES.SIFT.SIXTY.PIXIE.DUST.BOXES")
        if (c != '\0')
            data.push_back(c);

    burrowswheeler(data);

    BWT::TransformedData t = BWT::BWT(data);
    std::cout << t.originalIndex << std::endl;
    for (auto c : t.data)
        std::cout << c;

}