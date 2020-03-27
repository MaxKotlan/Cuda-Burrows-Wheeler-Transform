#include <iostream>
#include <vector>
#include <algorithm>


void printMatrix(std::vector<unsigned char>& data){
    int k =  data.size();
    for (int i = 0; i < k; i++){
        for(int j = 0; j < k; j++){
            unsigned int l = j-i;
            std::cout << data[l%k];
        }
        std::cout << std::endl;
    }   
}

void printSortedMatrix(std::vector<int>& indices, std::vector<unsigned char>& data){
    int k =  data.size();
    for (int i = 0; i < k; i++){
        for(int j = 0; j < k; j++){
            unsigned int l = j-indices[i];
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
            unsigned int la = i-a;
            unsigned int lb = i-b;
            resa += data[(la)%(k)]+ (UINT_MAX - 8*i*sizeof(unsigned char));
            resb += data[(lb)%(k)]+ (UINT_MAX - 8*i*sizeof(unsigned char));
        }
        return resa <= resb;
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
        std::cout << data[(-1 - indices[i])%(data.size())];


}

int main(int argc, char** argv){

    std::vector<unsigned char> data;

    
    for (auto c : "^BANANA@")
        if (c != '\0')
            data.push_back(c);
    //data.push_back('\0');

    printMatrix(data);

    burrowswheeler(data);

}