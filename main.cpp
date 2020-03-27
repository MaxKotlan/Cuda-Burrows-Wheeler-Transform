#include <iostream>
#include <vector>
#include <algorithm>


void printMatrix(std::vector<unsigned char>& data){
     for (int i = 0; i < data.size(); i++){
        for(int j = 0; j < data.size(); j++)
            std::cout << data[(j - i)%(data.size())];
        std::cout << std::endl;
    }   
}

void printNthColumn(std::vector<unsigned char>& data, unsigned int column){
    for (int i = 0; i < data.size(); i++){
        std::cout << data[(column - i)%(data.size())] << std::endl;
    }
}

std::vector<int> sortIndex(std::vector<unsigned char>& data){
    std::vector<int> indices(data.size());
    for (int i = 0; i < indices.size(); i++)
        indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&](int a, int b){
        int resa = 0; int resb = 0;
        for (int i = 0; i < data.size() && resa == resb; i++){
            resa += data[(i - a)%(data.size())]+ (INT_MAX - sizeof(unsigned char)) >> i;
            resb += data[(i - b)%(data.size())]+ (INT_MAX - sizeof(unsigned char)) >> i;
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

    
    for (auto c : "Tomorrow_and_tomorrow_and_tomorrow")
        if (c != '\0')
            data.push_back(c);
    data.push_back('\0');

    burrowswheeler(data);

}