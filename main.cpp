#include <iostream>
#include <vector>
#include <algorithm>



void burrowswheeler(std::vector<unsigned char>& data){
    
    std::vector<unsigned char> buffer(data.size()*data.size());
    std::vector<unsigned int> index(data.size());
    for (int i = 0; i < data.size(); i++)
        index[i] = i;

    for (int j = 0; j < data.size(); j++)
        for (int i = 0; i < data.size(); i++)
            buffer[i+j*data.size()] = data[(i-j)%data.size()];

    for (int j = 0; j < data.size(); j++){
        for (int i = 0; i < data.size(); i++)
            std::cout << buffer[i+j*data.size()];
        std::cout << std::endl;
    }

    for (auto i : index)
        std::cout << i;
    std::cout << std::endl;

    std::sort(index.begin(), index.end(), [buffer, data](int i1, int i2){ 
        int r1 = 0, r2 = 0;
        for (int i = 0; i < data.size(); i++){
            r1 += buffer[i1*data.size() + i]+i*10;
            r2 += buffer[i2*data.size() + i]+i*10; 
        }
        return r1 < r2; 
    });

    std::cout << std::endl;
    for (auto i : index){
        for (int j = 0; j < data.size(); j++)
            std::cout << buffer[i*data.size() + j];
        std::cout << std::endl;
    }
    std::cout << std::endl;

}

int main(int argc, char** argv){

    std::vector<unsigned char> data{'4', '2', '1', '1', '2', '2', '3', '1', '4'};
    burrowswheeler(data);

}