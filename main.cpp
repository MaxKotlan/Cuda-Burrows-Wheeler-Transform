#include <iostream>
#include <vector>


void burrowswheeler(std::vector<unsigned char>& data){
    for (int j = 0; j < data.size(); j++){
        for (int i = 0; i < data.size(); i++){
            std::cout << i;
        }std::cout << std::endl;
    }
}

int main(int argc, char** argv){

    std::vector<unsigned char> data;
    burrowswheeler(data);

}