#include "string_adapter.h"

std::vector<unsigned char> toByteArray(const std::string& input){
    return std::move(std::vector<unsigned char>(input.begin(), input.end()));
}

std::string toString(const std::vector<unsigned char>& input){
    return std::move(std::string(input.begin(), input.end()));
}
