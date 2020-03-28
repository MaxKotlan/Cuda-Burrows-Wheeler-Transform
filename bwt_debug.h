
#ifndef BWT_DEBUG_H
#define BWT_DEBUG_H
#include <vector>
#include <string>

void PrintUnsortedMatrix(std::string data);
void PrintUnsortedMatrix(const std::vector<unsigned char>& data);

void PrintSortedMatrix(std::string data);
void PrintSortedMatrix(const std::vector<unsigned char>& data);

#endif