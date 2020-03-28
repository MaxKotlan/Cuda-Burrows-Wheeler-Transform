#include "bwt_cuda.h"

struct KernelParameters{
    unsigned char* input;
    unsigned char* output;
    unsigned int  datasize;
    unsigned int*  indices;
};


__global__ void KERNEL_BWT(KernelParameters parameters){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < parameters.datasize){
        parameters.indices[idx] = idx;
        __syncthreads();


    
        parameters.output[idx] = parameters.input[(-1 - parameters.indices[idx] + parameters.datasize ) % parameters.datasize];
    }
}

TransformedData BWT_CUDA(const std::vector<unsigned char>& data){

}
