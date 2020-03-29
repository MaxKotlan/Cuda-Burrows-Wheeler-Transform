#include "bwt_cuda.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct KernelParameters{
    unsigned char* input;
    unsigned char* output;
    unsigned int*  indices;
    unsigned int  datasize;
};

__device__ void BWTBitonicSort(KernelParameters parameters){

}

__global__ void Main_Kernel_BWT(KernelParameters parameters){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < parameters.datasize){
        /*Initalize Indices to the integers*/
        parameters.indices[idx] = idx;
        __syncthreads();

        /*Sort Indices Using a Bitonic Sort*/
        BWTBitonicSort(parameters);
        __syncthreads();

        /*Convert Input Parameters to Output Parameters Using Sorted Indices*/
        parameters.output[idx] = parameters.input[(-1 - parameters.indices[idx] + parameters.datasize ) % parameters.datasize];
    }
}

TransformedData BWT_CUDA(const std::vector<unsigned char>& input){
    unsigned char* device_input = nullptr; unsigned char* device_output = nullptr; unsigned int* device_indices = nullptr;
    unsigned int k = input.size();
    std::vector<unsigned char> output(k);

    gpuErrchk(cudaMalloc((void **)&device_input,   k*sizeof(unsigned char)));
    gpuErrchk(cudaMalloc((void **)&device_output,  k*sizeof(unsigned char)));
    gpuErrchk(cudaMalloc((void **)&device_indices, k*sizeof(unsigned int)));

    gpuErrchk(cudaMemcpy(device_input,   input.data(), k*sizeof(unsigned char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_output, output.data(), k*sizeof(unsigned char), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    KernelParameters parameters = { device_input, device_output, device_indices, k };
    
    gpuErrchk(cudaEventRecord(start));
    unsigned int threadsperblock = 1024;
    Main_Kernel_BWT<<< k/threadsperblock+1, threadsperblock>>>(parameters);
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));

    gpuErrchk(cudaMemcpy(output.data(), device_output, k*sizeof(unsigned char), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_input)); gpuErrchk(cudaFree(device_indices)); gpuErrchk(cudaFree(device_output));

    for (auto c: output)
        std::cout << c;
    std::cout << std::endl;

    return {};
}
