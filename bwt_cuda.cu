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

__device__ void swap ( unsigned int& a, unsigned int& b )
{
    unsigned int c = a; 
    a=b; 
    b=c;
}

__device__ bool sortcompare( const unsigned int& a, const unsigned int& b, unsigned char* input, unsigned int datasize){
    unsigned char diffa = 0; unsigned char diffb = 0;
    for (int i = 0; i < datasize && diffa == diffb; i++){
        unsigned int la = i-a+datasize;
        unsigned int lb = i-b+datasize;
        diffa = input[(la)%(datasize)];
        diffb = input[(lb)%(datasize)];
    }
    return diffa < diffb;
}

__device__ void BitonicMerge(KernelParameters parameters) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (unsigned int k = 2; k <= parameters.datasize; k *= 2){
        for (unsigned int j = k / 2; j>0; j /= 2){
            //if(sortcompare(parameters.indices[i+j], parameters.indices[i], parameters.input, parameters.datasize))
            if (parameters.indices[i]>parameters.indices[i+j])
                swap(parameters.indices[i], parameters.indices[i+j]);
            __syncthreads();
        }
    }
}

__device__ void BitonicSort(KernelParameters parameters){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    for (unsigned int k = 2; k <= parameters.datasize; k <<= 1){
        for (unsigned int j=k>>1; j>0; j=j>>1){
            unsigned int ixj = i ^ j;
            if ((ixj)>i) {
                if ((i&k)==0) {
                    //if(sortcompare(parameters.indices[ixj], parameters.indices[i], parameters.input, parameters.datasize)){
                    if (parameters.indices[i]>parameters.indices[ixj]) {
                    //if (atomicMax(&parameters.indices[i], parameters.indices[ixj]) == )
                        swap(parameters.indices[i], parameters.indices[ixj]);
                    }
                }
                if ((i&k)!=0) {
                    /* Sort descending */
                    //if (sortcompare(parameters.indices[i], parameters.indices[ixj], parameters.input, parameters.datasize)){
                    if (parameters.indices[i]<parameters.indices[ixj]) {
                        swap(parameters.indices[i], parameters.indices[ixj]); 
                    }
                }
            }   
            __syncthreads();
        }
    }
}

__global__ void Main_Kernel_BWT(KernelParameters parameters){
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < parameters.datasize){
        /*Initalize Indices to the integers*/
        parameters.indices[idx] = idx%parameters.datasize;
        __syncthreads();

        /*Sort Indices Using a Bitonic Sort*/
        BitonicSort(parameters);
        BitonicMerge(parameters);
        __syncthreads();

        /*Convert Input Parameters to Output Parameters Using Sorted Indices*/
        parameters.output[idx] = parameters.input[(-1 - parameters.indices[idx] + parameters.datasize ) % parameters.datasize];
    }
}

TransformedData BWT_CUDA(const std::vector<unsigned char>& input){
    unsigned char* device_input = nullptr; unsigned char* device_output = nullptr; unsigned int* device_indices = nullptr;
    unsigned int k = input.size();
    std::vector<unsigned char> output(k); std::vector<unsigned int> indices(k);

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
    gpuErrchk(cudaMemcpy(indices.data(), device_indices, k*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_input)); gpuErrchk(cudaFree(device_indices)); gpuErrchk(cudaFree(device_output));

    for (auto c: indices)
        std::cout << c << " ";
    std::cout << std::endl;

    for (auto c: output)
        std::cout << c;
    std::cout << std::endl;

    return {};
}
