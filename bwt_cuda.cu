#include "bwt_cuda.h"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


struct BWTCompare{
    unsigned int  input_size;
    unsigned char* input_device;
    __host__ __device__
    bool operator()(unsigned int a, unsigned int b) const{
        unsigned char diffa = 0; unsigned char diffb = 0;
        for (int i = 0; i < input_size && diffa == diffb; i++){
            unsigned int la = i-a+input_size;
            unsigned int lb = i-b+input_size;
            diffa = input_device[(la)%(input_size)];
            diffb = input_device[(lb)%(input_size)];
        }
        return diffa < diffb;
    }

};

struct KernelParameters{
    unsigned char* input;
    unsigned char* output;
    unsigned int*  indices;
    unsigned int  datasize;
};

__global__ void IndiciesToTransform(KernelParameters parameters){ 
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < parameters.datasize){
        parameters.output[idx] = parameters.input[((parameters.datasize - 1) - parameters.indices[idx] + parameters.datasize ) % parameters.datasize];
    }
}

TransformedData BWT_CUDA(const std::vector<unsigned char>& data){
    unsigned int k = data.size();

    TransformedData result;
    result.data = std::move(std::vector<unsigned char>(k));
    thrust::device_vector<unsigned char> input(data);
    thrust::device_vector<unsigned char> output(k);
    thrust::device_vector<unsigned int> indices(k);

    thrust::sequence(indices.begin(), indices.end());
    BWTCompare comparator = {k, thrust::raw_pointer_cast(input.data())};
    thrust::sort(indices.begin(), indices.end(), comparator);

    KernelParameters parameters = { 
        thrust::raw_pointer_cast(input.data()),  
        thrust::raw_pointer_cast(output.data()),
        thrust::raw_pointer_cast(indices.data()),
        k
    };
    unsigned int threadsperblock = 1024;
    IndiciesToTransform<<< k/threadsperblock+1, threadsperblock>>>(parameters);
    
    thrust::copy(output.begin(), output.end(), result.data.begin());

    for (auto c: result.data)
        std::cout << c;
    std::cout << std::endl;

    return std::move(result);
}