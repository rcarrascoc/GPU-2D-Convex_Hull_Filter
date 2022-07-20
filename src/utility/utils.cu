#include <cuda.h>
#include <mma.h>
#include <cub/cub.cuh> 

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/remove.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>

#define WARPSIZE 32
#define WARPS_PER_BLOCK 8
#define BSIZE 256

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUASSERT: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define kernelCallCheck() \
	{ gpuErrCheck( cudaPeekAtLastError() ); \
	gpuErrCheck( cudaDeviceSynchronize() ); } 

using namespace nvcuda;
static const int M              = 16;
static const int N              = 16;
static const int K              = 16;
static const int WMMA_TILE_SIZE = (M * N);

// kernel convert to half
__global__ void convert_to_half(float *in, half *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

template <typename T>
__global__ void convert_to_half(T *in, half *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// kernel convert form half to T 
template <typename T>
__global__ void convert_from_half(half *in, T *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}
