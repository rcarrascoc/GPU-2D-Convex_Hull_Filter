//#include "utils.cu"

// find the max index of the array, output the max value and its index, and the input is of type T
template <typename T>
void findMax(INDEX *max_index, T *arr, INDEX n) {
    T max = arr[0];
    *max_index = 0;
    for (INDEX i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
            *max_index = i;
        }
    }
}

// find the index of max element in an array using parallelism and reduction
template <typename T>
void findMax_parallel(INDEX *max_index, T *arr, INDEX n) {
    INDEX max_index_local = 0;
    T max_local = arr[0];
    #pragma omp parallel for reduction(max:max_local, max_index_local)
    for (INDEX i = 1; i < n; i++) {
        if (arr[i] > max_local) {
            max_local = arr[i];
            max_index_local = i;
        }
    }
    *max_index = max_index_local;
}


// find the index of minimum element in an array using CUDA programming model
template <typename T>
__device__ INDEX max_index_atomic(INDEX *addr, INDEX value, T *d_in){
	INDEX old = *addr, assumed;
	while(d_in[old] < d_in[value]){
		assumed = old;
		old = atomicCAS((unsigned int*)addr, assumed, value);
	}
	return value;
}

template <typename T>
__inline__ __device__ INDEX max_index_warp(INDEX val, T *d_in){
	INDEX val_tmp;
	for (INDEX offset = WARPSIZE >> 1; offset > 0; offset >>= 1){
		val_tmp = __shfl_down_sync(0xFFFF, val, offset, WARPSIZE);
		if (d_in[val_tmp] > d_in[val])
				val = val_tmp; 
	}
	//__syncthreads();
	return val;
}

template <typename T>
__inline__ __device__ INDEX max_index_block(INDEX val, T *d_in){
	__shared__ INDEX shared[WARPSIZE];
	INDEX tid = threadIdx.x; 
	INDEX lane = tid & (WARPSIZE-1);
	INDEX wid = tid/WARPSIZE;
	shared[lane] = 0;
	val = max_index_warp<T>(val, d_in);
	__syncthreads();
	if(lane == 0)
		shared[wid] = val;
	__syncthreads();
	val = (tid < blockDim.x/WARPSIZE) ? shared[lane] : 0;
	if(wid == 0)
		val = max_index_warp<T>(val, d_in);
	return val;
}

template <typename T>
__global__ void max_index_kernel(INDEX *out, T *d_in, INDEX n){
	INDEX off = blockIdx.x * blockDim.x + threadIdx.x; 
	if(off < n){
		INDEX val = off;
		val = max_index_block<T>(val, d_in); 
		//__syncthreads();
		if(threadIdx.x == 0)
			max_index_atomic<T>(out, val, d_in);
	}
}

// find the index of minimum element in an array using CUDA programming model
template <typename T>
void findMax_kernel(INDEX *d_max_index, T *d_in, INDEX n){
    max_index_kernel<T><<<(n+BSIZE-1)/BSIZE, BSIZE>>>(d_max_index, d_in, n);
    //cudaDeviceSynchronize();
}

// find the index of minumum element in an array using cub and cuda programming model
template <typename T>
void findMax_cub(cub::KeyValuePair<int, T> *d_out, T *d_in, INDEX n){
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run argmin-reduction
    cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
}

// find the index of minimum element in an array using thrust and cuda programming model
template <typename T>
void findMax_thrust(INDEX *max_index, thrust::device_ptr<float> d_in_ptr, INDEX n){
    //thrust::device_ptr<float> d_in_ptr = thrust::device_pointer_cast(d_input);
    thrust::device_vector<float>::iterator d_max = thrust::max_element(d_in_ptr, d_in_ptr + n);
    *max_index = d_max - (thrust::device_vector<float>::iterator)d_in_ptr;
}