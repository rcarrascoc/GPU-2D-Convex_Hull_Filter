//#include "utils.cu"

// find the index of minimum element in an array
template <typename T>
void findMin(INDEX *min_index, T *arr, INDEX n){
    T min = arr[0];
    *min_index = 0;
    for (INDEX i = 1; i < n; i++) {
        if (arr[i] < min) {
            min = arr[i];
            *min_index = i;
        }
    }
    //printf("\nMinimum element is %f at index %i", (float)min, (int)*min_index);
}

// find the index of minimum element in an array using parallelism and reduction
template <typename T>
void findMin_parallel_reduction(INDEX *min_index, T *arr, INDEX n){
    INDEX min = arr[0];
    INDEX min_index_aux = 0;
    #pragma omp parallel for reduction(min:min, min_index_aux)
    for (INDEX i = 1; i < n; i++) {
        if (arr[i] < min) {
            min = arr[i];
            min_index_aux = i;
        }
    }
    *min_index = min_index_aux;
    //printf("\nMinimum element is %f at index %i", (float)min, (int)*min_index);
}

// find the index of minimum element in an array using CUDA programming model
template <typename T>
__device__ INDEX min_index_atomic(INDEX *addr, INDEX value, T *d_in){
		INDEX old = *addr, assumed;
        while(d_in[old] > d_in[value]){
			assumed = old;
			old = atomicCAS((unsigned int*)addr, assumed, value);
        }
        return value;
}

template <typename T>
__inline__ __device__ INDEX min_index_warp(INDEX val, T *d_in){
	INDEX val_tmp;
	for (INDEX offset = WARPSIZE >> 1; offset > 0; offset >>= 1){
		val_tmp = __shfl_down_sync(0xFFFF, val, offset, WARPSIZE);
		if (d_in[val_tmp] < d_in[val])
				val = val_tmp; 
	}
	//__syncthreads();
	return val;
}

template <typename T>
__inline__ __device__ INDEX min_index_block(INDEX val, T *d_in){
	__shared__ INDEX shared[WARPSIZE];
	INDEX tid = threadIdx.x; 
	INDEX lane = tid & (WARPSIZE-1);
	INDEX wid = tid/WARPSIZE;
	shared[lane] = 0;
	val = min_index_warp<T>(val, d_in);
	__syncthreads();
	if(lane == 0)
		shared[wid] = val;
	__syncthreads();
	val = (tid < blockDim.x/WARPSIZE) ? shared[lane] : 0;
	if(wid == 0)
		val = min_index_warp<T>(val, d_in);
	return val;
}

template <typename T>
__global__ void min_index_kernel(INDEX *out, T *d_in, INDEX n){
	INDEX off = blockIdx.x * blockDim.x + threadIdx.x; 
	if(off < n){
		INDEX val = off;
		val = min_index_block<T>(val, d_in); 
		//__syncthreads();
		if(threadIdx.x == 0)
			min_index_atomic<T>(out, val, d_in);
	}
}

// find the index of minimum element in an array using CUDA programming model
template <typename T>
void findMin_kernel(INDEX *d_min_index, T *d_in, INDEX n){
    min_index_kernel<T><<<(n+BSIZE-1)/BSIZE, BSIZE>>>(d_min_index, d_in, n);
    //cudaDeviceSynchronize();
}

// find the index of minumum element in an array using cub and cuda programming model
template <typename T>
void findMin_cub(cub::KeyValuePair<int, T> *d_out, T *d_in, INDEX n){
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run argmin-reduction
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
}

// find the index of minimum element in an array using thrust and cuda programming model
template <typename T>
void findMin_thrust(INDEX *min_index, thrust::device_ptr<float> d_in_ptr, INDEX n){
    //thrust::device_ptr<float> d_in_ptr = thrust::device_pointer_cast(d_input);
    thrust::device_vector<float>::iterator d_min = thrust::min_element(d_in_ptr, d_in_ptr + n);
    *min_index = d_min - (thrust::device_vector<float>::iterator)d_in_ptr;
}