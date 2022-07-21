//#include "utils.cu"

/*#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/remove.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h> */

// scan serial calcule the prefix sum from the array in and write the result in out
template <typename T>
void prescan_serial(T *in, T *out, INDEX n) {
    INDEX i;
    out[0] = in[0];
    for (i = 1; i < n; i++) {
        out[i] = out[i - 1] + in[i];
    }
}

// the prefix sum starts from the first element of the array
template <typename T>
void scan_serial(T *in, T *out, INDEX n) {
    INDEX i;
    out[0] = 0;
    for (i = 1; i < n; i++) {
        out[i] = out[i-1] + in[i-1];
    }
}

// cub version of the scan parallel
template <typename T>
void scan_parallel_cub(T *in, T *out, INDEX n) {
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n); //kernelCallCheck();
    // Allocate temporary storage*/
    cudaMalloc(&d_temp_storage, temp_storage_bytes); //kernelCallCheck();
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n); //kernelCallCheck();
}


// typical version of the scan parallel that we can find in the literature
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

template <typename T>
__global__ void gpu_add_block_sums(T* d_out, T* d_in, T* d_block_sums, INDEX n) {
	INDEX d_block_sum_val = d_block_sums[blockIdx.x];
	INDEX cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (cpy_idx < n) {
		d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
		if (cpy_idx + blockDim.x < n)
			d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
	}
}

template <typename T>
__global__
void gpu_prescan(T* d_out, T* d_in, T* d_block_sums, INDEX len, INDEX shmem_sz, INDEX max_elems_per_block) {
	extern __shared__ INDEX s_out[];
	INDEX thid = threadIdx.x;
	INDEX ai = thid;
	INDEX bi = thid + blockDim.x;
	s_out[thid] = 0;
	s_out[thid + blockDim.x] = 0;
	if (thid + max_elems_per_block < shmem_sz)
		s_out[thid + max_elems_per_block] = 0;

	__syncthreads();
	INDEX cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
	if (cpy_idx < len) {
		s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
		if (cpy_idx + blockDim.x < len)
			s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
	}
	INDEX offset = 1;
	for (INDEX d = max_elems_per_block >> 1; d > 0; d >>= 1) {
		__syncthreads();

		if (thid < d)
		{
			INDEX ai = offset * ((thid << 1) + 1) - 1;
			INDEX bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			s_out[bi] += s_out[ai];
		}
		offset <<= 1;
	}

	if (thid == 0) { 
		d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1 
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
		s_out[max_elems_per_block - 1 
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
	}
	//printf("%i %i %i\n",cpy_idx,s_out[cpy_idx],s_out[cpy_idx + blockDim.x]);
	for (INDEX d = 1; d < max_elems_per_block; d <<= 1)	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			INDEX ai = offset * ((thid << 1) + 1) - 1;
			INDEX bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			INDEX temp = s_out[ai];
			s_out[ai] = s_out[bi];
			s_out[bi] += temp;
		}
	}
	__syncthreads();

	if (cpy_idx < len)
	{
		d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
		if (cpy_idx + blockDim.x < len)
			d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
	}
  //printf("%i %i %i\n",cpy_idx,(INDEX)d_out[cpy_idx],(INDEX)d_out[cpy_idx + blockDim.x]);
}

template <typename T>
void sum_scan_blelloch(T* d_out, T* d_in, INDEX n) {
	cudaMemset(d_out, 0, n * sizeof(T));
	INDEX block_sz = 2*BSIZE/2;
	INDEX max_elems_per_block = 2 * block_sz; 
	INDEX grid_sz = (n + max_elems_per_block - 1)/ max_elems_per_block;

	// Conflict free padding requires that shared memory be more than 2 * block_sz
	INDEX shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

	T* d_block_sums;
	cudaMalloc(&d_block_sums, sizeof(T) * grid_sz);
	cudaMemset(d_block_sums, 0, sizeof(T) * grid_sz);

  	gpu_prescan<T><<<grid_sz, block_sz, sizeof(T) * shmem_sz>>>(d_out, 
																	d_in, d_block_sums, n, 
																	shmem_sz,
																	max_elems_per_block);

	if (grid_sz <= max_elems_per_block)	{
		T* d_dummy_blocks_sums;
		cudaMalloc(&d_dummy_blocks_sums, sizeof(T));
		cudaMemset(d_dummy_blocks_sums, 0, sizeof(T));
		gpu_prescan<T><<<1, block_sz, sizeof(T) * shmem_sz>>>(d_block_sums, 
																	d_block_sums, 
																	d_dummy_blocks_sums, 
																	grid_sz, 
																	shmem_sz,
																	max_elems_per_block);
    	cudaFree(d_dummy_blocks_sums);
	}
	else {
		T* d_in_block_sums;
		cudaMalloc(&d_in_block_sums, sizeof(T) * grid_sz);
		cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(T) * grid_sz, cudaMemcpyDeviceToDevice);
		sum_scan_blelloch<T>(d_block_sums, d_in_block_sums, grid_sz);
		cudaFree(d_in_block_sums);
	}
	gpu_add_block_sums<T><<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, n);
	cudaFree(d_block_sums);
}

template <typename T>
thrust::device_vector<T> scan_parallel_thrust(thrust::device_ptr<T> q_ptr, INDEX n){
	thrust::device_vector<T> q_out(n,0);
	thrust::exclusive_scan(thrust::device, q_ptr, q_ptr + n, q_out.begin(), 0);
	return q_out;
}


template <typename T>
static __device__ T one() {
  	return T{1};
}

template <typename T>
static __device__ T zero() {
  	return T{0};
}

#define WARP_PER_BLOCK 32
#define SEGMENT_SIZE 256 * WARP_PER_BLOCK
#define BLOCK_DIM WARP_PER_BLOCK * WARPSIZE

// Compute scan using Tensor Cores
//template <INDEX SEGMENT_SIZE, INDEX WARPS_PER_BLOCK, INDEX BLOCK_DIM>
template <typename T, typename V>
static __global__ void compute_wmma_segmented_prefixsum_256n_block_ps(V *d_in, V *d_out, T *partial_sums, INDEX num_segments) {
	
	__shared__ half u_frag_s[WMMA_TILE_SIZE];
	__shared__ half l_frag_s[WMMA_TILE_SIZE];
	__shared__ half la_mat_s[SEGMENT_SIZE];
	//INDEX acc = 0; // only use the first 16
	
	INDEX localWarpIdx = threadIdx.x / WARPSIZE;
	INDEX local_offset = localWarpIdx * WMMA_TILE_SIZE;
	INDEX laneid = threadIdx.x % WARPSIZE;
	//const INDEX globalSegmentIdx = (threadIdx.x + blockDim.x * blockIdx.x)/SEGMENT_SIZE;
	INDEX globalWarpIdx = (threadIdx.x + blockDim.x * blockIdx.x)/WARPSIZE;
	INDEX offset = local_offset + blockIdx.x*SEGMENT_SIZE; //global_offset + (+ localWarpIdx) * WMMA_TILE_SIZE;
	
	#pragma unroll
	for (INDEX idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
		INDEX ii = idx / N;
		INDEX jj = idx % N;
		u_frag_s[idx] = ii <= jj ? one<half>() : zero<half>();
		l_frag_s[idx] = ii <= jj ? zero<half>() : one<half>();
	}
	
	__syncthreads();
	
	wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
	wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> l_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> o_frag;
	wmma::fragment<wmma::accumulator, M, N, K, half> la_frag;
	wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> la_mat_frag;
	wmma::fragment<wmma::accumulator, M, N, K, half> au_frag;
	wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;
	
	wmma::load_matrix_sync(u_frag, u_frag_s, 16);
	wmma::load_matrix_sync(l_frag, l_frag_s, 16);
	wmma::fill_fragment(o_frag, one<half>());
	wmma::fill_fragment(out_frag, zero<half>());

	wmma::fill_fragment(out_frag, zero<half>());
	wmma::fill_fragment(la_frag, zero<half>());
	wmma::load_matrix_sync(a_frag, d_in + offset, 16);
	wmma::load_matrix_sync(b_frag, d_in + offset, 16);

	wmma::mma_sync(au_frag, a_frag, u_frag, out_frag);
	wmma::mma_sync(la_frag, l_frag, b_frag, la_frag);

	// store accumulator la_frag into shared memory and load it into
	// matrix_a
	// fragment la_mat_frag
	wmma::store_matrix_sync(la_mat_s + local_offset, la_frag, 16, wmma::mem_row_major);
	wmma::load_matrix_sync(la_mat_frag, la_mat_s + local_offset, 16);

	wmma::mma_sync(out_frag, la_mat_frag, o_frag, au_frag);

	wmma::store_matrix_sync(d_out + offset, out_frag, 16, wmma::mem_row_major);

	//__syncthreads();
	if (laneid == 0) {
		partial_sums[globalWarpIdx] = d_out[offset + 255];
		// printf globalsegmentIdx, partial_sums[globalSegmentIdx]
		//printf("%i %i %i %i %i %i\n",blockIdx.x, offset, globalWarpIdx, partial_sums[globalWarpIdx],offset + 255,(INDEX)d_out[offset + 255]);
	}

	/*// Block segmentation scan, This code is not used in the current implementation and not tested.
	//__syncthreads();

	// first get the partial sum of the current warp, only use the first WARP_PER_BLOCK threads
	__shared__ half partial_sums_s[WARP_PER_BLOCK];
	if(threadIdx.x < WARP_PER_BLOCK)
		acc = d_out[(threadIdx.x+1) * WMMA_TILE_SIZE - 1];

	// then, do the scan on the warp accumulation
	__syncthreads();
    // Specialize WarpScan for type T
    typedef cub::WarpScan<T> WarpScan;
    // Allocate WarpScan shared memory for WARP_PER_BLOCK warps
    __shared__ typename WarpScan::TempStorage temp_storage[WARP_PER_BLOCK];
    // Obtain one input item per thread
    // Compute inclusive warp-wide prefix sums
    WarpScan(temp_storage[localWarpIdx]).InclusiveSum(acc, acc);
	//__syncthreads();

	// store the partial sum of the current warp in shared memory
	if(threadIdx.x < WARP_PER_BLOCK){
		partial_sums[threadIdx.x] = partial_sums_s;
		//printf("%i %i %i %i\n",threadIdx.x,partial_sums_s,local_offset-1,partial_sums[threadIdx.x]);
	}
	__syncthreads();

	// finally, do the scan on the block accumulation, this part is incomplete...
	#pragma unroll
	for(INDEX i = 0; i < SEGMENT_SIZE; i+=BLOCK_DIM){
		d_out[threadIdx.x + i] += + partial_sums_s[localWarpIdx];
	*/
}


template <typename T>
__global__ void add_partial_sums(T *output, half *partial_sums, T *segmented_partial_sums, INDEX num_elements) {
	INDEX offset = threadIdx.x + blockIdx.x * blockDim.x;
	//const INDEX globalWarpIdx = (threadIdx.x + blockDim.x * blockIdx.x)/WARPSIZE;
	INDEX globalSegmentIdx = (threadIdx.x + blockDim.x * blockIdx.x)/WMMA_TILE_SIZE;
	if (offset < num_elements) {
		output[offset+1] += (T) partial_sums[offset] + segmented_partial_sums[globalSegmentIdx];
	}
}


// scan parallel calcule the prefix sum using CUDA-TC programming from the array in and write the result in out
template <typename T>
void scan_parallel_tc(half *in, T *out, INDEX n) {
	//cout << "--> " << SEGMENT_SIZE << ", " << WARP_PER_BLOCK << ", " << BLOCK_DIM << endl;
	INDEX num_segments = (n + WMMA_TILE_SIZE - 1)/WMMA_TILE_SIZE + 1; //(n + SEGMENT_SIZE - 1) / SEGMENT_SIZE;
    dim3 blockDim(BLOCK_DIM,1,1);
    dim3 gridDim((n + 8192 - 1)/8192,1,1);
	// print num_segments, n, n, blockDim, gridDim
	//printf("-> %i %i %i %i\n",num_segments,n,SEGMENT_SIZE,(int)(n + 2048 - 1)/2048);
	T *segmented_partial_sums;
	half *partial_sums;
	cudaMalloc(&segmented_partial_sums,sizeof(T)*num_segments);
	cudaMalloc(&partial_sums,sizeof(half)*n);
	//cudaDeviceSynchronize();
    compute_wmma_segmented_prefixsum_256n_block_ps<T,half><<<gridDim, blockDim>>>(in, partial_sums, segmented_partial_sums, n); kernelCallCheck();
    //cudaDeviceSynchronize();
	scan_parallel_cub<T>(segmented_partial_sums,segmented_partial_sums,num_segments); kernelCallCheck();
	/*T *aux;
	cudaMalloc(&aux,sizeof(T)*num_segments);
	sum_scan_blelloch<T>(aux,segmented_partial_sums,num_segments); kernelCallCheck();
	//cudaDeviceSynchronize();//*/
	add_partial_sums<T><<<(n+BSIZE-1)/BSIZE+1, BSIZE>>>(out, partial_sums, segmented_partial_sums, n); kernelCallCheck(); //<256, SEGMENT_SIZE>*/
    cudaDeviceSynchronize();//*/
	// free memorry
	//cudaFree(segmented_partial_sums);
	//cudaFree(partial_sums);
}