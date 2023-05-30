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
void prescan_serial(T *out, T *in, int n) {
    int i;
    out[0] = in[0];
    for (i = 1; i < n; i++) {
        out[i] = out[i - 1] + in[i];
    }
}

// the prefix sum starts from the first element of the array
template <typename T>
void scan_serial(T *out, T *in, int n) {
    int i;
    out[0] = 0;
    for (i = 1; i < n; i++) {
        out[i] = out[i-1] + in[i-1];
    }
}

// cub version of the scan parallel
template <typename T>
void scan_parallel_cub(T *out, T *in, int n) {
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
__global__ void gpu_add_block_sums(T* d_out, T* d_in, T* d_block_sums, int n) {
	int d_block_sum_val = d_block_sums[blockIdx.x];
	int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (cpy_idx < n) {
		d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
		if (cpy_idx + blockDim.x < n)
			d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
	}
}

template <typename T>
__global__
void gpu_prescan(T* d_out, T* d_in, T* d_block_sums, int len, int shmem_sz, int max_elems_per_block) {
	extern __shared__ int s_out[];
	int thid = threadIdx.x;
	int ai = thid;
	int bi = thid + blockDim.x;
	s_out[thid] = 0;
	s_out[thid + blockDim.x] = 0;
	if (thid + max_elems_per_block < shmem_sz)
		s_out[thid + max_elems_per_block] = 0;

	__syncthreads();
	int cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
	if (cpy_idx < len) {
		s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
		if (cpy_idx + blockDim.x < len)
			s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
	}
	int offset = 1;
	for (int d = max_elems_per_block >> 1; d > 0; d >>= 1) {
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
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
	for (int d = 1; d < max_elems_per_block; d <<= 1)	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int temp = s_out[ai];
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
  //printf("%i %i %i\n",cpy_idx,(int)d_out[cpy_idx],(int)d_out[cpy_idx + blockDim.x]);
}

template <typename T>
void sum_scan_blelloch(T* d_out, T* d_in, int n) {
	cudaMemset(d_out, 0, n * sizeof(T));
	int block_sz = 2*BSIZE/2;
	int max_elems_per_block = 2 * block_sz; 
	int grid_sz = (n + max_elems_per_block - 1)/ max_elems_per_block;

	// Conflict free padding requires that shared memory be more than 2 * block_sz
	int shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

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
thrust::device_vector<T> scan_parallel_thrust(thrust::device_ptr<T> q_ptr, int n){
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
//template <int SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
template <typename T, typename V>
static __global__ void compute_wmma_segmented_prefixsum_256n_block_ps(V *d_out, T *partial_sums, V *d_in, int num_segments) {
	
	__shared__ half u_frag_s[WMMA_TILE_SIZE];
	__shared__ half l_frag_s[WMMA_TILE_SIZE];
	__shared__ half la_mat_s[SEGMENT_SIZE];
	//int acc = 0; // only use the first 16
	
	const int localWarpIdx = threadIdx.x / WARPSIZE;
	const int local_offset = localWarpIdx * WMMA_TILE_SIZE;
	const int laneid = threadIdx.x % WARPSIZE;
	//const int globalSegmentIdx = (threadIdx.x + blockDim.x * blockIdx.x)/SEGMENT_SIZE;
	const int globalWarpIdx = (threadIdx.x + blockDim.x * blockIdx.x)/WARPSIZE;
	const int offset = local_offset + blockIdx.x*SEGMENT_SIZE; //global_offset + (+ localWarpIdx) * WMMA_TILE_SIZE;
	
	#pragma unroll
	for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
		const auto ii = idx / N;
		const auto jj = idx % N;
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
		//printf("%i %i %i %i %i %i\n",blockIdx.x, offset, globalWarpIdx, partial_sums[globalWarpIdx],offset + 255,(int)d_out[offset + 255]);
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
	for(int i = 0; i < SEGMENT_SIZE; i+=BLOCK_DIM){
		d_out[threadIdx.x + i] += + partial_sums_s[localWarpIdx];
	*/
}


template <typename T>
__global__ void add_partial_sums(T *output, half *partial_sums, T *segmented_partial_sums, int num_elements) {
	const int offset = threadIdx.x + blockIdx.x * blockDim.x;
	//const int globalWarpIdx = (threadIdx.x + blockDim.x * blockIdx.x)/WARPSIZE;
	const int globalSegmentIdx = (threadIdx.x + blockDim.x * blockIdx.x)/WMMA_TILE_SIZE;
	if (offset < num_elements) {
		output[offset+1] += (T) partial_sums[offset] + segmented_partial_sums[globalSegmentIdx];
	}
}


// scan parallel calcule the prefix sum using CUDA-TC programming from the array in and write the result in out
template <typename T>
void scan_parallel_tc(T *out, half *in, int n) {
	//cout << "--> " << SEGMENT_SIZE << ", " << WARP_PER_BLOCK << ", " << BLOCK_DIM << endl;
	int num_segments = (n + WMMA_TILE_SIZE - 1)/WMMA_TILE_SIZE + 1; //(n + SEGMENT_SIZE - 1) / SEGMENT_SIZE;
    dim3 blockDim(BLOCK_DIM,1,1);
    dim3 gridDim((n + 8192 - 1)/8192,1,1);
	// print num_segments, n, n, blockDim, gridDim
	//printf("-> %i %i %i %i\n",num_segments,n,SEGMENT_SIZE,(int)(n + 2048 - 1)/2048);
	T *segmented_partial_sums;
	half *partial_sums;
	cudaMalloc(&segmented_partial_sums,sizeof(T)* num_segments);
	cudaMalloc(&partial_sums,sizeof(half)*n);
	//cudaDeviceSynchronize();
    compute_wmma_segmented_prefixsum_256n_block_ps<T,half><<<gridDim, blockDim>>>(partial_sums, segmented_partial_sums, in, n); kernelCallCheck();
    //cudaDeviceSynchronize();
	T *aux;
	cudaMalloc(&aux,sizeof(T)*num_segments);
	scan_parallel_cub<T>(aux, segmented_partial_sums, num_segments); kernelCallCheck();
	//sum_scan_blelloch<T>(aux,segmented_partial_sums,num_segments); kernelCallCheck();
	//cudaDeviceSynchronize();//*/
	add_partial_sums<T><<<(n+BSIZE-1)/BSIZE, BSIZE>>>(out, partial_sums, aux, n); kernelCallCheck(); //<256, SEGMENT_SIZE>*/
    cudaDeviceSynchronize();//*/

	/*// copy partial_sums to out
	T *h_out = (T*)malloc(sizeof(T)*n);
	cudaMemcpy(h_out,segmented_partial_sums,sizeof(T)*(num_segments),cudaMemcpyDeviceToHost);
	// print h_out
	for(int i = 0; i < num_segments; i++)
		//if ((int)h_out[i] != i) 
			printf("%i %i\n",i,(int)h_out[i]);//*/


	// free memorry
	//cudaFree(segmented_partial_sums);
	//cudaFree(partial_sums);
}

template <typename T>
__global__ void add_partial_sums_2(T *output, half *d_in, T *sums_warp, T *sums_block, int num_elements) {
	const int offset = threadIdx.x + blockIdx.x * blockDim.x;
	//const int globalWarpIdx = (threadIdx.x + blockDim.x * blockIdx.x)/WARPSIZE;
	const int globalSegmentIdx = offset >> 13; // /8192;
	const int globalWarpIdx = offset >> 8; // /256
	if (offset < num_elements) {
		output[offset+1] = (T)d_in[offset] + sums_warp[globalWarpIdx] + sums_block[globalSegmentIdx];
		//printf("%i %i %i %i\n",offset,(int)partial_sums[offset],globalSegmentIdx,(int)segmented_partial_sums[globalSegmentIdx]);
	}
}


// scan
__inline__ __device__ int warp_scan(int val, int lane){
	for (int offset = 1; offset < WARPSIZE; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffff, val, offset, WARPSIZE);
		if ((lane & 31) >= offset)
			val += n;
	}
	return val;
}

// Compute scan using Tensor Cores
//template <int SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
template <typename T, typename V>
static __global__ void compute_wmma_segmented_prefixsum_256n_block_ps_2(V *d_out, T *sums_warp, T *sums_block, V *d_in, int num_segments) {
	
	T acc = 0;
	//__shared__ T partial_acc;
	__shared__ half u_frag_s[WMMA_TILE_SIZE];
	__shared__ half l_frag_s[WMMA_TILE_SIZE];
	__shared__ half la_mat_s[SEGMENT_SIZE];
	//int acc = 0; // only use the first 16

	//__shared__ V l_sum[WARP_PER_BLOCK];
	//__shared__ V l_out[SEGMENT_SIZE];
	
	const int localWarpIdx = threadIdx.x >> 5;// WARPSIZE;
	const int local_offset = localWarpIdx * WMMA_TILE_SIZE;
	//const int laneid = threadIdx.x % WARPSIZE;
	//const int globalSegmentIdx = (threadIdx.x + blockDim.x * blockIdx.x)/SEGMENT_SIZE;
	const int globalWarpIdx = (threadIdx.x + blockDim.x * blockIdx.x) >> 5; //WARPSIZE;
	const int offset = local_offset + blockIdx.x*SEGMENT_SIZE; //global_offset + (+ localWarpIdx) * WMMA_TILE_SIZE;
	
	#pragma unroll
	for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
		const auto ii = idx / N;
		const auto jj = idx % N;
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
	//wmma::store_matrix_sync(l_out + local_offset, out_frag, 16, wmma::mem_row_major);

	/* //__syncthreads();
	if (laneid == 0) {
		partial_sums[globalWarpIdx] = d_out[offset + 255];
		// printf globalsegmentIdx, partial_sums[globalSegmentIdx]
		//printf("%i %i %i %i %i %i\n",blockIdx.x, offset, globalWarpIdx, partial_sums[globalWarpIdx],offset + 255,(int)d_out[offset + 255]);
	}*/

	// Block segmentation scan, This code is not used in the current implementation and not tested.
	//__syncthreads();

	// first get the partial sum of the current warp, only use the first WARP_PER_BLOCK threads
	//__shared__ T partial_sums_s[WARP_PER_BLOCK];
	/*__syncthreads();
	if(threadIdx.x < WARP_PER_BLOCK) {
		acc = (T)d_out[threadIdx.x*256 + blockIdx.x*8192 + 255]; // offset 255
		sums_warp[threadIdx.x+blockIdx.x*WARP_PER_BLOCK + 1]  = warp_scan(acc, threadIdx.x);
		printf("-> %i %i %i %i %i\n",(T)d_out[threadIdx.x*256 + blockIdx.x*8192 + 255], threadIdx.x*256 + blockIdx.x*8192 + 255, threadIdx.x, (T)sums_warp[threadIdx.x+blockIdx.x*WARP_PER_BLOCK + 1], acc);
	}
	__syncthreads();*/
	/*if(threadIdx.x < blockDim.x/WARPSIZE) {
		printf("%i %i %i %i %i\n",blockIdx.x, laneid, localWarpIdx, threadIdx.x*256  + 255, (int)acc);
	}//*/

	__syncthreads();
	// then, do the scan on the warp accumulation
	if (threadIdx.x < WARP_PER_BLOCK) {
		acc = d_out[threadIdx.x*WMMA_TILE_SIZE + blockIdx.x*SEGMENT_SIZE + WMMA_TILE_SIZE - 1];
		//printf("-> %i %i %i\n",threadIdx.x, threadIdx.x*256 + blockIdx.x*8192 + 255, (int)acc);
	}
	__syncthreads();
    // Specialize WarpScan for type T
    typedef cub::WarpScan<T> WarpScan;
    // Allocate WarpScan shared memory for WARP_PER_BLOCK warps
    __shared__ typename WarpScan::TempStorage temp_storage[WARP_PER_BLOCK];
    // Obtain one input item per thread
    // Compute inclusive warp-wide prefix sums
	//__syncthreads();
    WarpScan(temp_storage[globalWarpIdx]).InclusiveSum(acc, acc);
	__syncthreads(); 

	if (threadIdx.x < WARP_PER_BLOCK - 1) {
		sums_warp[threadIdx.x+blockIdx.x*WARP_PER_BLOCK + 1] = acc;
		//printf("-----> %i %i %i\n",threadIdx.x, threadIdx.x*256 + blockIdx.x*8192 + 255, (int)acc);
	}
	__syncthreads();

	// store the partial sum of the current warp in shared memory
	if(threadIdx.x == WARP_PER_BLOCK - 1) {
		//partial_acc = (T) acc;
		sums_block[blockIdx.x] = acc; //(T) sums_warp[WARP_PER_BLOCK];// acc;//temp_storage[threadIdx.x];
		if (blockIdx.x == 0) {
			sums_warp[0] = 0;
			//sums_block[0] = 0;
		}
		//l_sum[0] = 0;
		//printf("%i %i\n",blockIdx.x, threadIdx.x);
	}
	//__syncthreads();
	/*if (threadIdx.x < blockDim.x/WARPSIZE) {
		printf("%i %i %i %i %i\n",threadIdx.x,blockIdx.x,(int)partial_sums[blockIdx.x], (int)acc, (int)l_sum[threadIdx.x] );
	}//*/

	// finally, do the scan on the block accumulation, this part is incomplete...
	/*#pragma unroll
	for(int i = 0; i < SEGMENT_SIZE; i+=BLOCK_DIM){
		partial_sums[threadIdx.x + i + blockIdx.x*8192] = (T)l_out[i+threadIdx.x] + (T)d_out[((i+threadIdx.x))/256]; 
		__syncthreads();
	}//*/
	
	/*__syncthreads();
	if (threadIdx.x/32 == 19 || threadIdx.x/32 == 18) {
		// print the threadid, d_in[threadid], d_out[threadid], partial_sums[threadid]
		printf("---> %i %i %i %i %i\n", threadIdx.x/32, threadIdx.x, (int) d_in[threadIdx.x + (threadIdx.x/32)*256],(int) d_out[threadIdx.x + (threadIdx.x/32)*256], (int) sums_warp[threadIdx.x/32]);
	} // */
}

// scan parallel calcule the prefix sum using CUDA-TC programming from the array in and write the result in out
template <typename T>
void scan_parallel_tc_2(T *out, half *in, int n) {
	//cout << "--> " << SEGMENT_SIZE << ", " << WARP_PER_BLOCK << ", " << BLOCK_DIM << endl;
	int num_segments = (n + 255) >> 8; //256;
	int num_block = (n + 8191) >> 13; //8192; //(n + SEGMENT_SIZE - 1) / SEGMENT_SIZE; //(n + 32 - 1)/32 + 1; //
    dim3 blockDim(BLOCK_DIM,1,1);
    dim3 gridDim(num_block,1,1);
	// print num_segments, n, n, blockDim, gridDim
	//printf("-> %i %i %i %i\n",num_segments,n,SEGMENT_SIZE,(int)(n + 2048 - 1)/2048);
	T *sums_block;
	T *sums_warp;
	half *sums_thread;
	cudaMalloc(&sums_block,sizeof(T)*num_block);
	cudaMalloc(&sums_warp,sizeof(T)*(num_segments));
	cudaMalloc(&sums_thread,sizeof(half)*n);
	//cudaDeviceSynchronize();
    compute_wmma_segmented_prefixsum_256n_block_ps_2<T,half><<<gridDim, blockDim>>>(sums_thread, sums_warp, sums_block, in, n); kernelCallCheck();
    cudaDeviceSynchronize();
	
	T *aux;
	cudaMalloc(&aux,sizeof(T)*num_block);
	scan_parallel_cub<T>(aux,sums_block,num_block); kernelCallCheck();
	//cudaDeviceSynchronize();
	//sum_scan_blelloch<T>(aux,sums_block,num_segments); kernelCallCheck();
	//cudaDeviceSynchronize();//*/
	add_partial_sums_2<T><<<(n+BSIZE-1)/BSIZE, BSIZE>>>(out, sums_thread, sums_warp, aux, n); kernelCallCheck(); //<256, SEGMENT_SIZE>*/
    cudaDeviceSynchronize();//*/

	/*// copy partial_sums to out
	T *h_out = (T*)malloc(sizeof(T)*num_block);
	cudaMemcpy(h_out,sums_block,sizeof(T)*num_block,cudaMemcpyDeviceToHost);
	// print h_out
	for(int i = 0; i < num_block; i++)
		//if ((int)h_out[i] != i) 
			printf("%i %i\n",i,(int)h_out[i]);

	printf("num_segments: %i\n",num_segments); //*/
	// free memorry
	//cudaFree(segmented_partial_sums);
	//cudaFree(partial_sums);
	//printf("-> %i",BSIZE);
}