//#include "scan.cu"

// Compaction a sparce array of integers.
// The array is sparse and contains only a few elements.
// it used a auxiliary to determinate if a element is present in the array.
// The auxiliary is a array of integers.
// the output is a array of integers.
// The output array has the same size of the input array.
template <typename T, typename V>
void compaction_serial(T *output, INDEX *h_num, V *auxiliary, T *input, INDEX size){
    INDEX i, j=0;
    for (i = 0; i < size; i++)
    {
        if (auxiliary[i] == 1)
        {
            output[j] = input[i];
            j++;
        }
    }
    *h_num = j;
}

// Parallel compaction of a sparce array of integers.
// The array is sparse and contains only a few elements.
// it used a auxiliary to determinate if a element is present in the array.
// The auxiliary is a array of integers.
// the output is a array of integers.
// The output array has the same size of the input array.
template <typename T, typename V>
void compaction_parallel(T *output, INDEX *h_num, V *auxiliary, T *input, INDEX size){
    INDEX i, j=0;
    #pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        if (auxiliary[i] == 1)
        {
            output[j] = input[i];
            j++;
        }
    }
    *h_num = j;
}

// store the elements of the input array using the auxiliary array.
// the auxiliary array said where the element is present in the input array.
// the input array is a array of integers.
// the auxiliary array is a array of integers.
// the output array is a array of integers.
// the output array has the same size of the input array.
// this function is in GPU.
template <typename T, typename V>
__global__ void store_array(T *output, INDEX *d_num, V *auxiliary, T *scan, T *input, INDEX n){
    INDEX off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off < n){
        // debug: print of variable, the element of the input array, auxiliary array and output array.
        //printf("%d %d %d %d\n", off, input[off], auxiliary[off], output[off]);
        if (auxiliary[off] == 1){
            output[scan[off]] = input[off];
        }
        if (off == n-1)
            *d_num = scan[off] + auxiliary[off];
    }
}

// gpu compactation of a sparce array of integers.
// The array is sparse and contains only a few elements.
// it used a auxiliary to determinate if a element is present in the array.
// The auxiliary is a array of integers.
// the output is a array of integers.
// The output array has the same size of the input array.
template <typename T, typename V>
void compaction_cub_scan(T *d_out, INDEX *d_num, V *bit_vector, T *d_in, INDEX n){
    INDEX *aux_scan;
    cudaMalloc(&aux_scan, sizeof(INDEX)*n);
    cudaMemset(aux_scan, 0, sizeof(INDEX)*n);
    scan_parallel_cub<INDEX>(bit_vector, aux_scan, n);  kernelCallCheck();
    store_array<<<(n+BSIZE-1)/BSIZE, BSIZE>>>(d_out, d_num, bit_vector, aux_scan, d_in, n);
    cudaDeviceSynchronize();  kernelCallCheck();
    //cudaFree(aux_scan);
}

// compaction an array of T.
// using the cub library.
// the input array is a array of T.
// the output array is a array of T.
// the flag array is a array of V.
template <typename T, typename V>
void compaction_cub(T *d_out, INDEX *d_num_selected_out, V *d_flags, T *d_in, INDEX n){
    //char *d_flags;               // e.g., [1, 0, 0, 1, 0, 1, 1, 0]
    //int  *d_num_selected_out;    // e.g., [ ]
    //cudaMalloc(&d_num_selected_out, sizeof(int));
    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, n);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run selection
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, n);
    // Free temporary storage
    //cudaFree(d_temp_storage);
}


template <typename T, typename B>
void compaction_scan_thrust(T *d_out, INDEX *d_num, B *d_bit_set, T *d_in, INDEX n){
    thrust::device_ptr<B> b_ptr = thrust::device_pointer_cast(d_bit_set);
	thrust::device_vector<T> q_out(n,0);
	thrust::exclusive_scan(thrust::device, b_ptr, b_ptr + n, q_out.begin(), 0);
	T *d_scan = thrust::raw_pointer_cast(&q_out[0]);
    store_array<<<(n+BSIZE-1)/BSIZE, BSIZE>>>(d_out, d_num, d_bit_set, d_scan, d_in, n);
    cudaDeviceSynchronize();  kernelCallCheck();
    // Free temporary storage
    //cudaFree(d_scan);
}

/*template <typename T, typename B>
void compaction_copy_thrust(T *d_in, B *d_bit_set, T *d_out, int n){
	auto result_end = thrust::copy_if(	thrust::device, p_ptr, p_ptr + n, q_vec.begin(), ff);
}*/

template <typename T, typename V>
__global__ void compact_partial_sums(T *output, INDEX *d_num, T *input, V *partial_sums, T *segmented_partial_sums, V *d_bit_vector, INDEX num_elements) {
	INDEX offset = threadIdx.x + blockIdx.x * blockDim.x;
	//const INDEX globalWarpIdx = (threadIdx.x + blockDim.x * blockIdx.x)/WARPSIZE;
	INDEX globalSegmentIdx = (threadIdx.x + blockDim.x * blockIdx.x)/WMMA_TILE_SIZE;
	if (offset < num_elements) {
        INDEX ind = (INDEX) partial_sums[offset] + segmented_partial_sums[globalSegmentIdx];
        if ((int)d_bit_vector[offset] == 1){
            if (ind > 0)
                output[ind-1] = input[offset];
        }
        if (offset == num_elements - 1)
            *d_num = ind;
	}
}


template <typename T, typename V>
void compaction_tc_scan(T *d_out, INDEX *d_num, V *d_bit_vector, T *d_in, INDEX n){
	INDEX num_segments = (n + WMMA_TILE_SIZE - 1)/WMMA_TILE_SIZE + 1; //(n + SEGMENT_SIZE - 1) / SEGMENT_SIZE;
    dim3 blockDim(BLOCK_DIM,1,1);
    dim3 gridDim((n + 8192 - 1)/8192,1,1);
	T *segmented_partial_sums;
	V *partial_sums; kernelCallCheck();
	cudaMalloc(&partial_sums,sizeof(V)*n); kernelCallCheck();
	cudaMalloc(&segmented_partial_sums,sizeof(T)*num_segments); kernelCallCheck();
    compute_wmma_segmented_prefixsum_256n_block_ps<T,V><<<gridDim, blockDim>>>(d_bit_vector, partial_sums, segmented_partial_sums, n); kernelCallCheck();
    cudaDeviceSynchronize();
	scan_parallel_cub<T>(segmented_partial_sums,segmented_partial_sums,num_segments); kernelCallCheck();
	compact_partial_sums<T,V><<<(n+BSIZE-1)/BSIZE+1, BSIZE>>>(d_out, d_num, d_in, partial_sums, segmented_partial_sums, d_bit_vector, n); kernelCallCheck();
    cudaDeviceSynchronize();//*/
	// free memorry
	cudaFree(segmented_partial_sums);
	cudaFree(partial_sums);
}