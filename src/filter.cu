#include <cuda.h>
#include <cfloat>
#include <omp.h>

#include "filter.cuh"
#include "utility/utilities.cuh"
#include "kernel/kernel.cu"
#include "cuda_time_m.cu"

#include "filter_gpu_scan.cu"
#include "filter_cub_flagged.cu"
#include "filter_thrust_scan.cu"
#include "filter_thrust_copy.cu"
#include "filter_cpu_serial.cpp"
#include "filter_cpu_parallel.cpp"

// include generate points functions
#include "utility/normal_random_points.cu"
#include "utility/circumference_random_points.cu"
#include "utility/uniform_random_points.cu" //*/
