
#include <random>

#include <curand.h>
#include <curand_kernel.h>

__global__ void kernel_generate_random_uniform_points_gpu(float* x, float* y, int n, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curandState state;
        curand_init(seed, i, 0, &state);

        // Parameters for the normal distribution
        float mean = 0.5f;
        float stddev = 0.1f;

        x[i] = mean + stddev * curand_uniform(&state);
        y[i] = mean + stddev * curand_uniform(&state);
    }
}

void generate_random_uniform_points_gpu(int n, float* d_x, float* d_y) {
    // Define the number of threads and blocks
    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Use current time as a seed for the random number generator
    unsigned long seed = clock();

    kernel_generate_random_uniform_points_gpu<<<numBlocks, blockSize>>>(d_x, d_y, n, seed);

    // Check for errors in kernel launch
    checkCudaError2(cudaGetLastError(), "[kernel_generate_random_uniform_points_gpu] Kernel launch failed");
    // Check for errors on the GPU after kernel execution
    checkCudaError2(cudaDeviceSynchronize(), "CUDA Device Synchronization failed");
}

#include <omp.h>

void generate_random_uniform_points_omp(int n, float* x, float* y, unsigned long seed) {
    std::uniform_real_distribution<float> dist(0.5, 0.1);

    #pragma omp parallel  // Iniciar una sección paralela
    {
        std::random_device rd;  
        std::mt19937 generator(rd());  // Crear un generador de números aleatorios por hilo

        #pragma omp for  // Paralelizar el bucle for
        for(int i = 0; i < n; ++i) {
            x[i] = dist(generator);
            y[i] = dist(generator);
        }
    }
}

void generate_random_uniform_points(int n, float* x, float* y) {
    std::random_device rd;  
    std::mt19937 generator(rd());  // Crear un generador de números aleatorios por hilo
    std::uniform_real_distribution<float> dist(0.5, 0.1);
    for(int i = 0; i < n; i++) {
        x[i] = dist(generator);
        y[i] = dist(generator);
    }
}
