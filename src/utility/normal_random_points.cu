
#include <omp.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>



__global__ void kernel_generate_random_normal_points_gpu(float* x, float* y, int n, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curandState state;
        curand_init(seed, i, 0, &state);
        
        // Parameters for the normal distribution
        float mean = 0.5f;
        float stddev = 0.1f;

        x[i] = mean + stddev * curand_normal(&state);
        y[i] = mean + stddev * curand_normal(&state);
    }
}

void checkCudaError2(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        //std::cerr << "ERROR: " << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}


void generate_random_normal_points_gpu(int n, float *d_x, float *d_y) {
    // Define the number of threads and blocks
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Use current time as a seed for the random number generator
    unsigned long seed = clock();

    // Launch the kernel
    kernel_generate_random_normal_points_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, n, seed);

    // Check for errors in kernel launch
    checkCudaError2(cudaGetLastError(), "[kernel_generate_random_normal_points_gpu] Kernel launch failed");
    // Check for errors on the GPU after kernel execution
    checkCudaError2(cudaDeviceSynchronize(), "CUDA Device Synchronization failed");
}


void generate_random_normal_points_omp(int n, float *x, float *y, unsigned long seed) {
    double mean = 0.5;
    double stddev = 0.1;

    // Usar una distribución normal con la media y desviación estándar especificadas
    std::normal_distribution<float> dist(mean, stddev);

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


void generate_random_normal_points(int n, float *x, float *y) {
    double mean = 0.5;
    double stddev = 0.1;

    std::random_device rd;  
    std::mt19937 generator(rd());
    std::normal_distribution<float> dist(mean, stddev);

    for(int i = 0; i < n; ++i) {
        x[i] = dist(generator);
        y[i] = dist(generator);
    }
}