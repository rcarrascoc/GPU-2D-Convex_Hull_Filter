#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

#include <omp.h>
#include <cstdlib>

template <typename T>
__device__ void kernel_calculate_point(float alpha, float ra, T &x, T &y) {
    if (alpha == 0 || alpha == 360) {
        x = (alpha == 0) ? ra : -ra;
        y = 0;
    } else if (alpha == 90 || alpha == 270) {
        x = 0;
        y = (alpha == 90) ? ra : -ra;
    } else {
        float tg = tan(alpha * M_PI / 180.0);
        x = ra / sqrt(1 + tg * tg);
        y = x * tg;
        if (alpha > 90 && alpha < 270) {
            x *= -1;
        }
    }
}

template <typename T>
__global__ void kernel_generate_random_circumference_points(T *X, T *Y, int n, double prob, float cx, float cy, float ra, int N, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curandState state;
        curand_init(seed, i, 0, &state);
        
        int aux;    
        float x, y, alpha, fact;
        fact = curand_uniform(&state);
        alpha = fact * 360.0;

        if (i < N) {
            kernel_calculate_point(alpha, ra, x, y);
        } else {
            float dt = ra * prob;
            fact = curand_uniform(&state);
            aux = curand(&state);
            kernel_calculate_point(alpha, (aux % 2 == 0) ? ra + fact * dt : ra - fact * dt, x, y);
        }

        X[i] = cx + x;
        Y[i] = cy + y;
    }
}



template <typename T>
void generate_random_circumference_points_gpu(int n, T *d_x, T *d_y, double prob){
    // Use current time as a seed for the random number generator
    unsigned long seed = clock();

    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;
    int N = n - static_cast<int>(n * prob);
    kernel_generate_random_circumference_points<<<numBlocks, blockSize>>>(d_x, d_y, n, prob, 0.5f, 0.5f, 1.0f, N, seed);

    // Check for errors in kernel launch
    //checkCudaError2(cudaGetLastError(), "[kernel_generate_random_uniform_points_gpu] Kernel launch failed");
    // Check for errors on the GPU after kernel execution
    checkCudaError2(cudaDeviceSynchronize(), "CUDA Device Synchronization failed");
} 


#include <cmath>
#include <random>

template <typename T>
void calculate_point(float alpha, float ra, T &x, T &y) {
    if (alpha == 0 || alpha == 360) {
        x = (alpha == 0) ? ra : -ra;
        y = 0;
    } else if (alpha == 90 || alpha == 270) {
        x = 0;
        y = (alpha == 90) ? ra : -ra;
    } else {
        float tg = tan(alpha * M_PI / 180.0);
        x = ra / sqrt(1 + tg * tg);
        y = x * tg;
        if (alpha > 90 && alpha < 270) {
            x *= -1;
        }
    }
}

template <typename T>
void generate_random_circumference_points_omp(int n, T *X, T *Y, double prob) {
    srand(1);
    int N = n - static_cast<int>(n * prob);
    float cx = 0.5, cy = 0.5, ra = 1.0;

    #pragma omp parallel
    {
        std::mt19937 gen(rand());

        #pragma omp for
        for (int i = 0; i < n; i++) {
            float x, y, alpha, fact;
            fact = static_cast<float>(gen()) / gen.max();
            alpha = fact * 360.0;

            if (i < N) {
                calculate_point(alpha, ra, x, y);
            } else {
                float dt = ra * prob;
                fact = static_cast<float>(gen()) / gen.max();
                calculate_point(alpha, (gen() % 2 == 0) ? ra + fact * dt : ra - fact * dt, x, y);
            }

            X[i] = cx + x;
            Y[i] = cy + y;
        }
    }
}

template <typename T>
void generate_random_circumference_points(int n, T *X, T *Y, double prob){
    srand(1);
	int i;
	int N = n - n*prob;		// points on the circumference.	CASO 1
	float cx, cy, ra;
	float x, y, alpha, tg, fact;
	
	ra = 1.0;
	cx = cy = 0.5;	// Center (0.5, 0.5)

	// we generate N random points ON the circumference...
	//srand (time(NULL));
	// FOR PARA EL CASO 1
	for(i=0; i<N; i++){
		fact = static_cast <float> (rand())/static_cast <float> (RAND_MAX);
		alpha = fact*360.0;	// generate a random angle: 0 <= alpha <= 360
		if (alpha == 0 || alpha == 360){
			y = 0;
			if (alpha == 0)
				x = ra;
			else
				x = -1*ra;
		}else if (alpha == 90 || alpha == 270){
			x = 0;
			if (alpha == 90)
				y = ra;
			else
				y = -1*ra;
		} else {
			tg = tan(alpha);
			x = ra / sqrt(1 + pow(tg, 2.0));
			if (alpha > 90 && alpha < 270)
				x*=-1;
			y = x*tg;
		}
		//cout << alpha << "(" << x << "," << y << ") " << endl;
		// here know x^2 + y^2 = ra^2
		X[i]= cx+x;
		Y[i]= cy+y;
	}

	float dt = ra*prob;
	// we generate n-N random points INSIDE OF / OR ARAOUND OF the circumference...
	for(; i<n; i++){
		fact = static_cast <float> (rand())/static_cast <float> (RAND_MAX);
		alpha = fact*360.0;	// generate a random angle: 0 <= alpha <= 360
		fact = static_cast <float> (rand())/static_cast <float> (RAND_MAX);
		if (alpha == 0 || alpha == 360){
			y = 0;
			if (alpha == 0){
				x = ra+fact*dt;
			}else
				x = ra-fact*dt;
		}else if (alpha == 90 || alpha == 270){
			x = 0;
			if (alpha == 90)
				y = ra+fact*dt;
			else
				y = ra-fact*dt;
		} else {
			tg = tan(alpha);
			if (rand()%2)
				x = (ra+fact*dt) / sqrt(1 + pow(tg, 2.0));
			else
				x = (ra-fact*dt) / sqrt(1 + pow(tg, 2.0));

			if (alpha > 90 && alpha < 270)
				x*=-1;
			y = x*tg;
		}
		//cout << alpha << "(" << x << "," << y << ") " << endl;
		// at this moment: x^2 + y^2 = ra^2
		X[i]= cx+x;
		Y[i]= cy+y;
	}
}
