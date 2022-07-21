#include <stdio.h>
#include <string>

// define global variables
#define REAL float
#define INDEX uint
#define REPEATS 1

// include filter.cuh
#include "src/filter.cuh"

// include tool functions
#include "src/tools.cpp"

using namespace std;

#define test(ffx, ffy, ffn, ffalg) ffalg(ffx,ffy,ffn)
#define test2(ffthis, ffx, ffy, ffn, ffalg) ffalg *ffthis = new ffalg(ffx,ffy,ffn)
#define test3(ffthis, ffx, ffy, ffn, ffalg) ffthis = new ffalg(ffx,ffy,ffn)
#define test4(ffthis, ffp, ffn, ffalg) ffthis = new ffalg(ffp,ffn)

string arr_alg[4] = {"gpu kernel", "cub scan", "thrust scan", "thrust copy_if"};
string arr_shape[3] = {"normal distribution", "uniform distribution", "circumference distribution"};

// main function
// where read four arguments and call the corresponding function
// read: size of the array, algorithm to use, shape of distribution, probability of the distribution
int main(int argc, char *argv[]) {
    // check if the number of arguments is correct
    if (argc != 5) {
        printf("Error: wrong number of arguments\n");
        printf("Usage: ./main size algorithm shape probability\n");
        printf("size: size of the array\n");
        printf("algorithm: 0 kernel, 1 thrust-scan, 2 thrust-copy, 3 cub-flagged \n");
        printf("shape: 0 for uniform, 1 for normal, 2 for circumference\n");
        printf("probability: [0..1] probability of the distribution\n");
        return 1;
    }
    
    // read the arguments
    INDEX size = atoi(argv[1]);
    int algorithm = atoi(argv[2]);
    int shape = atoi(argv[3]);
    double probability = atof(argv[4]);
    
    // check if the size is correct
    if (size <= 0) {
        printf("Error: size must be positive\n");
        return 1;
    }
    
    /*// print size, algorithm, shape, probability
    printf("size: %d points\n", size);
    printf("algorithm: %s\n", arr_alg[algorithm].c_str());
    printf("shape: %s\n", arr_shape[shape].c_str());
    printf("probability: %.2f%\n", probability*100);//*/

    // initialize x and y arrays of type REAL
    REAL *x = new REAL[size];
    REAL *y = new REAL[size];

    // call the corresponding function
    // for generating x and y arrays
    switch (shape) {
        case 0:
            generate_random_points_normal<REAL>(x, y, size);
            break;
        case 1:
            generate_random_points_uniform<REAL>(x, y, size);
            break;
        case 2:
            generate_random_points_circumference<REAL>(x, y, size, probability);
            break;
        default:
            printf("Error: algorithm must be 0, 1 or 2\n");
            return 1;
    }

    // print the points
    //print_points(x, y, size);

    // start the timer
    float cuda_time = 0.0f, cuda_time_acc = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();

    //test2(thisHull, x, y, size, filter_gpu_scan);

    if (algorithm == 0){
        filter_gpu_scan *thisHull;
        //test3(thisHull, x, y, size, filter_gpu_scan);
        //delete thisHull;
        cudaDeviceSynchronize();
        for (int i = 0; i < REPEATS; i++){
            cudaEventRecord(start);
            test3(thisHull, x, y, size, filter_gpu_scan);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&cuda_time, start, stop);
            cuda_time_acc += cuda_time;
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //thisHull->print_extremes();
            //thisHull->clearGPU();
            delete thisHull;
        }
    }
    else if (algorithm == 1){
        filter_cub_flagged *thisHull;
        test3(thisHull, x, y, size, filter_cub_flagged);
        delete thisHull;
        cudaDeviceSynchronize();
        for (int i = 0; i < REPEATS; i++){
            cudaEventRecord(start);
            test3(thisHull, x, y, size, filter_cub_flagged);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&cuda_time, start, stop);
            cuda_time_acc += cuda_time;
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //thisHull->print_extremes();
            //thisHull->clearGPU();
            delete thisHull;
        }
    }
    else if (algorithm == 2){
        filter_thrust_scan *thisHull;
        test3(thisHull, x, y, size, filter_thrust_scan);
        delete thisHull;
        cudaDeviceSynchronize();
        for (int i = 0; i < REPEATS; i++){
            cudaEventRecord(start);
            test3(thisHull, x, y, size, filter_thrust_scan);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&cuda_time, start, stop);
            cuda_time_acc += cuda_time;
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //thisHull->print_extremes();
            //thisHull->clearGPU();
            delete thisHull;
        }
    }
    else if (algorithm == 3){
        filter_thrust_copy *thisHull;
        // copy from x and y to array of p type point
        Point *points = new Point[size];
        for (int i = 0; i < size; i++){
            points[i].x = x[i];
            points[i].y = y[i];
        }
        test4(thisHull, points, size, filter_thrust_copy);
        delete thisHull;
        cudaDeviceSynchronize();
        for (int i = 0; i < REPEATS; i++){
            cudaEventRecord(start);
            test4(thisHull, points, size, filter_thrust_copy);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&cuda_time, start, stop);
            cuda_time_acc += cuda_time;
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //thisHull->print_extremes();
            //thisHull->clearGPU();
            delete thisHull;
        }
        delete [] points;
    }


    /*filter *f;
    f = new filter(x, y, size);
    f->gpu_scan();*/

    //printf("Time: %f\n", cuda_time_acc/REPEATS);
    printf("%f %i %i\n",cuda_time_acc/REPEATS, size, 0);

}