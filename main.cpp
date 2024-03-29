#include <stdio.h>
#include <string>
#include <cfloat>

// define global variables
#define REAL float // define the real type, really not implemented yet
//#define INDEX uint // index type
#define REPEATS 5 // number of repetitions of the benchmark

// include the header file of the library
// include filter.cuh
#include "src/filter.cuh"

// include tool functions, generated and printed by the tool
#include "src/tools.cpp"

// include cgal functions
#include "src/cgal_convex_hull.cpp"


using namespace std;

#define test(ffx, ffy, ffn, ffalg) ffalg(ffx,ffy,ffn)
#define test2(ffthis, ffx, ffy, ffn, ffalg) ffalg *ffthis = new ffalg(ffx,ffy,ffn)
#define test3(ffthis, ffx, ffy, ffn, ffalg) ffthis = new ffalg(ffx,ffy,ffn)
#define test4(ffthis, ffp, ffn, ffalg) ffthis = new ffalg(ffp,ffn)

// function to save in a file all the points and candidate to hull in diffrent .off files, name_point.off and name_candidate.off respectively
void save_off_file(filter_cpu_serial *filter, std::string name){
    std::ofstream file_point;
    std::ofstream file_candidate;
    file_point.open(name+"_point.off");
    file_candidate.open(name+"_candidate.off");
    file_point << "OFF\n";
    file_candidate << "OFF\n";
    file_point << filter->n << " 0 0\n";
    file_candidate << filter->sizeHull << " 0 0\n";
    for(int i=0; i<filter->n; i++){
        file_point << filter->x[i] << " " << filter->y[i] << "\n";
    }
    for(int i=0; i<filter->sizeHull; i++){
        file_candidate << filter->x[filter->h_q[i]] << " " << filter->y[filter->h_q[i]] << "\n";
    }
    file_point.close();
    file_candidate.close();
}

// save timer in a json file
void save_timer_2(std::string filename, int n, int size, float t_total, unsigned long seed){
    std::ofstream file;
    file.open(filename);
    file << "{\n";
    file << "\"n\": " << n << ",\n";
    file << "\"size\": " << 0 << ",\n";
    file << "\"hull\": " << size << ",\n";
    file << "\"seed\": " << seed << ",\n";
    file << "\"copy2device\": " << 0 << ",\n";
    file << "\"find_extremes\": " << 0 << ",\n";
    file << "\"find_corners\": " << 0 << ",\n";
    file << "\"find_points_in_Q\": " << 0 << ",\n";
    file << "\"compaction\": " << 0 << ",\n";
    file << "\"copy2host\": " << 0 << ",\n";
    file << "\"delete\": " << 0 << ",\n";
    file << "\"preparation\": " << 0 << ",\n";
    file << "\"cgal_convex_hull\": " << t_total << ",\n";
    file << "\"total\": " << t_total << "\n";
    file << "}";
    file.close();
}



// include benchmark functions
//#include "src/benchmark.cu"

string arr_alg[10] = {"cpu_manhattan", "cpu_euclidean", "gpu_scan", "cub_flagger", "thrust_scan", "thrust_copy","convex_hull_2","andrew_graham", "omp_manhattan", "omp_euclidean"};
string arr_shape[3] = {"normal", "uniform", "circumference"};

// main function
// where read four arguments and call the corresponding function
// read: size of the array, algorithm to use, shape of distribution, probability of the distribution
int main(int argc, char *argv[]) {
    // check if the number of arguments is correct
    if (argc != 6) {
        printf("Error: wrong number of arguments\n");
        printf("Usage: ./main size algorithm shape probability\n");
        printf("size: number of points\n");
        printf("algorithms: 0 kernel, 1 thrust-scan, 2 thrust-copy, 3 cub-flagged \n");
        printf("shape: 0 for uniform, 1 for normal, 2 for circumference\n");
        printf("probability: [0..1] probability of the distribution\n");
        printf("path: path to save the data\n");
        return 1;
    }
    
    // read the arguments
    INDEX size = atoi(argv[1]);
    int algorithm = atoi(argv[2]);
    int shape = atoi(argv[3]);
    double probability = atof(argv[4]);

    // get path from the arguments
    std::string path = argv[5];
    
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
    INDEX filtered_size = 0, hull_size = 0;
    float mean_time = 0;
    unsigned long seed = time(NULL );

    // call the corresponding function
    // for generating x and y arrays
    switch (shape) {
        case 0:
            //generate_random_points_normal<REAL>(x, y, size);
            generate_random_normal_points_omp(size, x, y, seed);
            break;
        case 1:
            //generate_random_points_uniform<REAL>(x, y, size);
            generate_random_uniform_points_omp(size, x, y, seed);
            break;
        case 2:
            //generate_random_points_circumference<REAL>(x, y, size, probability);
            generate_random_circumference_points_omp(size, x, y, probability, seed);
            break;
        default:
            printf("Error: algorithm must be 0, 1 or 2\n");
            return 1;
    }

    // print the points
    //print_points(x, y, size);

    // start the timer
	cuda_time_m *time = new cuda_time_m();
	time->init();

    //test2(thisHull, x, y, size, filter_gpu_scan);

    if (algorithm == 0) {
        filter_cpu_serial *thisHull;
        for (int i = 0; i < REPEATS; i++){
	        time->start();
            test3(thisHull, x, y, size, filter_cpu_serial);
            thisHull->cpu_manhattan();
            convexHull_2<filter_cpu_serial,INDEX>(thisHull, x, y, size);
	        time->pause();
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //save_off_file(thisHull, "cpu_manhattan");
            filtered_size = thisHull->size;
            hull_size = thisHull->sizeHull;
            if (i == REPEATS-1) thisHull->save_timer(path +"/" + arr_alg[algorithm] + "_" + arr_shape[shape]+ "_" + std::to_string(size) + "_" + std::to_string(seed) + ".json");
            //thisHull->delete_filter();
            delete thisHull;
        }
    }
    if (algorithm == 1) {
        filter_cpu_serial *thisHull;
        for (int i = 0; i < REPEATS; i++){
            time->start();
            test3(thisHull, x, y, size, filter_cpu_serial);
            thisHull->cpu_euclidean();
            convexHull_2<filter_cpu_serial,INDEX>(thisHull, x, y, size);
	        time->pause();
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //thisHull->print_extremes();
            filtered_size = thisHull->size;
            hull_size = thisHull->sizeHull;
            if (i == REPEATS-1) thisHull->save_timer(path +"/" + arr_alg[algorithm] + "_" + arr_shape[shape]+ "_" + std::to_string(size) + "_" + std::to_string(seed) + ".json");
            //thisHull->delete_filter();
            delete thisHull;
        }
   }
    else if (algorithm == 2){
        filter_gpu_scan *thisHull;
        test3(thisHull, x, y, size, filter_gpu_scan);
        thisHull->delete_filter();
        delete thisHull;
        for (int i = 0; i < REPEATS; i++){
            time->start();
            test3(thisHull, x, y, size, filter_gpu_scan);
            convexHull_2<filter_gpu_scan,INDEX>(thisHull, x, y, size);
	        time->pause();
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //thisHull->print_extremes();
            filtered_size = thisHull->size;
            hull_size = thisHull->sizeHull;
            if (i == REPEATS-1) thisHull->save_timer(path +"/" + arr_alg[algorithm] + "_" + arr_shape[shape]+ "_" + std::to_string(size) + "_" + std::to_string(seed) + ".json");
            thisHull->delete_filter();
            delete thisHull;
        }
    }
    else if (algorithm == 3){
        filter_cub_flagged *thisHull;
        test3(thisHull, x, y, size, filter_cub_flagged);
        thisHull->delete_filter();
        delete thisHull;
        for (int i = 0; i < REPEATS; i++){
            time->start();
            test3(thisHull, x, y, size, filter_cub_flagged);
            convexHull_2<filter_cub_flagged,INDEX>(thisHull, x, y, size);
	        time->pause();
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //thisHull->print_extremes();
            filtered_size = thisHull->size;
            hull_size = thisHull->sizeHull;
            if (i == REPEATS-1) thisHull->save_timer(path +"/" + arr_alg[algorithm] + "_" + arr_shape[shape]+ "_" + std::to_string(size) + "_" + std::to_string(seed) + ".json");
            thisHull->delete_filter();
            delete thisHull;
        }
    }
    else if (algorithm == 4){
        filter_thrust_scan *thisHull;
        test3(thisHull, x, y, size, filter_thrust_scan);
        thisHull->delete_filter();
        delete thisHull;
        for (int i = 0; i < REPEATS; i++){
            time->start();
            test3(thisHull, x, y, size, filter_thrust_scan);
            convexHull_2<filter_thrust_scan,INDEX>(thisHull, x, y, size);
	        time->pause();
            filtered_size = thisHull->size;
            hull_size = thisHull->sizeHull;
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //thisHull->print_extremes();
            if (i == REPEATS-1) thisHull->save_timer(path +"/" + arr_alg[algorithm] + "_" + arr_shape[shape]+ "_" + std::to_string(size) + "_" + std::to_string(seed) + ".json");
            thisHull->delete_filter();
            delete thisHull;
        }
    }
    else if (algorithm == 5){
        filter_thrust_copy *thisHull;
        // copy from x and y to array of p type point
        Point *points = new Point[size];
        for (INDEX i = 0; i < size; i++){
            points[i].x = x[i];
            points[i].y = y[i];
        }
        test4(thisHull, points, size, filter_thrust_copy);
        thisHull->delete_filter();
        delete thisHull;
        for (int i = 0; i < REPEATS; i++){
            time->start();
            test4(thisHull, points, size, filter_thrust_copy);
            convexHull_2<filter_thrust_copy,INDEX>(thisHull, points, size);
	        time->pause();
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //thisHull->print_extremes();
            filtered_size = thisHull->size;
            hull_size = thisHull->sizeHull;
            if (i == REPEATS-1) thisHull->save_timer(path +"/" + arr_alg[algorithm] + "_" + arr_shape[shape]+ "_" + std::to_string(size) + "_" + std::to_string(seed) + ".json");
            thisHull->delete_filter();
            delete thisHull;
        }
        delete [] points;
    }
    else if (algorithm == 6){
        std::vector<Point_2> points;
        std::vector<Point_2> result;
        for (INDEX i = 0; i < size; i++){
            points.push_back( Point_2(x[i],y[i]));
        }
        for (int i = 0; i < REPEATS; i++){
            time->start();
            mean_time = cgal_2<INDEX>(&result, points, size, mean_time, i+1);
            time->pause();
            hull_size = result.size();
            // delete result amd points
            result.clear();
        }
        points.clear();
        save_timer_2(path +"/" + arr_alg[algorithm] + "_" + arr_shape[shape]+ "_" + std::to_string(size) + "_" + std::to_string(seed) + ".json", size, hull_size, mean_time, 0);
    }
    else if (algorithm == 7){
        std::vector<Point_2> points;
        std::vector<Point_2> result;
        for (INDEX i = 0; i < size; i++){
            points.push_back( Point_2(x[i],y[i]));
        }
        for (int i = 0; i < REPEATS; i++){
            time->start();
            mean_time = cgal<INDEX>(&result, points, size, mean_time, i+1);
            time->pause();
            hull_size = result.size();
            // delete result amd points
            result.clear();
        }
        points.clear();
        save_timer_2(path +"/" + arr_alg[algorithm] + "_" + arr_shape[shape]+ "_" + std::to_string(size) + "_" + std::to_string(seed) + ".json", size, hull_size, mean_time, 0);
    }
    else if (algorithm == 8) {
        filter_cpu_parallel *thisHull;
        for (int i = 0; i < REPEATS; i++){
	        time->start();
            test3(thisHull, x, y, size, filter_cpu_parallel);
            thisHull->cpu_manhattan();
            convexHull_2<filter_cpu_parallel,INDEX>(thisHull, x, y, size);
	        time->pause();
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //thisHull->print_extremes();
            filtered_size = thisHull->size;
            hull_size = thisHull->sizeHull;
            if (i == REPEATS-1) thisHull->save_timer(path +"/" + arr_alg[algorithm] + "_" + arr_shape[shape]+ "_" + std::to_string(size) + "_" + std::to_string(seed) + ".json");
            //thisHull->delete_filter();
            delete thisHull;
        }
    }
    else if (algorithm == 9) {
        filter_cpu_parallel *thisHull;
        for (int i = 0; i < REPEATS; i++){
	        time->start();
            test3(thisHull, x, y, size, filter_cpu_parallel);
            thisHull->cpu_euclidean();
            convexHull_2<filter_cpu_parallel,INDEX>(thisHull, x, y, size);
	        time->pause();
            //std::cout << "size after the filter: " << thisHull->size << std::endl;
            //thisHull->print_extremes();
            filtered_size = thisHull->size;
            hull_size = thisHull->sizeHull;
            if (i == REPEATS-1) thisHull->save_timer(path +"/" + arr_alg[algorithm] + "_" + arr_shape[shape]+ "_" + std::to_string(size) + "_" + std::to_string(seed) + ".json");
            //thisHull->delete_filter();
            delete thisHull;
        }
    }

    //benchmark(cuda_time_acc, x, y, algorithm, size);

    //printf("Time: %f\n", cuda_time_acc/REPEATS);
    printf("%f %i %i\n",(float)time->get_time()/REPEATS, filtered_size, hull_size);

    // free memory
    delete x;
    delete y;

}
