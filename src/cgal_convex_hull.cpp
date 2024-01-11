//#include "cgal_convex_hull.h"
#include <array>
#include <vector>
#include <numeric>


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel KK;
typedef KK::Point_2 Point_2;

template <typename HullType, typename INDEX_TYPE>
void convexHull(HullType *my, float *x, float *y, INDEX_TYPE n){
//void ConvexHull::convexHull(filter_cpu_serial *my, float *x, float *y, uint n){
    
    std::vector<Point_2> points;
    std::vector<Point_2> result;

	//points.push_back( Point_2(x[0], y[0]) );

    for (uint i=0; i<my->size; i++){
        points.push_back( Point_2(x[my->h_q[i]],y[my->h_q[i]]));
    }
	points.push_back( Point_2(x[my->ri],y[my->ri]) );
	points.push_back( Point_2(x[my->up],y[my->up]) );
	points.push_back( Point_2(x[my->le],y[my->le]) );
	points.push_back( Point_2(x[my->lo],y[my->lo]) );
	points.push_back( Point_2(x[my->c1],y[my->c1]) );
	points.push_back( Point_2(x[my->c2],y[my->c2]) );
	points.push_back( Point_2(x[my->c3],y[my->c3]) );
	points.push_back( Point_2(x[my->c4],y[my->c4]) );

	// save the time
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CGAL::ch_graham_andrew( points.begin(), points.end(), std::back_inserter(result) );
    my->sizeHull = result.size(); //*/

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	my->t_cgal_convex_hull = (milliseconds - my->t_preparation)/my->step + my->t_preparation;
}

template <typename HullType, typename INDEX_TYPE>
void convexHull(HullType *my, Point *p, INDEX_TYPE n){
//void ConvexHull::convexHull(filter_cpu_serial *my, float *x, float *y, uint n){
    
    std::vector<Point_2> points;
    std::vector<Point_2> result;

	//points.push_back( Point_2(x[0], y[0]) );

    for (uint i=0; i<my->size; i++){
        points.push_back( Point_2(my->h_q[i].x,my->h_q[i].y));
    }
	points.push_back( Point_2(p[my->ri].x,p[my->ri].y) );
	points.push_back( Point_2(p[my->up].x,p[my->up].y) );
	points.push_back( Point_2(p[my->le].x,p[my->le].y) );
	points.push_back( Point_2(p[my->lo].x,p[my->lo].y) );
	points.push_back( Point_2(p[my->c1].x,p[my->c1].y) );
	points.push_back( Point_2(p[my->c2].x,p[my->c2].y) );
	points.push_back( Point_2(p[my->c3].x,p[my->c3].y) );
	points.push_back( Point_2(p[my->c4].x,p[my->c4].y) );

	// save the time
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CGAL::ch_graham_andrew( points.begin(), points.end(), std::back_inserter(result) );
    my->sizeHull = result.size(); //*/

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	my->t_cgal_convex_hull = (milliseconds - my->t_preparation)/my->step + my->t_preparation;
}

template <typename HullType, typename INDEX_TYPE>
void convexHull_2(HullType *my, float *x, float *y, INDEX_TYPE n){
//void ConvexHull::convexHull(filter_cpu_serial *my, float *x, float *y, uint n){
    
	// save the time
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    std::vector<Point_2> points;
    std::vector<Point_2> result;

	//points.push_back( Point_2(x[0], y[0]) );

    for (uint i=0; i<my->size; i++){
        points.push_back( Point_2(x[my->h_q[i]],y[my->h_q[i]]));
    }
	points.push_back( Point_2(x[my->ri],y[my->ri]) );
	points.push_back( Point_2(x[my->up],y[my->up]) );
	points.push_back( Point_2(x[my->le],y[my->le]) );
	points.push_back( Point_2(x[my->lo],y[my->lo]) );
	points.push_back( Point_2(x[my->c1],y[my->c1]) );
	points.push_back( Point_2(x[my->c2],y[my->c2]) );
	points.push_back( Point_2(x[my->c3],y[my->c3]) );
	points.push_back( Point_2(x[my->c4],y[my->c4]) );


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	my->t_preparation = milliseconds;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CGAL::convex_hull_2( points.begin(), points.end(), std::back_inserter(result) );
    my->sizeHull = result.size(); //*/

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	my->t_cgal_convex_hull = (milliseconds - my->t_preparation)/my->step + my->t_preparation;
}

template <typename HullType, typename INDEX_TYPE>
void convexHull_2(HullType *my, Point *p, INDEX_TYPE n){
//void ConvexHull::convexHull(filter_cpu_serial *my, float *x, float *y, uint n){
    
	// save the time
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    std::vector<Point_2> points;
    std::vector<Point_2> result;

	//points.push_back( Point_2(x[0], y[0]) );

    for (uint i=0; i<my->size; i++){
        points.push_back( Point_2(my->h_q[i].x,my->h_q[i].y));
    }
	points.push_back( Point_2(p[my->ri].x,p[my->ri].y) );
	points.push_back( Point_2(p[my->up].x,p[my->up].y) );
	points.push_back( Point_2(p[my->le].x,p[my->le].y) );
	points.push_back( Point_2(p[my->lo].x,p[my->lo].y) );
	points.push_back( Point_2(p[my->c1].x,p[my->c1].y) );
	points.push_back( Point_2(p[my->c2].x,p[my->c2].y) );
	points.push_back( Point_2(p[my->c3].x,p[my->c3].y) );
	points.push_back( Point_2(p[my->c4].x,p[my->c4].y) );


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	my->t_preparation = milliseconds;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CGAL::convex_hull_2( points.begin(), points.end(), std::back_inserter(result) );
    my->sizeHull = result.size(); //*/

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	my->t_cgal_convex_hull = (milliseconds - my->t_preparation)/my->step + my->t_preparation;
}

template <typename INDEX_TYPE>
float cgal_2(std::vector<Point_2> *output, std::vector<Point_2> points, INDEX_TYPE n, float mean_time, int step){
	
	// save the time
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
	
	std::vector<Point_2> result;
    CGAL::convex_hull_2( points.begin(), points.end(), std::back_inserter(result) );
	*output = result;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	//my->t_cgal_convex_hull = milliseconds;

	return (milliseconds - mean_time) / step + mean_time;
}

template <typename INDEX_TYPE>
float cgal(std::vector<Point_2> *output, std::vector<Point_2> points, INDEX_TYPE n, float mean_time, int step){
	
	// save the time
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

	std::vector<Point_2> result;
    CGAL::ch_graham_andrew( points.begin(), points.end(), std::back_inserter(result) );
	*output = result;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	//my->t_cgal_convex_hull = milliseconds;

	return (milliseconds - mean_time) / step + mean_time;
}