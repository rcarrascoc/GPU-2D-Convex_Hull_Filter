//#include "kernel/kernel_thrust_copy.cu"
typedef struct {
    float x, y;
} Point;

#include <thrust/host_vector.h>

// class filter
class filter_thrust_copy : public filter{
public:
    filter_thrust_copy(Point *h_p, INDEX size);

    // compacting vector
    Point *p;
    Point *d_p;
    thrust::host_vector<Point> h_q;

    // function declarations
    void thrust_copy();
    void print_extremes();
    void delete_filter();

};