// class filter
class filter_thrust_scan : public filter{
public:
    filter_thrust_scan(float *x_in, float *y_in, INDEX size);

    // compacting vector
    char *d_vec_inQ;

    // function declarations
    void thrust_scan();
    void print_extremes();
    void delete_filter();
};