// class filter
class filter_cub_flagged : public filter{
public:
    filter_cub_flagged(float *x_in, float *y_in, INDEX size);

    // compacting vector
    char *d_vec_inQ;
    float *d_c;

    // declare functions
    void cub_flagged();
    void f_cub_flagged();
    void print_extremes();
    void copy_to_host();
    void delete_filter();
};

/*
        std::cout << "ri: " << ri << " ( " << xri << ", " << yri << ")" << std::endl;
        std::cout << "le: " << le << " ( " << xle << ", " << yle << ")" << std::endl;
        std::cout << "lo: " << lo << " ( " << xlo << ", " << ylo << ")" << std::endl;
        std::cout << "up: " << up << " ( " << xup << ", " << yup << ")" << std::endl;

        std::cout << "c1: " << c1 << " ( " << xc1 << ", " << yc1 << ")" << std::endl;
        std::cout << "c2: " << c2 << " ( " << xc2 << ", " << yc2 << ")" << std::endl;
        std::cout << "c3: " << c3 << " ( " << xc3 << ", " << yc3 << ")" << std::endl;
        std::cout << "c4: " << c4 << " ( " << xc4 << ", " << yc4 << ")" << std::endl;
*/


