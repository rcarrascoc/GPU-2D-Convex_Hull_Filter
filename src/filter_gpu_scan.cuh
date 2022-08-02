// class filter
class filter_gpu_scan : public filter{
public:
    filter_gpu_scan(float *x_in, float *y_in, INDEX size);

    INDEX *d_ri, *d_le, *d_lo, *d_up;
    INDEX *d_c1, *d_c2, *d_c3, *d_c4;
    float *d_c;

    // slope device variables
    float *d_m1, *d_m2, *d_m3, *d_m4, *d_m1b, *d_m2b, *d_m3b, *d_m4b, *d_mh;

    // function declarerations
    void gpu_scan();
    void f_gpu_scan();
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


