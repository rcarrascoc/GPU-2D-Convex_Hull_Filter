 //#include "filter_cpu_parallel.h"
 
filter_cpu_parallel::filter_cpu_parallel(float *x_in, float *y_in, INDEX size){
    x = x_in;
    y = y_in;
    n = size;

    //cpu_manhattan();
}

// find the minimum manhattan distance between a corner and a set of points
// T dist = std::abs(x[i] - xc) + std::abs(y[i] - yc);
template <typename T>
void findCorner_manhattan_parallel(INDEX* corner, const T* x, const T* y, T xc, T yc, INDEX n) {
    T minDistance = std::numeric_limits<T>::max();
    INDEX minIndex = 0;

    #pragma omp parallel
    {
        T localMinDistance = std::numeric_limits<T>::max();
        INDEX localMinIndex = 0;

        #pragma omp for
        for (INDEX i = 0; i < n; ++i) {
            T distance = std::abs(x[i] - xc) + std::abs(y[i] - yc);
            if (distance < localMinDistance) {
                localMinDistance = distance;
                localMinIndex = i;
            }
        }

        #pragma omp critical
        {
            if (localMinDistance < minDistance) {
                minDistance = localMinDistance;
                minIndex = localMinIndex;
            }
        }
    }

    *corner = minIndex;
}

// T distance = std::sqrt(std::pow(x[i] - xc, 2) + std::pow(y[i] - yc, 2));
// T distance = std::pow(x[i] - xc, 2) + std::pow(y[i] - yc, 2);
template <typename T>
void findCorner_euclidean_parallel(INDEX *corner, T *x, T *y, T xc, T yc, INDEX n){
    T minDistance = std::numeric_limits<T>::max();
    INDEX minIndex = 0;

    #pragma omp parallel
    {
        T localMinDistance = std::numeric_limits<T>::max();
        INDEX localMinIndex = 0;

        #pragma omp for
        for (INDEX i = 0; i < n; ++i) {
            T distance = std::pow(x[i] - xc, 2) + std::pow(y[i] - yc, 2);
            if (distance < localMinDistance) {
                localMinDistance = distance;
                localMinIndex = i;
            }
        }

        #pragma omp critical
        {
            if (localMinDistance < minDistance) {
                minDistance = localMinDistance;
                minIndex = localMinIndex;
            }
        }
    }

    *corner = minIndex;
}

void filter_cpu_parallel::cpu_manhattan(){

    // save the time for deleting
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    //printf("CPU MANHATTAN\n");

    // get extreme points
    findMax_parallel<float>(&ri, x, n);
    findMin_parallel<float>(&le, x, n);
    findMax_parallel<float>(&up, y, n);
    findMin_parallel<float>(&lo, y, n);

    xri = x[ri]; yri = y[ri];
    xle = x[le]; yle = y[le];
    xup = x[up]; yup = y[up];
    xlo = x[lo]; ylo = y[lo];


    // save the time for finding axis extreme points
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    t_find_extremes = (milliseconds - t_find_extremes) / step + t_find_extremes;

    // get the time for finding corner points
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    
    // find corners using manhattan distance
    findCorner_manhattan_parallel<float>(&c1, x, y, xri, yup, n);
    findCorner_manhattan_parallel<float>(&c2, x, y, xle, yup, n);
    findCorner_manhattan_parallel<float>(&c3, x, y, xle, ylo, n);
    findCorner_manhattan_parallel<float>(&c4, x, y, xri, ylo, n);
    
    xc1 = x[c1]; yc1 = y[c1];
    xc2 = x[c2]; yc2 = y[c2];
    xc3 = x[c3]; yc3 = y[c3];
    xc4 = x[c4]; yc4 = y[c4];


    // save the time for finding corner points
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    t_find_corners = (milliseconds - t_find_corners) / step + t_find_corners;

    // get the time for finding points in q
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    //print_extremes();

    // compute the slopes
    computeSlopes(); 

    h_q = new INDEX[n];
    INDEX j = 0;
    float h_x, h_y, m;

    #pragma omp parallel for private(h_x, h_y, m)
    for (INDEX i = 0; i < n; i++) {
        h_x = x[i];
        h_y = y[i];

        if (h_x != xri) {
            m = (h_y - yri) / (h_x - xri);

            if (m < mh) {
                if (h_x > xup) {
                    if ((m < m1) || (h_x < xc1 && (h_y - yc1) / (h_x - xc1) < m1b)) {
                        #pragma omp critical
                        {
                            h_q[j] = i;
                            j++;
                        }
                    }
                } else {
                    if (h_x < xup) {
                        m = (h_y - yup) / (h_x - xup);
                        if ((m < m2) || (h_x < xc2 && (h_y - yc2) / (h_x - xc2) < m2b)) {
                            #pragma omp critical
                            {
                                h_q[j] = i;
                                j++;
                            }
                        }
                    }
                }
            } else {
                if (h_x < xlo) {
                    m = (h_y - yle) / (h_x - xle);
                    if ((m < m3) || (h_x > xc3 && (h_y - yc3) / (h_x - xc3) < m3b)) {
                        #pragma omp critical
                        {
                            h_q[j] = i;
                            j++;
                        }
                    }
                } else {
                    if (h_x > xlo) {
                        m = (h_y - ylo) / (h_x - xlo);
                        if ((m < m4) || (h_x > xc4 && (h_y - yc4) / (h_x - xc4) < m4b)) {
                            #pragma omp critical
                            {
                                h_q[j] = i;
                                j++;
                            }
                        }
                    }
                }
            }
        }
    }
    size = j;

    //print_extremes();

    /*#pragma omp parallel for
    for (INDEX i = 0; i < n; i++){
        h_x = x[i];
        h_y = y[i];
        d_qa[i] = 0;
        if(h_x!=xri){
            m=(h_y-yri)/(h_x-xri);									// slope(i, ri1);
            if(m<mh){												// p_i is above horizontal line (le2...ri1)
                if(h_x>xup){
                    if ((m<m1)||(h_x<xc1 && (h_y-yc1)/(h_x-xc1)<m1b)){
                        d_qa[i] = 1;
                    }
                }else{
                    if(h_x<xup){
                        m=(h_y-yup)/(h_x-xup);						// slope(i, up2);
                        if ((m<m2)||(h_x<xc2 && (h_y-yc2)/(h_x-xc2)<m2b)){
                            d_qa[i] = 1;
                        }
                    }
                }
            }else{
                if(h_x<xlo){
                    m=(h_y-yle)/(h_x-xle);							//slope(i, le3);
                    if ((m<m3)||(h_x>xc3 && (h_y-yc3)/(h_x-xc3)<m3b)){	//slope(i, c3);
                        d_qa[i] = 1;
                    }
                }else{
                    if(h_x>xlo){
                        m=(h_y-ylo)/(h_x-xlo);						// slope(i, lo4);
                        if ((m<m4)||(h_x>xc4 && (h_y-yc4)/(h_x-xc4)<m4b)){// slope(i, c4);
                            d_qa[i] = 1;
                        }
                    }
                }
            }
        }
    }

    h_q = new INDEX[n];
    d_size = new INDEX[1];

    //print_extremes();

    compaction_parallel<INDEX,INDEX>(h_q, d_size, d_qa, n);
    size = d_size[0]; // */

    //printf("CPU MANHATTAN END %i\n", size );
    
    
    // save the time for finding points in q
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    t_compaction = (milliseconds - t_compaction) / step + t_compaction;

}

void filter_cpu_parallel::cpu_euclidean(){
    
    // save the time for deleting
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    

    // get extreme points

    findMax_parallel<float>(&ri, x, n);
    findMin_parallel<float>(&le, x, n);
    findMax_parallel<float>(&up, y, n);
    findMin_parallel<float>(&lo, y, n);

    xri = x[ri]; yri = y[ri];
    xle = x[le]; yle = y[le];
    xup = x[up]; yup = y[up];
    xlo = x[lo]; ylo = y[lo];


    // save the time for finding axis extreme points
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    t_find_extremes = (milliseconds - t_find_extremes) / step + t_find_extremes;

    // get the time for finding corner points
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    
    // find corners using manhattan distance
    findCorner_euclidean_parallel<float>(&c1, x, y, xri, yup, n);
    findCorner_euclidean_parallel<float>(&c2, x, y, xle, yup, n);
    findCorner_euclidean_parallel<float>(&c3, x, y, xle, ylo, n);
    findCorner_euclidean_parallel<float>(&c4, x, y, xri, ylo, n);
    
    xc1 = x[c1]; yc1 = y[c1];
    xc2 = x[c2]; yc2 = y[c2];
    xc3 = x[c3]; yc3 = y[c3];
    xc4 = x[c4]; yc4 = y[c4];


    // save the time for finding corner points
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    t_find_corners = (milliseconds - t_find_corners) / step + t_find_corners;

    // get the time for finding points in q
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


   // compute the slopes
    computeSlopes(); 

    h_q = new INDEX[n];
    INDEX j = 0;
    float h_x, h_y, m;

    #pragma omp parallel for private(h_x, h_y, m)
    for (INDEX i = 0; i < n; i++) {
        h_x = x[i];
        h_y = y[i];

        if (h_x != xri) {
            m = (h_y - yri) / (h_x - xri);

            if (m < mh) {
                if (h_x > xup) {
                    if ((m < m1) || (h_x < xc1 && (h_y - yc1) / (h_x - xc1) < m1b)) {
                        #pragma omp critical
                        {
                            h_q[j] = i;
                            j++;
                        }
                    }
                } else {
                    if (h_x < xup) {
                        m = (h_y - yup) / (h_x - xup);
                        if ((m < m2) || (h_x < xc2 && (h_y - yc2) / (h_x - xc2) < m2b)) {
                            #pragma omp critical
                            {
                                h_q[j] = i;
                                j++;
                            }
                        }
                    }
                }
            } else {
                if (h_x < xlo) {
                    m = (h_y - yle) / (h_x - xle);
                    if ((m < m3) || (h_x > xc3 && (h_y - yc3) / (h_x - xc3) < m3b)) {
                        #pragma omp critical
                        {
                            h_q[j] = i;
                            j++;
                        }
                    }
                } else {
                    if (h_x > xlo) {
                        m = (h_y - ylo) / (h_x - xlo);
                        if ((m < m4) || (h_x > xc4 && (h_y - yc4) / (h_x - xc4) < m4b)) {
                            #pragma omp critical
                            {
                                h_q[j] = i;
                                j++;
                            }
                        }
                    }
                }
            }
        }
    }
    size = j;    
    
    
    // save the time for finding points in q
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    t_compaction = (milliseconds - t_compaction) / step + t_compaction;

}

void filter_cpu_parallel::print_extremes(){
    printf("ri: %i, xri: %f, yri: %f\n", ri, xri, yri);
    printf("le: %i, xle: %f, yle: %f\n", le, xle, yle);
    printf("up: %i, xup: %f, yup: %f\n", up, xup, yup);
    printf("lo: %i, xlo: %f, ylo: %f\n", lo, xlo, ylo);
    printf("c1: %i, xc1: %f, yc1: %f\n", c1, xc1, yc1);
    printf("c2: %i, xc2: %f, yc2: %f\n", c2, xc2, yc2);
    printf("c3: %i, xc3: %f, yc3: %f\n", c3, xc3, yc3);
    printf("c4: %i, xc4: %f, yc4: %f\n", c4, xc4, yc4);
    printf("m1: %f, m2: %f, m3: %f, m4: %f\n", m1, m2, m3, m4);
    printf("mh: %f\n", mh);
    
    // print all h_q, x[h_q[i]], y[h_q[i]
    //for (INDEX i = 0; i < size; i++)
    //    printf("%i %f %f\n", h_q[i], x[h_q[i]], y[h_q[i]]);
    
}