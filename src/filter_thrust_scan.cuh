#include "kernel/kernel_thrust_scan.cu"

// class filter
class filter_thrust_scan : public filter{
public:
    filter_thrust_scan(float *x_in, float *y_in, INDEX size){
        x = x_in;
        y = y_in;
        n = size;
        thrust_scan();

        h_q = new INDEX[size];
        cudaMemcpy(h_q, d_q, sizeof(INDEX) * n, cudaMemcpyDeviceToHost);
        //print_extremes();
    }

    // compacting vector
    char *d_vec_inQ;

    void thrust_scan(){
        // cuda malloc for x and y arrays
        cudaMalloc(&d_x, sizeof(float) * n);
        cudaMalloc(&d_y, sizeof(float) * n);
        
        // cuda mem copy for x and y arrays
        cudaMemcpy(d_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, sizeof(float) * n, cudaMemcpyHostToDevice);

        // cuda malloc for d_min and d_max
        thrust::device_ptr<float> x_ptr = thrust::device_pointer_cast(d_x);
        thrust::device_ptr<float> y_ptr = thrust::device_pointer_cast(d_y);
    
        // get min and max
        thrust::device_vector<float>::iterator x_max = thrust::max_element(x_ptr, x_ptr + n);
        ri = x_max - (thrust::device_vector<float>::iterator)x_ptr;
        thrust::device_vector<float>::iterator y_max = thrust::max_element(y_ptr, y_ptr + n);
        up = y_max - (thrust::device_vector<float>::iterator)y_ptr;
        thrust::device_vector<float>::iterator x_min = thrust::min_element(x_ptr, x_ptr + n);
        le = x_min - (thrust::device_vector<float>::iterator)x_ptr;
        thrust::device_vector<float>::iterator y_min = thrust::min_element(y_ptr, y_ptr + n);
        lo = y_min - (thrust::device_vector<float>::iterator)y_ptr;

        // copy from device to host
        float d_xri = x[ri]; float d_yri = y[ri];
        float d_xup = x[up]; float d_yup = y[up];
        float d_xle = x[le]; float d_yle = y[le];
        float d_xlo = x[lo]; float d_ylo = y[lo];
        xri = d_xri; yri = d_yri;
        xup = d_xup; yup = d_yup;
        xle = d_xle; yle = d_yle;
        xlo = d_xlo; ylo = d_ylo;

        // get corners
        thrust::device_vector<float> c_ptr(n,0);
        thrust::transform(	x_ptr, 
                            x_ptr + n, 
                            y_ptr,  
                            c_ptr.begin(), 
                            [=] __host__ __device__ (float t_x, float t_y) { return fabsf(t_x - d_xri) + fabsf(t_y - d_yup); }      // --- Lambda expression 
                        ); 
        thrust::device_vector<float>::iterator c1_it = thrust::min_element( c_ptr.begin(), c_ptr.end());
        c1 = c1_it - c_ptr.begin();
        xc1 = x_ptr[c1];
        yc1 = y_ptr[c1];

        thrust::transform(	x_ptr, 
						x_ptr + n, 
						y_ptr,  
						c_ptr.begin(), 
						[=] __host__ __device__ (float t_x, float t_y) { return fabsf(t_x - d_xle) + fabsf(t_y - d_yup); }      // --- Lambda expression 
                     ); 
        thrust::device_vector<float>::iterator c2_it = thrust::min_element( c_ptr.begin(), c_ptr.end());
        c2 = c2_it - c_ptr.begin();
        xc2 = x_ptr[c2];
        yc2 = y_ptr[c2];

        thrust::transform(	x_ptr, 
                            x_ptr + n, 
                            y_ptr,  
                            c_ptr.begin(), 
                            [=] __host__ __device__ (float t_x, float t_y) { return fabsf(t_x - d_xle) + fabsf(t_y - d_ylo); }      // --- Lambda expression 
                        ); 
        thrust::device_vector<float>::iterator c3_it = thrust::min_element( c_ptr.begin(), c_ptr.end());
        c3 = c3_it - c_ptr.begin();
        xc3 = x_ptr[c3];
        yc3 = y_ptr[c3];

        thrust::transform(	x_ptr, 
                            x_ptr + n, 
                            y_ptr,  
                            c_ptr.begin(), 
                            [=] __host__ __device__ (float t_x, float t_y) { return fabsf(t_x - d_xri) + fabsf(t_y - d_ylo); }      // --- Lambda expression 
                        ); 
        thrust::device_vector<float>::iterator c4_it = thrust::min_element( c_ptr.begin(), c_ptr.end());
        c4 = c4_it - c_ptr.begin();
        xc4 = x_ptr[c4];
        yc4 = y_ptr[c4]; 

        // compute the slope of the line
        computeSlopes(); 

        // cuda malloc for compacting vector
        cudaMalloc(&d_vec_inQ, sizeof(char) * n);
        cudaMalloc(&d_qa, sizeof(INDEX) * n);
        cudaMalloc(&d_q, sizeof(INDEX) * n);
        cudaMalloc(&d_size, sizeof(INDEX));

        // is in polygon?
        kernel_inPointsInQ<char><<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_x, d_y, n, 
                    xri, yri, xup, yup, xle, yle, xlo, ylo,
                    xc1, yc1, xc2, yc2, xc3, yc3, xc4, yc4,
                    m1, m2, m3, m4, mh, m1b, m2b, m3b, m4b, 
                    d_vec_inQ, d_q ); kernelCallCheck();
        cudaDeviceSynchronize();

        compaction_scan_thrust<INDEX,char>(d_q, d_size, d_vec_inQ, d_qa, n);
        
        cudaMemcpy(&size, d_size, sizeof(INDEX), cudaMemcpyDeviceToHost);

    }


    // print indices and cooordinates of all axis extreme points, corners, and slopes
    void print_extremes(){

        printf("\n");
        printf("ri: %i, xri: %f, yri: %f\n", ri, xri, yri);
        printf("up: %i, xup: %f, yup: %f\n", up, xup, yup);
        printf("le: %i, xle: %f, yle: %f\n", le, xle, yle);
        printf("lo: %i, xlo: %f, ylo: %f\n", lo, xlo, ylo);
        printf("c1: %i, xc1: %f, yc1: %f\n", c1, xc1, yc1);
        printf("c2: %i, xc2: %f, yc2: %f\n", c2, xc2, yc2);
        printf("c3: %i, xc3: %f, yc3: %f\n", c3, xc3, yc3);
        printf("c4: %i, xc4: %f, yc4: %f\n", c4, xc4, yc4);
        //printf("m1: %f, m2: %f, m3: %f, m4: %f, mh: %f\n", m1, m2, m3, m4, mh);
        //printf("m1b: %f, m2b: %f, m3b: %f, m4b: %f\n", m1b, m2b, m3b, m4b);
        printf("\n");

        printf("compacted size: %d\n", size);
    }


    void delete_filter(){
        // delete host variables
        //delete[] x;
        //delete[] y;
        delete h_q;

        cudaDeviceSynchronize();
        kernelCallCheck();

        // delete device variables
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_vec_inQ);
        cudaFree(d_q);
        cudaFree(d_qa);
        cudaFree(d_size);

        cudaDeviceSynchronize();
        kernelCallCheck();
    }
};