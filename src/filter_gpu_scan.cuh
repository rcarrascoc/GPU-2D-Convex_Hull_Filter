#include "kernel/kernel_gpu_scan.cu"

// class filter
class filter_gpu_scan : public filter{
public:
    filter_gpu_scan(float *x_in, float *y_in, INDEX size){
        x = x_in;
        y = y_in;
        n = size;
        gpu_scan();
        //f_gpu_scan();

        // cuda copy d_q to host
        h_q = new INDEX[size];
        cudaMemcpy(h_q, d_q, sizeof(INDEX) * n, cudaMemcpyDeviceToHost);
        //print_extremes();
    }

    INDEX *d_ri, *d_le, *d_lo, *d_up;
    INDEX *d_c1, *d_c2, *d_c3, *d_c4;
    float *d_c;

    // slope device variables
    float *d_m1, *d_m2, *d_m3, *d_m4, *d_m1b, *d_m2b, *d_m3b, *d_m4b, *d_mh;

    // compacting vector
    half *d_vec_inQ;


    void gpu_scan(){
        // cuda malloc for x and y arrays
        cudaMalloc(&d_x, sizeof(float) * n);
        cudaMalloc(&d_y, sizeof(float) * n);
        
        // cuda mem copy for x and y arrays
        cudaMemcpy(d_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, sizeof(float) * n, cudaMemcpyHostToDevice);

        cudaMalloc(&d_ri, sizeof(INDEX));
        cudaMalloc(&d_le, sizeof(INDEX));
        cudaMalloc(&d_lo, sizeof(INDEX));
        cudaMalloc(&d_up, sizeof(INDEX));

        // find axis extreme points
        findMax_kernel<float>(d_ri, d_x, n); kernelCallCheck();
        findMax_kernel<float>(d_up, d_y, n); kernelCallCheck();
        findMin_kernel<float>(d_le, d_x, n); kernelCallCheck();
        findMin_kernel<float>(d_lo, d_y, n); kernelCallCheck();

        // copy from d_ri to host
        cudaMemcpy(&ri, d_ri, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&up, d_up, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&le, d_le, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lo, d_lo, sizeof(INDEX), cudaMemcpyDeviceToHost);

        xri = x[ri]; yri = y[ri];
        xle = x[le]; yle = y[le];
        xlo = x[lo]; ylo = y[lo];
        xup = x[up]; yup = y[up];

        cudaMalloc(&d_c1, sizeof(INDEX));
        cudaMalloc(&d_c2, sizeof(INDEX));
        cudaMalloc(&d_c3, sizeof(INDEX));
        cudaMalloc(&d_c4, sizeof(INDEX));

        cudaMalloc(&d_c, sizeof(float) * n);

        // find corner points
        findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c1, d_c, d_x, d_y, xri, yup, n); kernelCallCheck();
        findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c2, d_c, d_x, d_y, xle, yup, n); kernelCallCheck();
        findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c3, d_c, d_x, d_y, xle, ylo, n); kernelCallCheck();
        findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c4, d_c, d_x, d_y, xri, ylo, n); kernelCallCheck();

        // copy from d_c1 to host
        cudaMemcpy(&c1, d_c1, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&c2, d_c2, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&c3, d_c3, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&c4, d_c4, sizeof(INDEX), cudaMemcpyDeviceToHost);

        xc1 = x[c1]; yc1 = y[c1];
        xc2 = x[c2]; yc2 = y[c2];
        xc3 = x[c3]; yc3 = y[c3];
        xc4 = x[c4]; yc4 = y[c4];
        
        computeSlopes();
    
        cudaMalloc(&d_vec_inQ, sizeof(half) * n); kernelCallCheck();
        cudaMalloc(&d_qa, sizeof(INDEX) * n); kernelCallCheck();
        cudaMalloc(&d_q, sizeof(INDEX) * n); kernelCallCheck();
        cudaMalloc(&d_size, sizeof(INDEX)); kernelCallCheck();

        // is in polygon?
        kernel_inPointsInQ<half><<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_x, d_y, n, 
                    xri, yri, xup, yup, xle, yle, xlo, ylo,
                    xc1, yc1, xc2, yc2, xc3, yc3, xc4, yc4,
                    m1, m2, m3, m4, mh, m1b, m2b, m3b, m4b, 
                    d_vec_inQ, d_q ); kernelCallCheck();
        cudaDeviceSynchronize();

        compaction_tc_scan<INDEX,half>(d_q, d_size, d_vec_inQ, d_qa, n);

        cudaMemcpy(&size, d_size, sizeof(INDEX), cudaMemcpyDeviceToHost);

    }

    

    void f_gpu_scan(){
        // cuda malloc for x and y arrays
        cudaMalloc(&d_x, sizeof(float) * n);
        cudaMalloc(&d_y, sizeof(float) * n);

        // cuda mem copy for x and y arrays
        cudaMemcpy(d_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, sizeof(float) * n, cudaMemcpyHostToDevice);

        cudaMalloc(&d_ri, sizeof(INDEX));
        cudaMalloc(&d_le, sizeof(INDEX));
        cudaMalloc(&d_lo, sizeof(INDEX));
        cudaMalloc(&d_up, sizeof(INDEX));

        // find axis extreme points
        findMax_kernel<float>(d_ri, d_x, n); kernelCallCheck();
        findMax_kernel<float>(d_up, d_y, n); kernelCallCheck();
        findMin_kernel<float>(d_le, d_x, n); kernelCallCheck();
        findMin_kernel<float>(d_lo, d_y, n); kernelCallCheck();


        cudaMalloc(&d_c1, sizeof(INDEX));
        cudaMalloc(&d_c2, sizeof(INDEX));
        cudaMalloc(&d_c3, sizeof(INDEX));
        cudaMalloc(&d_c4, sizeof(INDEX));

        cudaMalloc(&d_c, sizeof(float) * n);

        // find corner points
        findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c1, d_c, d_x, d_y, d_ri, d_up, n); kernelCallCheck();
        findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c2, d_c, d_x, d_y, d_le, d_up, n); kernelCallCheck();
        findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c3, d_c, d_x, d_y, d_le, d_lo, n); kernelCallCheck();
        findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c4, d_c, d_x, d_y, d_ri, d_lo, n); kernelCallCheck();

        cudaMalloc(&d_m1, sizeof(float));
        cudaMalloc(&d_m2, sizeof(float));
        cudaMalloc(&d_m3, sizeof(float));
        cudaMalloc(&d_m4, sizeof(float));
        cudaMalloc(&d_m1b, sizeof(float));
        cudaMalloc(&d_m2b, sizeof(float));
        cudaMalloc(&d_m3b, sizeof(float));
        cudaMalloc(&d_m4b, sizeof(float));
        cudaMalloc(&d_mh, sizeof(float));

        // compute slopes
        gpu_compute_slopes<<<1,1>>>(d_m1, d_m2, d_m3, d_m4, d_m1b, d_m2b, d_m3b, d_m4b, d_mh,
                                    d_ri, d_le, d_lo, d_up, d_c1, d_c2, d_c3, d_c4,
                                    d_x, d_y, n); kernelCallCheck();
        cudaDeviceSynchronize();
  
        cudaMalloc(&d_vec_inQ, sizeof(half) * n);
        cudaMalloc(&d_qa, sizeof(INDEX) * n);
        cudaMalloc(&d_q, sizeof(INDEX) * n);
        cudaMalloc(&d_size, sizeof(INDEX));

        // is in polygon?
        kernel_inPointsInQ<half><<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_x, d_y, n, 
                    d_ri, d_up, d_le, d_lo, d_c1, d_c2, d_c3, d_c4,
                    d_m1, d_m2, d_m3, d_m4, d_m1b, d_m2b, d_m3b, d_m4b, d_mh, 
                    d_vec_inQ, d_q ); kernelCallCheck();
        cudaDeviceSynchronize();

        compaction_tc_scan<INDEX,half>(d_q, d_size, d_vec_inQ, d_qa, n);

        cudaMemcpy(&size, d_size, sizeof(INDEX), cudaMemcpyDeviceToHost);

    }

    


   // copy the device variables to the host variables
   template <typename T>
   void copy_to_host(){
       
       cudaMalloc(&d_out_x, sizeof(float) * n);
       cudaMalloc(&d_out_y, sizeof(float) * n);
       get_coor<T><<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_out_x, d_out_y, d_vec_inQ, d_q, d_x, d_y, d_size, n); kernelCallCheck();
       cudaDeviceSynchronize();

       // output malloc
       cudaMalloc(&out_x, sizeof(float) * n);
       cudaMalloc(&out_y, sizeof(float) * n);

       // copy output to host
       cudaMemcpy(out_x, d_out_x, sizeof(float) * n, cudaMemcpyDeviceToHost);
       cudaMemcpy(out_y, d_out_y, sizeof(float) * n, cudaMemcpyDeviceToHost);//

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
        cudaFree(d_ri);
        cudaFree(d_le);
        cudaFree(d_lo);
        cudaFree(d_up);
        cudaFree(d_c1);
        cudaFree(d_c2);
        cudaFree(d_c3);
        cudaFree(d_c4);
        cudaFree(d_c);
       /* cudaFree(d_m1);
        cudaFree(d_m2);
        cudaFree(d_m3);
        cudaFree(d_m4);
        cudaFree(d_m1b);
        cudaFree(d_m2b);
        cudaFree(d_m3b);
        cudaFree(d_m4b);
        cudaFree(d_mh); // */
        cudaFree(d_vec_inQ);
        cudaFree(d_q);
        cudaFree(d_qa);
        cudaFree(d_size);

        cudaDeviceSynchronize();
        kernelCallCheck();
    }

    // print indices and cooordinates of all axis extreme points, corners, and slopes
    void print_extremes(){
        // copy from device to host
        cudaMemcpy(&ri, d_ri, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&up, d_up, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&le, d_le, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lo, d_lo, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&c1, d_c1, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&c2, d_c2, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&c3, d_c3, sizeof(INDEX), cudaMemcpyDeviceToHost);
        cudaMemcpy(&c4, d_c4, sizeof(INDEX), cudaMemcpyDeviceToHost);

        xri = x[ri]; yri = y[ri];
        xle = x[le]; yle = y[le];
        xlo = x[lo]; ylo = y[lo];
        xup = x[up]; yup = y[up];
        xc1 = x[c1]; yc1 = y[c1];
        xc2 = x[c2]; yc2 = y[c2];
        xc3 = x[c3]; yc3 = y[c3];
        xc4 = x[c4]; yc4 = y[c4];

        printf("\n");
        printf("ri: %i, xri: %f, yri: %f\n", (INDEX)ri, xri, yri);
        printf("up: %i, xup: %f, yup: %f\n", (INDEX)up, xup, yup);
        printf("le: %i, xle: %f, yle: %f\n", (INDEX)le, xle, yle);
        printf("lo: %i, xlo: %f, ylo: %f\n", (INDEX)lo, xlo, ylo);
        printf("c1: %i, xc1: %f, yc1: %f\n", (INDEX)c1, xc1, yc1);
        printf("c2: %i, xc2: %f, yc2: %f\n", (INDEX)c2, xc2, yc2);
        printf("c3: %i, xc3: %f, yc3: %f\n", (INDEX)c3, xc3, yc3);
        printf("c4: %i, xc4: %f, yc4: %f\n", (INDEX)c4, xc4, yc4);
        //printf("m1: %f, m2: %f, m3: %f, m4: %f, mh: %f\n", m1, m2, m3, m4, mh);
        //printf("m1b: %f, m2b: %f, m3b: %f, m4b: %f\n", m1b, m2b, m3b, m4b);
        printf("\n");

        printf("compacted size: %d\n", size);
    }

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


