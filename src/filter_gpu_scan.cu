//#include "filter_gpu_scan.cuh"
#include "kernel/kernel_gpu_scan.cu"

// compacting vector
half *d_vec_inQ;

filter_gpu_scan::filter_gpu_scan(float *x_in, float *y_in, INDEX siz2e){
    x = x_in;
    y = y_in;
    n = size2;
    gpu_scan();
    //f_gpu_scan();

    // save the time for deleting
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // cuda copy d_q to host
    h_q = new INDEX[size];
    cudaMemcpy(h_q, d_q, sizeof(INDEX) * size, cudaMemcpyDeviceToHost); kernelCallCheck();
    //print_extremes();

    // save the time for copying to host
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    t_copy2host = (milliseconds - t_copy2host) / step + t_copy2host;
}

void filter_gpu_scan::gpu_scan(){
    // GET CUDA TIME
    cudaDeviceSynchronize();
    step++;
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

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

    // save the time for copying to device
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    t_copy2device = (milliseconds - t_copy2device) / step + t_copy2device;

    cudaEvent_t start_filter, stop_filter;
    cudaEventCreate(&start_filter);
    cudaEventCreate(&stop_filter);
    cudaEventRecord(start_filter);

    // get the time for finding axis extreme points
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


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

    cudaMalloc(&d_c1, sizeof(INDEX));
    cudaMalloc(&d_c2, sizeof(INDEX));
    cudaMalloc(&d_c3, sizeof(INDEX));
    cudaMalloc(&d_c4, sizeof(INDEX));

    cudaMalloc(&d_c, sizeof(float) * n);
    
    // compute the manhattan distance and find the minimum
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, xri, yup, n);
    cudaDeviceSynchronize();
    findMin_kernel<float>(d_c1, d_c, n);
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, xle, yup, n);
    cudaDeviceSynchronize();
    findMin_kernel<float>(d_c2, d_c, n);
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, xle, ylo, n);
    cudaDeviceSynchronize();
    findMin_kernel<float>(d_c3, d_c, n);
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, xri, ylo, n);
    cudaDeviceSynchronize();
    findMin_kernel<float>(d_c4, d_c, n);

    /*// find corner points
    findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c1, d_c, d_x, d_y, xri, yup, n); kernelCallCheck();
    findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c2, d_c, d_x, d_y, xle, yup, n); kernelCallCheck();
    findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c3, d_c, d_x, d_y, xle, ylo, n); kernelCallCheck();
    findMin_kernel_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c4, d_c, d_x, d_y, xri, ylo, n); kernelCallCheck();*/

    // copy from d_c1 to host
    cudaMemcpy(&c1, d_c1, sizeof(INDEX), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c2, d_c2, sizeof(INDEX), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c3, d_c3, sizeof(INDEX), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c4, d_c4, sizeof(INDEX), cudaMemcpyDeviceToHost);

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
                d_vec_inQ, d_qa ); kernelCallCheck();
    cudaDeviceSynchronize();

    /*// copy d_vec_inQ to host and print all element of h_vec_inQ
    half *h_vec_inQ = new half[n];
    cudaMemcpy(h_vec_inQ, d_vec_inQ, sizeof(half) * n, cudaMemcpyDeviceToHost); kernelCallCheck();
    INDEX *h_qa = new INDEX[n];
    cudaMemcpy(h_qa, d_qa, sizeof(INDEX) * n, cudaMemcpyDeviceToHost); kernelCallCheck();
    for (int i = 0; i < n; i++){
        if ((int)h_vec_inQ[i] == 1){
            printf("%d %i\n", i, (int)h_qa[i]);
        }
        //printf("%i %i\n", i, (int) h_vec_inQ[i]);
    } //*/

    // save the time for finding points in q
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    t_find_points_in_Q = (milliseconds - t_find_points_in_Q) / step + t_find_points_in_Q;

    // get the time for compacting 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    compaction_tc_scan<INDEX,half>(d_q, d_size, d_vec_inQ, d_qa, n);
    //cudaDeviceSynchronize();
    cudaMemcpy(&size, d_size, sizeof(INDEX), cudaMemcpyDeviceToHost);

    // save the time for compacting
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);    
    t_compaction = (milliseconds - t_compaction) / step + t_compaction;

    /*// copy d_qa to host and print all element
    INDEX *h_qa = new INDEX[n];
    cudaMemcpy(h_qa, d_q, sizeof(INDEX) * n, cudaMemcpyDeviceToHost); kernelCallCheck();
    for (int i = 0; i < size; i++){
        printf("%i %i\n", i,(int) h_qa[i]);
    }*/

    // save the time for compacting
    cudaEventRecord(stop_filter);
    cudaEventSynchronize(stop_filter);
    cudaEventElapsedTime(&milliseconds, start_filter, stop_filter);    
    t_total = (milliseconds - t_total) / step + t_total;
}



void filter_gpu_scan::f_gpu_scan(){
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
                d_vec_inQ, d_qa ); kernelCallCheck();
    cudaDeviceSynchronize();

    compaction_tc_scan<INDEX,half>(d_q, d_size, d_vec_inQ, d_qa, n);

    cudaMemcpy(&size, d_size, sizeof(INDEX), cudaMemcpyDeviceToHost);

}




// copy the device variables to the host variables
void filter_gpu_scan::copy_to_host(){
   
   cudaMalloc(&d_out_x, sizeof(float) * n);
   cudaMalloc(&d_out_y, sizeof(float) * n);
   get_coor<half><<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_out_x, d_out_y, d_vec_inQ, d_q, d_x, d_y, d_size, n); kernelCallCheck();
   cudaDeviceSynchronize();

   // output malloc
   cudaMalloc(&out_x, sizeof(float) * n);
   cudaMalloc(&out_y, sizeof(float) * n);

   // copy output to host
   cudaMemcpy(out_x, d_out_x, sizeof(float) * n, cudaMemcpyDeviceToHost);
   cudaMemcpy(out_y, d_out_y, sizeof(float) * n, cudaMemcpyDeviceToHost);//

}


void filter_gpu_scan::delete_filter(){
    // get the time for deleting
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


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

    // save the time for deleting
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    t_delete = (milliseconds - t_delete) / step;
}

// print indices and cooordinates of all axis extreme points, corners, and slopes
void filter_gpu_scan::print_extremes(){
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

    // print x[q[i]] and y[q[i]] for all i in [0, size]
    for(int i = 0; i < size; i++)
        printf("-> %i: %f, %f\n", h_q[i], x[h_q[i]], y[h_q[i]]); //*/
}