//#include "filter_cub_flagged.cuh"
#include "kernel/kernel_cub_flagged.cu"


// key value pair for extremes of the polygon
cub::KeyValuePair<int, float> *d_ri, *d_le, *d_lo, *d_up;
cub::KeyValuePair<int, float>  *d_c1, *d_c2, *d_c3, *d_c4;

filter_cub_flagged::filter_cub_flagged(float *x_in, float *y_in, INDEX size){
    x = x_in;
    y = y_in;
    n = size;
    cub_flagged();
    //f_cub_flagged();

    // cuda copy d_q to host
    h_q = new INDEX[size];
    cudaMemcpy(h_q, d_q, sizeof(INDEX) * n, cudaMemcpyDeviceToHost); kernelCallCheck();
    //print_extremes();
}

void filter_cub_flagged::cub_flagged(){
    // cuda malloc for x and y arrays
    cudaMalloc(&d_x, sizeof(float) * n);
    cudaMalloc(&d_y, sizeof(float) * n);
    // cuda mem copy for x and y arrays
    cudaMemcpy(d_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * n, cudaMemcpyHostToDevice);

    cudaMalloc(&d_ri, sizeof(cub::KeyValuePair<INDEX, float>));
    cudaMalloc(&d_le, sizeof(cub::KeyValuePair<INDEX, float>));
    cudaMalloc(&d_lo, sizeof(cub::KeyValuePair<INDEX, float>));
    cudaMalloc(&d_up, sizeof(cub::KeyValuePair<INDEX, float>));

    // find axis extreme points
    findMax_cub<float>(d_ri, d_x, n);
    findMax_cub<float>(d_up, d_y, n);
    findMin_cub<float>(d_le, d_x, n);
    findMin_cub<float>(d_lo, d_y, n);

    // copy from d_ri to host
    cudaMemcpy(&ri, d_ri, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&up, d_up, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&le, d_le, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lo, d_lo, sizeof(int), cudaMemcpyDeviceToHost);

    xri = x[ri]; yri = y[ri];
    xle = x[le]; yle = y[le];
    xlo = x[lo]; ylo = y[lo];
    xup = x[up]; yup = y[up];

    // index corner points
    cudaMalloc(&d_c1, sizeof(cub::KeyValuePair<int, float>));
    cudaMalloc(&d_c2, sizeof(cub::KeyValuePair<int, float>));
    cudaMalloc(&d_c3, sizeof(cub::KeyValuePair<int, float>));
    cudaMalloc(&d_c4, sizeof(cub::KeyValuePair<int, float>));

    cudaMalloc(&d_c, sizeof(float) * n);

    // compute the manhattan distance and find the minimum
    // corner 1
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, xri, yup, n);
    cudaDeviceSynchronize();
    findMin_cub<float>(d_c1, d_c, n);

    // corner 2
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, xle, yup, n);
    cudaDeviceSynchronize();
    findMin_cub<float>(d_c2, d_c, n);

    // corner 3
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, xle, ylo, n);
    cudaDeviceSynchronize();
    findMin_cub<float>(d_c3, d_c, n);

    // corner 4
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, xri, ylo, n);
    cudaDeviceSynchronize();
    findMin_cub<float>(d_c4, d_c, n);

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
    
    cudaMalloc(&d_vec_inQ, sizeof(char) * n);
    cudaMalloc(&d_qa, sizeof(INDEX) * n);
    cudaMalloc(&d_q, sizeof(INDEX) * n);
    cudaMalloc(&d_size, sizeof(INDEX));

    // is in polygon?
    kernel_inPointsInQ<char><<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_x, d_y, n, 
                xri, yri, xup, yup, xle, yle, xlo, ylo,
                xc1, yc1, xc2, yc2, xc3, yc3, xc4, yc4,
                m1, m2, m3, m4, mh, m1b, m2b, m3b, m4b, 
                d_vec_inQ, d_qa ); kernelCallCheck();
    cudaDeviceSynchronize();

    compaction_cub<INDEX,char>(d_q, d_size, d_vec_inQ, d_qa, n);

    cudaMemcpy(&size, d_size, sizeof(INDEX), cudaMemcpyDeviceToHost);

}

void filter_cub_flagged::f_cub_flagged(){
    // cuda malloc for x and y arrays
    cudaMalloc(&d_x, sizeof(float) * n);
    cudaMalloc(&d_y, sizeof(float) * n);
    // cuda mem copy for x and y arrays
    cudaMemcpy(d_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * n, cudaMemcpyHostToDevice);

    cudaMalloc(&d_ri, sizeof(cub::KeyValuePair<int, float>));
    cudaMalloc(&d_le, sizeof(cub::KeyValuePair<int, float>));
    cudaMalloc(&d_lo, sizeof(cub::KeyValuePair<int, float>));
    cudaMalloc(&d_up, sizeof(cub::KeyValuePair<int, float>));

    // find axis extreme points
    findMax_cub<float>(d_ri, d_x, n);
    findMax_cub<float>(d_up, d_y, n);
    findMin_cub<float>(d_le, d_x, n);
    findMin_cub<float>(d_lo, d_y, n);


    // index corner points
    cudaMalloc(&d_c1, sizeof(cub::KeyValuePair<int, float>));
    cudaMalloc(&d_c2, sizeof(cub::KeyValuePair<int, float>));
    cudaMalloc(&d_c3, sizeof(cub::KeyValuePair<int, float>));
    cudaMalloc(&d_c4, sizeof(cub::KeyValuePair<int, float>));

    cudaMalloc(&d_c, sizeof(float) * n);

    // compute the manhattan distance and find the minimum
    // corner 1
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, d_ri, d_up, n);
    cudaDeviceSynchronize();
    findMin_cub<float>(d_c1, d_c, n);

    // corner 2
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, d_le, d_up, n);
    cudaDeviceSynchronize();
    findMin_cub<float>(d_c2, d_c, n);

    // corner 3
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, d_le, d_lo, n);
    cudaDeviceSynchronize();
    findMin_cub<float>(d_c3, d_c, n);

    // corner 4
    compute_manhattan<<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_c, d_x, d_y, d_ri, d_lo, n);
    cudaDeviceSynchronize();
    findMin_cub<float>(d_c4, d_c, n);

    // slope device variables
    float *d_m1, *d_m2, *d_m3, *d_m4, *d_m1b, *d_m2b, *d_m3b, *d_m4b, *d_mh;
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
                                d_ri, d_le, d_lo, d_up, d_ri, d_le, d_lo, d_up,
                                d_x, d_y, n);

    cudaDeviceSynchronize();

    cudaMalloc(&d_vec_inQ, sizeof(char) * n);
    cudaMalloc(&d_qa, sizeof(INDEX) * n);
    cudaMalloc(&d_q, sizeof(INDEX) * n);
    cudaMalloc(&d_size, sizeof(INDEX));

    // is in polygon?
    kernel_inPointsInQ<char><<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_x, d_y, n, 
                d_ri, d_up, d_le, d_lo, d_c1, d_c2, d_c3, d_c4,
                d_m1, d_m2, d_m3, d_m4, d_m1b, d_m2b, d_m3b, d_m4b, d_mh, 
                d_vec_inQ, d_qa ); kernelCallCheck();
    cudaDeviceSynchronize();

    compaction_cub<INDEX,char>(d_q, d_size, d_vec_inQ, d_qa, n);

    cudaMemcpy(&size, d_size, sizeof(INDEX), cudaMemcpyDeviceToHost);
}





// copy the device variables to the host variables
void filter_cub_flagged::copy_to_host(){
   
   cudaMalloc(&d_out_x, sizeof(float) * n);
   cudaMalloc(&d_out_y, sizeof(float) * n);
   get_coor<char><<<(n+BSIZE-1)/BSIZE,BSIZE>>>(d_out_x, d_out_y, d_vec_inQ, d_q, d_x, d_y, d_size, n); kernelCallCheck();
   cudaDeviceSynchronize();

   // output malloc
   cudaMalloc(&out_x, sizeof(float) * n);
   cudaMalloc(&out_y, sizeof(float) * n);

   // copy output to host
   cudaMemcpy(out_x, d_out_x, sizeof(float) * n, cudaMemcpyDeviceToHost);
   cudaMemcpy(out_y, d_out_y, sizeof(float) * n, cudaMemcpyDeviceToHost);//

}

// print indices and cooordinates of all axis extreme points, corners, and slopes
void filter_cub_flagged::print_extremes(){
    // copy from device to host
    cudaMemcpy(&ri, d_ri, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&up, d_up, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&le, d_le, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lo, d_lo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c1, d_c1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c2, d_c2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c3, d_c3, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c4, d_c4, sizeof(int), cudaMemcpyDeviceToHost);

    xri = x[ri]; yri = y[ri];
    xle = x[le]; yle = y[le];
    xlo = x[lo]; ylo = y[lo];
    xup = x[up]; yup = y[up];
    xc1 = x[c1]; yc1 = y[c1];
    xc2 = x[c2]; yc2 = y[c2];
    xc3 = x[c3]; yc3 = y[c3];
    xc4 = x[c4]; yc4 = y[c4];

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

    // print x[q[i]] and y[q[i]] for all i in [0, size]
    for(int i = 0; i < size; i++)
        printf("-> %i: %f, %f\n", h_q[i], x[h_q[i]], y[h_q[i]]); //*/
}

void filter_cub_flagged::delete_filter(){
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