// determine if a point is inside a polygon
template <typename T>
__global__ void kernel_inPointsInQ(   float *d_x, float *d_y, INDEX n,
                                        INDEX *ri, INDEX *up, INDEX *le, INDEX *lo, INDEX *c1, INDEX *c2, INDEX *c3, INDEX *c4,
                                        float *m1, float *m2, float *m3, float *m4, float *m1b, float *m2b, float *m3b, float *m4b, float *mh,
                                        T *Q, INDEX *qa   ){
                                
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
    float m, x, y;
    if(i < n){
        Q[i] = 0;
        float xri = d_x[*ri], yri = d_y[*ri];
        float xle = d_x[*le], yle = d_y[*le];
        float xlo = d_x[*lo], ylo = d_y[*lo];
        float xup = d_x[*up], yup = d_y[*up];
        float xc1 = d_x[*c1], yc1 = d_y[*c1];
        float xc2 = d_x[*c2], yc2 = d_y[*c2];
        float xc3 = d_x[*c3], yc3 = d_y[*c3];
        float xc4 = d_x[*c4], yc4 = d_y[*c4];

        qa[i] = i;
        x=d_x[i];y=d_y[i];
        if(x!=xri){
            m=(y-yri)/(x-xri);									// slope(i, ri1);
            if(m<*mh){												// p_i is above horizontal line (le2...ri1)
                if(x>xup){
                    if ((m<*m1)||(x<xc1 && (y-yc1)/(x-xc1)<*m1b)){
                        Q[i] = 1;
                    }
                }else{
                    if(x<xup){
                        m=(y-yup)/(x-xup);						// slope(i, up2);
                        if ((m<*m2)||(x<xc2 && (y-yc2)/(x-xc2)<*m2b)){
                            Q[i] = 1;
                        }
                    }
                }
            }else{
                if(x<xlo){
                    m=(y-yle)/(x-xle);							//slope(i, le3);
                    if ((m<*m3)||(x>xc3 && (y-yc3)/(x-xc3)<*m3b)){	//slope(i, c3);
                        Q[i] = 1;
                    }
                }else{
                    if(x>xlo){
                        m=(y-ylo)/(x-xlo);						// slope(i, lo4);
                        if ((m<*m4)||(x>xc4 && (y-yc4)/(x-xc4)<*m4b)){// slope(i, c4);
                            Q[i] = 1;
                        }
                    }
                }
            }
        }
    }
}


// determine the coor x and y of the points inside the polygon
template <typename T>
__global__ void get_coor(float *out_x, float *out_y, T *vec_inQ, INDEX *q, float *in_x, float *in_y, INDEX *d_size, INDEX n){
    INDEX i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < n){
        if((int)vec_inQ[i] == 1){
            out_x[q[i]] = in_x[i];
            out_y[q[i]] = in_y[i];
        }
    }
}