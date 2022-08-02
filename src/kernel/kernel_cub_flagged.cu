__global__ void compute_manhattan(float *d_dist, float *d_x, float *d_y, 
            cub::KeyValuePair<int, float> *xc, cub::KeyValuePair<int, float> *yc, INDEX n){
    INDEX offset = blockIdx.x * blockDim.x + threadIdx.x;
    if(offset < n){
        // compute the manhattan distance
        float x = d_x[offset];
        float y = d_y[offset];
        float dx = x - xc[0].value;
        float dy = y - yc[0].value;
        float dist = (fabsf(dx) + fabsf(dy));
        d_dist[offset] =  dist;
    }
}



__global__ void gpu_compute_slopes(
        float *m1, float *m2, float *m3, float *m4, float *m1b, float *m2b, float *m3b, float *m4b, float *mh,
        cub::KeyValuePair<int, float> *ri,  cub::KeyValuePair<int, float> *le, 
        cub::KeyValuePair<int, float> *lo,  cub::KeyValuePair<int, float> *up, 
        cub::KeyValuePair<int, float> *c1,  cub::KeyValuePair<int, float> *c2,
        cub::KeyValuePair<int, float> *c3,  cub::KeyValuePair<int, float> *c4,
        float *d_x, float *d_y, int n){

    float xri = d_x[ri[0].key], yri = d_y[ri[0].key];
    float xle = d_x[le[0].key], yle = d_y[le[0].key];
    float xlo = d_x[lo[0].key], ylo = d_y[lo[0].key];
    float xup = d_x[up[0].key], yup = d_y[up[0].key];
    float xc1 = d_x[c1[0].key], yc1 = d_y[c1[0].key];
    float xc2 = d_x[c2[0].key], yc2 = d_y[c2[0].key];
    float xc3 = d_x[c3[0].key], yc3 = d_y[c3[0].key];
    float xc4 = d_x[c4[0].key], yc4 = d_y[c4[0].key];

    if (xri!=xup){
        if (xc1>xup && yc1>yri){
            *m1 = (yri-yc1)/(xri-xc1); 		//slope(ri1, c1);
            *m1b = (yri-yup)/(xri-xup);		//slope(ri1, up1);
            if (*m1 < *m1b){
                *m1b = (yc1-yup)/(xc1-xup); 	//slope(c1, up1);
            }else{
                *m1 = *m1b;
                *m1b = 0;
                xc1 = yc1 = -1*FLT_MAX;
            }
        }else{
            *m1 = (yri-yup)/(xri-xup);		//slope(ri1, up1);
            *m1b = 0;
            xc1 = yc1 = -1*FLT_MAX;
        }
    }else{
        *m1 = *m1b = 0;
        xc1 = yc1 = -1*FLT_MAX;
    }

    if (xup!=xle){
        if (xc2<xup && yc2>yle){
            *m2 = (yup-yc2)/(xup-xc2);			//slope(up2, c2);
            *m2b = (yup-yle)/(xup-xle);		//slope(up2, le2);
            if (*m2 < *m2b){
                *m2b = (yc2-yle)/(xc2-xle);	//slope(c2, le2);
            }else{
                *m2 = *m2b;
                *m2b = 0;
                xc2 = yc2 = -1*FLT_MAX;
            }
        }else{
            *m2 = (yup-yle)/(xup-xle);		//slope(up2, le2);
            *m2b = 0;
            xc2 = yc2 = -1*FLT_MAX;
        }
    }else{
        *m2 = *m2b = 0;
        xc2 = yc2 = -1*FLT_MAX;
    }

    if (xle!=xlo){
        if (xc3<xlo && yc3<yle){
            *m3 = (yle-yc3)/(xle-xc3);			//slope(le3, c3);
            *m3b = (yle-ylo)/(xle-xlo);		//slope(le3, lo3);
            if (*m3 < *m3b){
                *m3b = (ylo-yc3)/(xlo-xc3);	//slope(c3, lo3);
            }else{
                *m3 = *m3b;
                *m3b = 0;
                xc3 = yc3 = FLT_MAX;
            }
        }else{
            *m3 = (yle-ylo)/(xle-xlo);		//slope(le3, lo3);
            *m3b = 0;
            xc3 = yc3 = FLT_MAX;
        }
    }else{
        *m3 = *m3b = 0;
        xc3 = yc3 = FLT_MAX;
    }

    if (xlo!=xri){
        if (xc4>xlo && yc4<yri){
            *m4 = (ylo-yc4)/(xlo-xc4);			//slope(lo4, c4);
            *m4b = (ylo-yri)/(xlo-xri);		//slope(lo4, ri4);
            if (*m4 < *m4b){
                *m4b = (yc4-yri)/(xc4-xri);	//slope(c4, ri4);
            }else{
                *m4 = *m4b;
                *m4b = 0;
                xc4 = yc4 = FLT_MAX;
            }
        }else{
            *m4 = (ylo-yri)/(xlo-xri);		//slope(lo4, ri4);
            *m4b = 0;
            xc4 = yc4 = FLT_MAX;
        }
    }else{
        *m4 = *m4b = 0;
        xc4 = yc4 = FLT_MAX;
    }

    *mh = 0;
    if (xri!=xle)
        *mh = (yri-yle)/(xri-xle);	 		//slope(le2, ri1);
}


// determine if a point is inside a polygon
template <typename T>
__global__ void kernel_inPointsInQ(   float *d_x, float *d_y, int n,
                                        cub::KeyValuePair<int, float> *ri, cub::KeyValuePair<int, float> *up, 
                                        cub::KeyValuePair<int, float> *le, cub::KeyValuePair<int, float> *lo, 
                                        cub::KeyValuePair<int, float> *c1, cub::KeyValuePair<int, float> *c2, 
                                        cub::KeyValuePair<int, float> *c3, cub::KeyValuePair<int, float> *c4,
                                        float *m1, float *m2, float *m3, float *m4, float *m1b, float *m2b, float *m3b, float *m4b, float *mh,
                                        T *Q, INDEX *qa   ){
                                
    INDEX i = blockIdx.x * blockDim.x + threadIdx.x; 
    //Q[i] = 0;
    float m, x, y;
    if(i < n){
        float xri = d_x[ri[0].key], yri = d_y[ri[0].key];
        float xle = d_x[le[0].key], yle = d_y[le[0].key];
        float xlo = d_x[lo[0].key], ylo = d_y[lo[0].key];
        float xup = d_x[up[0].key], yup = d_y[up[0].key];
        float xc1 = d_x[c1[0].key], yc1 = d_y[c1[0].key];
        float xc2 = d_x[c2[0].key], yc2 = d_y[c2[0].key];
        float xc3 = d_x[c3[0].key], yc3 = d_y[c3[0].key];
        float xc4 = d_x[c4[0].key], yc4 = d_y[c4[0].key];

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
