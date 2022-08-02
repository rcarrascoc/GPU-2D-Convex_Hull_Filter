// find the minimum manhattan distance between a points and a corner of the grid
//template <typename T>
__global__ void findMin_kernel_manhattan(INDEX *out, float *d_dist, float *d_x, float *d_y, float xc, float yc, INDEX n){
    INDEX offset = blockIdx.x * blockDim.x + threadIdx.x;
    if(offset < n){
        // compute the manhattan distance
        float x = d_x[offset];
        float y = d_y[offset];
        float dx = x - xc;
        float dy = y - yc;
        float dist = (fabsf(dx) + fabsf(dy));
        d_dist[offset] =  dist;
        INDEX val = offset;

        // find the minimum
        val = min_index_block<float>(val, d_dist); 
		//__syncthreads();
		if(threadIdx.x == 0)
			min_index_atomic<float>(out, val, d_dist);
    }
}

__global__ void findMin_kernel_manhattan(INDEX *out, float *d_dist, float *d_x, float *d_y, INDEX *xc, INDEX *yc, INDEX n){
    INDEX offset = blockIdx.x * blockDim.x + threadIdx.x;
    if(offset < n){
        // compute the manhattan distance
        float x = d_x[offset];
        float y = d_y[offset];
        float dx = x - d_x[*xc];
        float dy = y - d_y[*yc];
        float dist = (fabsf(dx) + fabsf(dy));
        d_dist[offset] =  dist;
        INDEX val = offset;

        // find the minimum
        val = min_index_block<float>(val, d_dist); 
		//__syncthreads();
		if(threadIdx.x == 0)
			min_index_atomic<float>(out, val, d_dist);
    }
}

__global__ void gpu_compute_slopes(
        float *m1, float *m2, float *m3, float *m4, float *m1b, float *m2b, float *m3b, float *m4b, float *mh,
        INDEX *ri, INDEX *le, INDEX *lo, INDEX *up, INDEX *c1, INDEX *c2, INDEX *c3, INDEX *c4,
        float *d_x, float *d_y, INDEX n){

    float xri = d_x[*ri], yri = d_y[*ri];
    float xle = d_x[*le], yle = d_y[*le];
    float xlo = d_x[*lo], ylo = d_y[*lo];
    float xup = d_x[*up], yup = d_y[*up];
    float xc1 = d_x[*c1], yc1 = d_y[*c1];
    float xc2 = d_x[*c2], yc2 = d_y[*c2];
    float xc3 = d_x[*c3], yc3 = d_y[*c3];
    float xc4 = d_x[*c4], yc4 = d_y[*c4];

    /*// print coordinate of the points
    printf("-> %i %f %f\n", *ri, xri, yri);
    printf("-> %i %f %f\n", *le, xle, yle);
    printf("-> %i %f %f\n", *lo, xlo, ylo);
    printf("-> %i %f %f\n", *up, xup, yup);
    printf("-> %i %f %f\n", *c1, xc1, yc1);
    printf("-> %i %f %f\n", *c2, xc2, yc2);
    printf("-> %i %f %f\n", *c3, xc3, yc3);
    printf("-> %i %f %f\n", *c4, xc4, yc4);//*/

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

template <typename T>
__global__ void kernel_inPointsInQ(   float *X, float *Y, INDEX n,
                                        float xri, float yri, float xup, float yup, float xle, float yle, float xlo, float ylo,
                                        float xc1, float yc1, float xc2, float yc2, float xc3, float yc3, float xc4, float yc4,
                                        float m1, float m2, float m3, float m4, float mh, float m1b, float m2b, float m3b, float m4b, 
                                        T *Q, INDEX *qa   ){
                                
	INDEX i = blockIdx.x * blockDim.x + threadIdx.x; 
	Q[i] = 0.0f;
	float m, x, y;
	if(i < n){
		qa[i] = (INDEX)i;
		x=X[i];y=Y[i];
		if(x!=xri){
			m=(y-yri)/(x-xri);									// slope(i, ri1);
			if(m<mh){												// p_i is above horizontal line (le2...ri1)
				if(x>xup){
					if ((m<m1)||(x<xc1 && (y-yc1)/(x-xc1)<m1b)){
						Q[i] = 1.0f;
					}
				}else{
					if(x<xup){
						m=(y-yup)/(x-xup);						// slope(i, up2);
						if ((m<m2)||(x<xc2 && (y-yc2)/(x-xc2)<m2b)){
							Q[i] = 1.0f;
						}
					}
				}
			}else{
				if(x<xlo){
					m=(y-yle)/(x-xle);							//slope(i, le3);
					if ((m<m3)||(x>xc3 && (y-yc3)/(x-xc3)<m3b)){	//slope(i, c3);
						Q[i] = 1.0f;
					}
				}else{
					if(x>xlo){
						m=(y-ylo)/(x-xlo);						// slope(i, lo4);
						if ((m<m4)||(x>xc4 && (y-yc4)/(x-xc4)<m4b)){// slope(i, c4);
                            Q[i] = 1.0f;
						}
					}
				}
			}
		}
	}
}

