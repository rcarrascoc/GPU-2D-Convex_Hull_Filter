//#include "filter_thrust_copy.cuh"

filter_thrust_copy::filter_thrust_copy(Point *h_p, INDEX size){
    p = h_p;
    n = size;
    thrust_copy();

    //print_extremes();
}

void filter_thrust_copy::thrust_copy(){
    auto ffx = [=] __host__ __device__ (Point lhs, Point rhs) { return lhs.x < rhs.x; };
    auto ffy = [=] __host__ __device__ (Point lhs, Point rhs) { return lhs.y < rhs.y; };

    cudaMalloc(&d_p, sizeof(Point)*n);
    cudaMemcpy(d_p, p, sizeof(Point)*n, cudaMemcpyHostToDevice);

    thrust::device_ptr<Point> p_ptr = thrust::device_pointer_cast(d_p);

    thrust::device_vector<Point>::iterator x_max = thrust::max_element(p_ptr, p_ptr + n, ffx);
    ri = x_max - (thrust::device_vector<Point>::iterator)p_ptr;
    thrust::device_vector<Point>::iterator y_max = thrust::max_element(p_ptr, p_ptr + n, ffy);
    up = y_max - (thrust::device_vector<Point>::iterator)p_ptr;
    thrust::device_vector<Point>::iterator x_min = thrust::min_element(p_ptr, p_ptr + n, ffx);
    le = x_min - (thrust::device_vector<Point>::iterator)p_ptr;
    thrust::device_vector<Point>::iterator y_min = thrust::min_element(p_ptr, p_ptr + n, ffy);
    lo = y_min - (thrust::device_vector<Point>::iterator)p_ptr;


    float d_xri = p[ri].x; float d_yri = p[ri].y;
    float d_xup = p[up].x; float d_yup = p[up].y;
    float d_xle = p[le].x; float d_yle = p[le].y;
    float d_xlo = p[lo].x; float d_ylo = p[lo].y;
    xri = d_xri; yri = d_yri; xup = d_xup; yup = d_yup; xle = d_xle; yle = d_yle; xlo = d_xlo; ylo = d_ylo;

    thrust::device_vector<float> c_ptr(n,0);
    thrust::transform(	p_ptr, 
                        p_ptr + n, 
                        c_ptr.begin(), 
                        [=] __host__ __device__ (Point t_p) { return fabsf(t_p.x - d_xri) + fabsf(t_p.y - d_yup); }      // --- Lambda expression 
                        );
    thrust::device_vector<float>::iterator c1_it = thrust::min_element( c_ptr.begin(), c_ptr.end());
    c1 = c1_it - c_ptr.begin();
    float d_xc1 = p[c1].x;
    float d_yc1 = p[c1].y;
    xc1 = d_xc1; yc1 = d_yc1;

    thrust::transform(	p_ptr, 
                    p_ptr + n, 
                    c_ptr.begin(), 
                    [=] __host__ __device__ (Point t_p) { return fabsf(t_p.x - d_xle) + fabsf(t_p.y - d_yup); }      // --- Lambda expression 
                    ); 
    thrust::device_vector<float>::iterator c2_it = thrust::min_element( c_ptr.begin(), c_ptr.end());
    c2 = c2_it - c_ptr.begin();
    float d_xc2 = p[c2].x;
    float d_yc2 = p[c2].y;
    xc2 = d_xc2; yc2 = d_yc2;

    thrust::transform(	p_ptr, 
                        p_ptr + n, 
                        c_ptr.begin(), 
                        [=] __host__ __device__ (Point t_p) { return fabsf(t_p.x - d_xle) + fabsf(t_p.y - d_ylo); }      // --- Lambda expression 
                        );
    thrust::device_vector<float>::iterator c3_it = thrust::min_element( c_ptr.begin(), c_ptr.end());
    c3 = c3_it - c_ptr.begin();
    float d_xc3 = p[c3].x;
    float d_yc3 = p[c3].y;
    xc3 = d_xc3; yc3 = d_yc3;

    thrust::transform(	p_ptr, 
                        p_ptr + n, 
                        c_ptr.begin(), 
                        [=] __host__ __device__ (Point t_p) { return fabsf(t_p.x - d_xri) + fabsf(t_p.y - d_ylo); }      // --- Lambda expression 
                        ); 
    thrust::device_vector<float>::iterator c4_it = thrust::min_element( c_ptr.begin(), c_ptr.end());
    c4 = c4_it - c_ptr.begin();
    float d_xc4 = p[c4].x;
    float d_yc4 = p[c4].y;
    xc4 = d_xc4; yc4 = d_yc4;

    float m1, m2, m3, m4, mh, m1b, m2b, m3b, m4b;

    if (d_xri!=d_xup){
        if (d_xc1>d_xup && d_yc1>d_yri){
            m1 = (d_yri-d_yc1)/(d_xri-d_xc1); 		//slope(ri1, c1);
            m1b = (d_yri-d_yup)/(d_xri-d_xup);		//slope(ri1, up1);
            if (m1 < m1b){
                m1b = (d_yc1-d_yup)/(d_xc1-d_xup); 	//slope(c1, up1);
            }else{
                m1 = m1b;
                m1b = 0;
                d_xc1 = d_yc1 = -1*FLT_MAX;
            }
        }else{
            m1 = (d_yri-d_yup)/(d_xri-d_xup);		//slope(ri1, up1);
            m1b = 0;
            d_xc1 = d_yc1 = -1*FLT_MAX;
        }
    }else{
        m1 = m1b = 0;
        d_xc1 = d_yc1 = -1*FLT_MAX;
    }

    if (d_xup!=d_xle){
        if (d_xc2<d_xup && d_yc2>d_yle){
            m2 = (d_yup-d_yc2)/(d_xup-d_xc2);			//slope(up2, c2);
            m2b = (d_yup-d_yle)/(d_xup-d_xle);		//slope(up2, le2);
            if (m2 < m2b){
                m2b = (d_yc2-d_yle)/(d_xc2-d_xle);	//slope(c2, le2);
            }else{
                m2 = m2b;
                m2b = 0;
                d_xc2 = d_yc2 = -1*FLT_MAX;
            }
        }else{
            m2 = (d_yup-d_yle)/(d_xup-d_xle);		//slope(up2, le2);
            m2b = 0;
            d_xc2 = d_yc2 = -1*FLT_MAX;
        }
    }else{
        m2 = m2b = 0;
        d_xc2 = d_yc2 = -1*FLT_MAX;
    }

    if (d_xle!=d_xlo){
        if (d_xc3<d_xlo && d_yc3<d_yle){
            m3 = (d_yle-d_yc3)/(d_xle-d_xc3);			//slope(le3, c3);
            m3b = (d_yle-d_ylo)/(d_xle-d_xlo);		//slope(le3, lo3);
            if (m3 < m3b){
                m3b = (d_ylo-d_yc3)/(d_xlo-d_xc3);	//slope(c3, lo3);
            }else{
                m3 = m3b;
                m3b = 0;
                d_xc3 = d_yc3 = FLT_MAX;
            }
        }else{
            m3 = (d_yle-d_ylo)/(d_xle-d_xlo);		//slope(le3, lo3);
            m3b = 0;
            d_xc3 = d_yc3 = FLT_MAX;
        }
    }else{
        m3 = m3b = 0;
        d_xc3 = d_yc3 = FLT_MAX;
    }

    if (d_xlo!=d_xri){
        if (d_xc4>d_xlo && d_yc4<d_yri){
            m4 = (d_ylo-d_yc4)/(d_xlo-d_xc4);			//slope(lo4, c4);
            m4b = (d_ylo-d_yri)/(d_xlo-d_xri);		//slope(lo4, ri4);
            if (m4 < m4b){
                m4b = (d_yc4-d_yri)/(d_xc4-d_xri);	//slope(c4, ri4);
            }else{
                m4 = m4b;
                m4b = 0;
                d_xc4 = d_yc4 = FLT_MAX;
            }
        }else{
            m4 = (d_ylo-d_yri)/(d_xlo-d_xri);		//slope(lo4, ri4);
            m4b = 0;
            d_xc4 = d_yc4 = FLT_MAX;
        }
    }else{
        m4 = m4b = 0;
        d_xc4 = d_yc4 = FLT_MAX;
    }

    mh = 0;
    if (d_xri!=d_xle)
        mh = (d_yri-d_yle)/(d_xri-d_xle);	 		//slope(le2, ri1);



    auto ff = [=] __host__ __device__ (Point t_p) { 	float m;
        if(t_p.x!=d_xri){
            m=(t_p.y-d_yri)/(t_p.x-d_xri);									// slope(i, ri1);
            if(m<mh){												// p_i is above horizontal line (le2...ri1)
                if(t_p.x>d_xup){
                    if ((m<m1)||(t_p.x<d_xc1 && (t_p.y-d_yc1)/(t_p.x-d_xc1)<m1b)){
                        return 1;
                    }
                }else{
                    if(t_p.x<d_xup){
                        m=(t_p.y-d_yup)/(t_p.x-d_xup);						// slope(i, up2);
                        if ((m<m2)||(t_p.x<d_xc2 && (t_p.y-d_yc2)/(t_p.x-d_xc2)<m2b)){
                            return 1;
                        }
                    }
                }
            }else{
                if(t_p.x<d_xlo){
                    m=(t_p.y-d_yle)/(t_p.x-d_xle);							//slope(i, le3);
                    if ((m<m3)||(t_p.x>d_xc3 && (t_p.y-d_yc3)/(t_p.x-d_xc3)<m3b)){	//slope(i, c3);
                        return 1;
                    }
                }else{
                    if(t_p.x>d_xlo){
                        m=(t_p.y-d_ylo)/(t_p.x-d_xlo);						// slope(i, lo4);
                        if ((m<m4)||(t_p.x>d_xc4 && (t_p.y-d_yc4)/(t_p.x-d_xc4)<m4b)){// slope(i, c4);
                            return 1;
                        }
                    }
                }
            }
        }
        return 0;
    };        

    thrust::device_vector<Point> q_vec(n);
    auto result_end = thrust::copy_if(	thrust::device, p_ptr, 
                                                        p_ptr + n, 
                                                        q_vec.begin(), 
                                                        ff);

    thrust::host_vector<Point> h_res(q_vec.begin(), result_end);
    cudaDeviceSynchronize();kernelCallCheck();
    size = h_res.size();
    h_q = h_res;
    //printf("%i\n",h_res[5]);
}


// print indices and cooordinates of all axis extreme points, corners, and slopes
void filter_thrust_copy::print_extremes(){

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

void filter_thrust_copy::delete_filter(){
    // delete host variables
    //delete[] x;
    //delete[] y;

    cudaDeviceSynchronize();
    kernelCallCheck();

    // delete device variables
    cudaFree(d_p);

    cudaDeviceSynchronize();
    kernelCallCheck();
}