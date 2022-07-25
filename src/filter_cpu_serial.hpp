// class filter
class filter_cpu_serial : public filter{
public:
    filter_cpu_serial(float *x_in, float *y_in, INDEX size){
        x = x_in;
        y = y_in;
        n = size;
    }



    // find the minimum manhattan distance between a corner and a set of points
    template <typename T>
    void findCorner_manhattan(INDEX *corner, T *x, T *y, T xc, T yc, INDEX n){
        T min = std::numeric_limits<float>::max();
        for(INDEX i = 0; i < n; i++){
            T dist = std::abs(x[i] - xc) + std::abs(y[i] - yc);
            if(dist < min){
                min = dist;
                corner[0] = i;
            }
        }
    }

    // find the minimum euclidean distance between a corner and a set of points
    template <typename T>
    void findCorner_euclidean(INDEX *corner, T *x, T *y, T xc, T yc, INDEX n){
        T min = std::numeric_limits<float>::max();
        for(INDEX i = 0; i < n; i++){
            T dist = std::sqrt(std::pow(x[i] - xc, 2) + std::pow(y[i] - yc, 2));
            if(dist < min){
                min = dist;
                corner[0] = i;
            }
        }
    }

    void cpu_manhattan(){
        // get extreme points
        findMax<float>(&ri, x, n);
        findMin<float>(&le, x, n);
        findMax<float>(&up, y, n);
        findMin<float>(&lo, y, n);

        xri = x[ri]; yri = y[ri];
        xle = x[le]; yle = y[le];
        xup = x[up]; yup = y[up];
        xlo = x[lo]; ylo = y[lo];

        // find corners using manhattan distance
        findCorner_manhattan<float>(&c1, x, y, xri, yup, n);
        findCorner_manhattan<float>(&c2, x, y, xle, yup, n);
        findCorner_manhattan<float>(&c3, x, y, xle, ylo, n);
        findCorner_manhattan<float>(&c4, x, y, xri, ylo, n);
        
        xc1 = x[c1]; yc1 = y[c1];
        xc2 = x[c2]; yc2 = y[c2];
        xc3 = x[c3]; yc3 = y[c3];
        xc4 = x[c4]; yc4 = y[c4];

        // compute the slopes
        computeSlopes(); 

        h_q = new INDEX[n];
        INDEX j = 0;
        float h_x, h_y, m;

        for (INDEX i = 0; i < n; i++){
            h_x = x[i];
            h_y = y[i];
            if(h_x!=xri){
                m=(h_y-yri)/(h_x-xri);									// slope(i, ri1);
                if(m<mh){												// p_i is above horizontal line (le2...ri1)
                    if(h_x>xup){
                        if ((m<m1)||(h_x<xc1 && (h_y-yc1)/(h_x-xc1)<m1b)){
                            h_q[j] = i;
                            j++;
                        }
                    }else{
                        if(h_x<xup){
                            m=(h_y-yup)/(h_x-xup);						// slope(i, up2);
                            if ((m<m2)||(h_x<xc2 && (h_y-yc2)/(h_x-xc2)<m2b)){
                                h_q[j] = i;
                                j++;
                            }
                        }
                    }
                }else{
                    if(h_x<xlo){
                        m=(h_y-yle)/(h_x-xle);							//slope(i, le3);
                        if ((m<m3)||(h_x>xc3 && (h_y-yc3)/(h_x-xc3)<m3b)){	//slope(i, c3);
                            h_q[j] = i;
                            j++;
                        }
                    }else{
                        if(h_x>xlo){
                            m=(h_y-ylo)/(h_x-xlo);						// slope(i, lo4);
                            if ((m<m4)||(h_x>xc4 && (h_y-yc4)/(h_x-xc4)<m4b)){// slope(i, c4);
                                h_q[j] = i;
                                j++;
                            }
                        }
                    }
                }
            }
        }
        size = j;

    }

    void cpu_euclidean(){
        // get extreme points
        findMax<float>(&ri, x, n);
        findMin<float>(&le, x, n);
        findMax<float>(&up, y, n);
        findMin<float>(&lo, y, n);

        xri = x[ri]; yri = y[ri];
        xle = x[le]; yle = y[le];
        xup = x[up]; yup = y[up];
        xlo = x[lo]; ylo = y[lo];

        // find corners using manhattan distance
        findCorner_euclidean<float>(&c1, x, y, xri, yup, n);
        findCorner_euclidean<float>(&c2, x, y, xle, yup, n);
        findCorner_euclidean<float>(&c3, x, y, xle, ylo, n);
        findCorner_euclidean<float>(&c4, x, y, xri, ylo, n);
        
        xc1 = x[c1]; yc1 = y[c1];
        xc2 = x[c2]; yc2 = y[c2];
        xc3 = x[c3]; yc3 = y[c3];
        xc4 = x[c4]; yc4 = y[c4];

        // compute the slopes
        computeSlopes(); 

        h_q = new INDEX[n];
        INDEX j = 0;
        float h_x, h_y, m;

        for (INDEX i = 0; i < n; i++){
            h_x = x[i];
            h_y = y[i];
            if(h_x!=xri){
                m=(h_y-yri)/(h_x-xri);									// slope(i, ri1);
                if(m<mh){												// p_i is above horizontal line (le2...ri1)
                    if(h_x>xup){
                        if ((m<m1)||(h_x<xc1 && (h_y-yc1)/(h_x-xc1)<m1b)){
                            h_q[j] = i;
                            j++;
                        }
                    }else{
                        if(h_x<xup){
                            m=(h_y-yup)/(h_x-xup);						// slope(i, up2);
                            if ((m<m2)||(h_x<xc2 && (h_y-yc2)/(h_x-xc2)<m2b)){
                                h_q[j] = i;
                                j++;
                            }
                        }
                    }
                }else{
                    if(h_x<xlo){
                        m=(h_y-yle)/(h_x-xle);							//slope(i, le3);
                        if ((m<m3)||(h_x>xc3 && (h_y-yc3)/(h_x-xc3)<m3b)){	//slope(i, c3);
                            h_q[j] = i;
                            j++;
                        }
                    }else{
                        if(h_x>xlo){
                            m=(h_y-ylo)/(h_x-xlo);						// slope(i, lo4);
                            if ((m<m4)||(h_x>xc4 && (h_y-yc4)/(h_x-xc4)<m4b)){// slope(i, c4);
                                h_q[j] = i;
                                j++;
                            }
                        }
                    }
                }
            }
        }
        size = j;

    }

};
