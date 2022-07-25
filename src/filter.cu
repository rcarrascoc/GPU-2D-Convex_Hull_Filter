class filter {
public:
    // host variables
    float *x;
    float *y;
    INDEX n;

    // host extreme points
    INDEX ri, le, lo, up;
    float xri, xle, xlo, xup;
    float yri, yle, ylo, yup;

    // corner points
    INDEX c1, c2, c3, c4;
    float xc1, xc2, xc3, xc4;
    float yc1, yc2, yc3, yc4;

    // slope of the line
	float m1, m2, m3, m4;		
	float m1b, m2b, m3b, m4b;
	float mh;	

    // device variables
    float *d_x;
    float *d_y;
    INDEX *d_q, *d_qa, *d_size;

    // ouptut variables
    INDEX size;    
    float *d_out_x, *d_out_y;
    float *out_x, *out_y;
    INDEX *h_q;

    void print_extremes();

    void computeSlopes(){
        if (xri!=xup){
            if (xc1>xup && yc1>yri){
                m1 = (yri-yc1)/(xri-xc1); 		//slope(ri1, c1);
                m1b = (yri-yup)/(xri-xup);		//slope(ri1, up1);
                if (m1 < m1b){
                    m1b = (yc1-yup)/(xc1-xup); 	//slope(c1, up1);
                }else{
                    m1 = m1b;
                    m1b = 0;
                    xc1 = yc1 = -1*FLT_MAX;
                }
            }else{
                m1 = (yri-yup)/(xri-xup);		//slope(ri1, up1);
                m1b = 0;
                xc1 = yc1 = -1*FLT_MAX;
            }
        }else{
            m1 = m1b = 0;
            xc1 = yc1 = -1*FLT_MAX;
        }
    
        if (xup!=xle){
            if (xc2<xup && yc2>yle){
                m2 = (yup-yc2)/(xup-xc2);			//slope(up2, c2);
                m2b = (yup-yle)/(xup-xle);		//slope(up2, le2);
                if (m2 < m2b){
                    m2b = (yc2-yle)/(xc2-xle);	//slope(c2, le2);
                }else{
                    m2 = m2b;
                    m2b = 0;
                    xc2 = yc2 = -1*FLT_MAX;
                }
            }else{
                m2 = (yup-yle)/(xup-xle);		//slope(up2, le2);
                m2b = 0;
                xc2 = yc2 = -1*FLT_MAX;
            }
        }else{
            m2 = m2b = 0;
            xc2 = yc2 = -1*FLT_MAX;
        }
    
        if (xle!=xlo){
            if (xc3<xlo && yc3<yle){
                m3 = (yle-yc3)/(xle-xc3);			//slope(le3, c3);
                m3b = (yle-ylo)/(xle-xlo);		//slope(le3, lo3);
                if (m3 < m3b){
                    m3b = (ylo-yc3)/(xlo-xc3);	//slope(c3, lo3);
                }else{
                    m3 = m3b;
                    m3b = 0;
                    xc3 = yc3 = FLT_MAX;
                }
            }else{
                m3 = (yle-ylo)/(xle-xlo);		//slope(le3, lo3);
                m3b = 0;
                xc3 = yc3 = FLT_MAX;
            }
        }else{
            m3 = m3b = 0;
            xc3 = yc3 = FLT_MAX;
        }
    
        if (xlo!=xri){
            if (xc4>xlo && yc4<yri){
                m4 = (ylo-yc4)/(xlo-xc4);			//slope(lo4, c4);
                m4b = (ylo-yri)/(xlo-xri);		//slope(lo4, ri4);
                if (m4 < m4b){
                    m4b = (yc4-yri)/(xc4-xri);	//slope(c4, ri4);
                }else{
                    m4 = m4b;
                    m4b = 0;
                    xc4 = yc4 = FLT_MAX;
                }
            }else{
                m4 = (ylo-yri)/(xlo-xri);		//slope(lo4, ri4);
                m4b = 0;
                xc4 = yc4 = FLT_MAX;
            }
        }else{
            m4 = m4b = 0;
            xc4 = yc4 = FLT_MAX;
        }
    
        if (xri!=xle)
            mh = (yri-yle)/(xri-xle);	 		//slope(le2, ri1);
        else
            mh = 0;
    }

    /*// declare destructor
    ~filter(){
        printf("aca estoy");
    } //*/
};