#include <random>

// generate x and y random arrays of size n
// where the x and y arrays are T types
template <typename T>
void generate_random_points(T* x, T* y, INDEX n) {
    for (INDEX i = 0; i < n; i++) {
        x[i] = rand() % n;
        y[i] = rand() % n;
    }
}

// generate x and y random arrays of size n
// where the x and y arrays are T types
// and normal distribution is used
template <typename T>
void generate_random_points_normal(T* x, T* y, INDEX n) {
    srand(1);
    std::mt19937 generator;
	double mean = 0.5;
	double stddev  = 0.1;
	std::normal_distribution<float> dist(mean, stddev);
	for(INDEX i=0; i<n; i++){
		x[i] = dist(generator);
		y[i] = dist(generator);
	}
}

// generate x and y random arrays of size n
// where the x and y arrays are T types
// and uniform distribution is used
template <typename T>
void generate_random_points_uniform(T* x, T* y, INDEX n) {
    srand(1);
    std::mt19937 generator;
    double mean = 0.5;
    double stddev  = 0.1;
    std::uniform_real_distribution<float> dist(mean, stddev);
    for(INDEX i=0; i<n; i++){
        x[i] = dist(generator);
        y[i] = dist(generator);
    }
}

// generate x and y random arrays of size n
// where the x and y arrays are T types
// and circular distribution is used
template <typename T>
void generate_random_points_circumference(T *X, T *Y, INDEX n, double prob){
    srand(1);
	INDEX i;
	INDEX N = n - n*prob;		// points on the circumference.	CASO 1
	float cx, cy, ra;
	float x, y, alpha, tg, fact;
	
	ra = 1.0;
	cx = cy = 0.5;	// Center (0.5, 0.5)

	// we generate N random points ON the circumference...
	//srand (time(NULL));
	// FOR PARA EL CASO 1
	for(i=0; i<N; i++){
		fact = static_cast <float> (rand())/static_cast <float> (RAND_MAX);
		alpha = fact*360.0;	// generate a random angle: 0 <= alpha <= 360
		if (alpha == 0 || alpha == 360){
			y = 0;
			if (alpha == 0)
				x = ra;
			else
				x = -1*ra;
		}else if (alpha == 90 || alpha == 270){
			x = 0;
			if (alpha == 90)
				y = ra;
			else
				y = -1*ra;
		} else {
			tg = tan(alpha);
			x = ra / sqrt(1 + pow(tg, 2.0));
			if (alpha > 90 && alpha < 270)
				x*=-1;
			y = x*tg;
		}
		//cout << alpha << "(" << x << "," << y << ") " << endl;
		// here know x^2 + y^2 = ra^2
		X[i]= cx+x;
		Y[i]= cy+y;
	}

	float dt = ra*prob;
	// we generate n-N random points INSIDE OF / OR ARAOUND OF the circumference...
	for(; i<n; i++){
		fact = static_cast <float> (rand())/static_cast <float> (RAND_MAX);
		alpha = fact*360.0;	// generate a random angle: 0 <= alpha <= 360
		fact = static_cast <float> (rand())/static_cast <float> (RAND_MAX);
		if (alpha == 0 || alpha == 360){
			y = 0;
			if (alpha == 0){
				x = ra+fact*dt;
			}else
				x = ra-fact*dt;
		}else if (alpha == 90 || alpha == 270){
			x = 0;
			if (alpha == 90)
				y = ra+fact*dt;
			else
				y = ra-fact*dt;
		} else {
			tg = tan(alpha);
			if (rand()%2)
				x = (ra+fact*dt) / sqrt(1 + pow(tg, 2.0));
			else
				x = (ra-fact*dt) / sqrt(1 + pow(tg, 2.0));

			if (alpha > 90 && alpha < 270)
				x*=-1;
			y = x*tg;
		}
		//cout << alpha << "(" << x << "," << y << ") " << endl;
		// at this moment: x^2 + y^2 = ra^2
		X[i]= cx+x;
		Y[i]= cy+y;
	}
}

// print the x and y arrays of size n
template <typename T>
void print_points(T* x, T* y, INDEX n) {
	for (int i = 0; i < n; i++) {
		std::cout << x[i] << " " << y[i] << std::endl;
	}
	std::cout << std::endl;
}