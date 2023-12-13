#include<random>

// Generate n numebers normal distributed
// with mean m and standard deviation s
// and store them in the array out
template <typename T>
void generate_normal(T *out, int n, double m, double s) {
    //std::random_device rd;
    //std::mt19937 gen(rd());
    srand(1);
    std::mt19937 gen;
    std::normal_distribution<> d(m, s);
    for (int i = 0; i < n; i++) {
        out[i] = (T) d(gen);
    }
}

// Generate n numbers uniform distributed
// between 0 and 1 and store them in the array out
template <typename T>
void generate_uniform(T *out, int n) {
    //std::random_device rd;
    //std::mt19937 gen(rd());
    srand(1);
    std::mt19937 gen;
    std::uniform_real_distribution<> d(0, 1);
    for (int i = 0; i < n; i++) {
        out[i] = (T) d(gen);
    }
}

// generate rand array of size n, using rand()
template <typename T>
void generate_rand(T *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = rand();
    }
}

// Generate sequence of n natural numbers
// and store them in the array out
template <typename T>
void generate_sequence(T *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = i/1;
    }
}

// Generate a array of size n with a constant value
// and store it in the array out
template <typename T>
void generate_constant(T *out, int n, double value) {
    for (int i = 0; i < n; i++) {
        out[i] = value;
    }
}

// Generate a bit array of size n with probability p
// and store it in the array out
template <typename T>
void generate_bit(T *out, int n, double p) {
    for (int i = 0; i < n; i++) {
        out[i] = (std::rand() / (double)RAND_MAX) < p;
    }
}

// Print array of size n
void print_array(float *out, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", out[i]);
    }
    printf("\n");
}

void print_array(int *out, int n) {
    for (int i = 0; i < n; i++) {
        printf("%i ", out[i]);
    }
    printf("\n");
}

template <typename T>
void print_array(T *out, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", (T)out[i]);
    }
    printf("\n");
}

// Test if input is equal to the array out
// of size n
template <typename T>
bool array_equal(T *out, T *input, int n) {
    for (int i = 0; i < n; i++) {
        if ((T)out[i] != (T)input[i]) {
            return false;
        }
    }
    return true;
}

// function to save in a file all the points and candidate to hull in diffrent .off files, name_point.off and name_candidate.off respectively
void save_off_file(filter_cpu_serial *filter, std::string name){
    std::ofstream file_point;
    std::ofstream file_candidate;
    file_point.open(name+"_point.off");
    file_candidate.open(name+"_candidate.off");
    file_point << "OFF\n";
    file_candidate << "OFF\n";
    file_point << filter->n << " 0 0\n";
    file_candidate << filter->sizeHull << " 0 0\n";
    for(int i=0; i<filter->n; i++){
        file_point << filter->x[i] << " " << filter->y[i] << " " << filter->z[i] << "\n";
    }
    for(int i=0; i<filter->sizeHull; i++){
        file_candidate << filter->x[filter->h_q[i]] << " " << filter->y[filter->h_q[i]] << " " << filter->z[filter->h_q[i]] << "\n";
    }
    file_point.close();
    file_candidate.close();
}

