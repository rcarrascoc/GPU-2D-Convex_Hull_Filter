// class filter
class filter_cpu_parallel : public filter{
public:
   filter_cpu_parallel(float *x_in, float *y_in, INDEX size);

   void cpu_manhattan();
   void cpu_euclidean();

   void print_extremes();
};
