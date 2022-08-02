// class filter
class filter_cpu_serial : public filter{
public:
   filter_cpu_serial(float *x_in, float *y_in, INDEX size);

   void cpu_manhattan();
   void cpu_euclidean();

   void print_extremes();
};
