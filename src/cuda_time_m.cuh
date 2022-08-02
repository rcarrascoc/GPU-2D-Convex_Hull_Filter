
class cuda_time_m
{
public:
    // declare start time, end time, and time difference as a float
    cuda_time_m();

//private:
    float acc_time;
    float end_time;
    float time_acc;

    void init();
    void start();
    void pause();
    void stop();
    float get_time();

};