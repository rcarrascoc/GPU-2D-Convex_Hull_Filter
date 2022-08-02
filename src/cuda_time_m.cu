// get cuda time for each kernel
//#include "cuda_time_m.cuh"

cudaEvent_t start_event, stop_event;

cuda_time_m::cuda_time_m(){
    acc_time = 0;
    end_time = 0;
    time_acc = 0;
}

void cuda_time_m::init(){
    // start_event and stop_event are CUDA events
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaDeviceSynchronize();
}

void cuda_time_m::start(){
    cudaEventRecord(start_event);
}

void cuda_time_m::pause(){
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&acc_time, start_event, stop_event);
    end_time += acc_time;
}

void cuda_time_m::stop(){
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
}

// get time 
float cuda_time_m::get_time(){
    return end_time;
}
