// writing custom cuda kernel for gelu operation 
// nvcc gelu_kernel.cu -o gelu_kernel.o
// nsys profile -t cuda --stats=true ./gelu_kernel

#include <stdio.h> 
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda_fp16.h>
#include <math_constants.h>

// do I have this lib ?  

using namespace std;

// __global__
// float tanh(float *x){
//     int thread = blockIdx.x * blockDim.x + threadIdx.x
//     return math.exp()
// }

__global__
void naive_gelu_kernel(float *x , int N){
    // This is GELU in one kernel (computationally this is in as single kernel but this is not fused / tilled etc)  
    // This is a naive kernel because you are reading the value of x[i] multiple times ( 4 times loading it from the global memory)   
    
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i=thread;i<N;i+=stride){ 
        x[i] = 0.5 * x[i] * (1 + tanhf(1.1283 * (x[i] + 0.044715 * x[i] * x[i] * x[i])));
    }
}

__global__
void fused_gelu_kernel(float *x , int N){
    // first step for fusion : load once, reuse v 
    // second step : you are computing v*v*v that same thing is getting computed again and again and  is storing intermediates (so any variable that is local to a thread is stored in the registers)
    // third step : Using fmaf aka fused multiply addition ( this reduces counts)
    // 4th step : using constantexp so that they dont occupy registers and keep those outside the loop
    // 5th step : long lived variable, you dont need those and rather we need to avoid there longliveness
    // 6th step : use the fused (__tanh) operation to avoid spills and cut latency 
    // immediate consumption that makes sure a variable is not long live in memory : they get used up and they die immediately ! 
    // still not fused something important is missing  

    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    constexpr float a = 0.044715f;
    constexpr float b = 1.1283f;
    constexpr float half = 0.5f;

    for(int i=thread;i<N;i+=stride){

        float v = x[i];
        float v2 = v * v;
        float v3 = v2 * v;

        float inner = fmaf(a, v3, v);
        float scaled_arg = fmaf(b, inner, 0.f);
        float t = tanhf(scaled_arg); 
        float one_plus_t = fmaf(1.f, t, 1.f); 
        float v_times = v * one_plus_t;

        // x[i] = half * v * (one + t);
        x[i] = half * v_times;
    }
}

int main(){
    // this is the function inside main that runs !
    float *x;
    int N = 1<<20; // array length of 1Million

    cudaMallocManaged(&x,N*sizeof(float));
    
    for(int i=0;i<N;i++){
        x[i] = i-10.0;
    }

    int blocksize = 256;
    int numBlocks = (N+blocksize-1)/ blocksize; // (1048576 + 255) / 256
    fused_gelu_kernel<<<numBlocks, blocksize>>>(x, N);

    cudaDeviceSynchronize();

    for (int i =0;i<15;i++){
        printf("%f \n" ,x[i]);
    }

    cudaFree(x);
}


