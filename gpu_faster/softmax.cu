// writing softmax code 

#include <iostream>
#include <cuda_runtime.h>

// allocate some memory
const int BLOCK_DIM_Y = ;
__shared__ float reductionp[BLOCK_DIM_Y];

float maxval = FLOAT_MIN;

// softmax : first find the max value from the array, then subtract that from all the values present in the array, then exponentiate it, then sum all the exponents then divide the exponents individual values with the sum !   

// FLOPS : N (finding max) + N ( subtract) + N (exponent) + N (sum of exp) + N (division by sum) => 5*N operations  

// __global__ memory : this is dram memory aka vram memory
// __shared__ memory : these are given / shared between 1024 threads aka 32 warps and each warps here has 32 threads 

// normal accessing memory that is using idx , idx+N , idx+2N .. that is very bad for memory colaesing , so better is to use the threads that are in patterns / blocks that are sequentially aligned , so memory pattern should be in a way that each warp access value that are in adjacent in memory  






