// this adds the values that are present in the array 

#include <iostream> 
#include <math.h>

using namespace std;

// For memory allocation whether its a gpu or a cpu use the unified memory access it auto manages the allocation part for us 
 
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x; // gets the index of the current thread within its block 
  int stride = blockDim.x; // contains the no. of threads in a blocks
 for (int i = index; i < n; i+=stride)
   y[i] = x[i] + y[i];
}
 
int main(void)
{
 int N = 1<<20;
 float *x, *y;
 
 // Allocate Unified Memory – accessible from CPU or GPU
 cudaMallocManaged(&x, N*sizeof(float));
 cudaMallocManaged(&y, N*sizeof(float));
 
 // initialize x and y arrays on the host
 for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }
 
 // Run kernel on 1M elements on the GPU
 int blocksize = 256;
 int numBlocks = (N+blocksize-1)/ blocksize;
 add<<<numBlocks, blocksize>>>(N, x, y);  // 1st param tells no. of thread blocks to use , 2nd param tells no. of threads in a thread block (multiples of 32) 
 // hardware sets the bound, the programmer decides how many he needs to use at at time !
 // we do have a logical-mapping that makes this whole easier to work with structured data like images / videos !! 
  

 
 // Wait for GPU to finish before accessing on host
 cudaDeviceSynchronize();
 
 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++) {
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 }
 cout << "Max error: " << maxError << endl;
 
 // Free memory
 cudaFree(x);
 cudaFree(y);

  return 0;
}