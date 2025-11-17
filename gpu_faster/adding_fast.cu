// here I am not using the unified allocator rather just using the fast allocator 

#include <iostream>
#include <cmath>
using namespace std;

__global__
void add(int n, float *x, float *y)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    float *h_x, *h_y;
    float *d_x, *d_y;

    // Allocate host memory
    h_x = (float*) malloc(size);
    h_y = (float*) malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    // Copy from host to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blocksize = 256;
    int numBlocks = (N + blocksize - 1) / blocksize;
    add<<<numBlocks, blocksize>>>(N, d_x, d_y);

    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    // Verify results
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(h_y[i] - 3.0f));
    }

    cout << "Max error: " << maxError << endl;

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}
