#include <iostream>
#include <cuda_runtime.h>

// Simple matrix multiplication kernel (C = A * B)
__global__
void matMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

int main() {

    const int N = 1024;     // 1024 x 1024 matrix
    const int runs = 5;     // run 5 times

    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N*N];
    float *h_B = new float[N*N];
    float *h_C = new float[N*N];

    // Initialize matrices
    for (int i = 0; i < N*N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    float total_time = 0.0f;

    std::cout << "Running heavy matrix multiplication (" << N << "x" << N
              << ") 5 times...\n" << std::endl;

    for (int r = 1; r <= runs; r++) {

        cudaEventRecord(start);

        // Launch kernel
        matMul<<<blocks, threads>>>(d_A, d_B, d_C, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;

        std::cout << "Run " << r << " execution time: "
                  << ms << " ms" << std::endl;
    }

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Print average time
    std::cout << "\nAverage execution time over 5 runs: "
              << (total_time / runs) << " ms" << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
