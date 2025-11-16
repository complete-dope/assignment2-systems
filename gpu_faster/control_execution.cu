#include <iostream>
#include <cuda_runtime.h>

const int N = 10240;   // Matrix size
const int runs = 5;   // Timed iterations

// --------------------------------------
// Kernel with control divergence
__global__ void matMulConditional(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }

        // Conditional divergence
        if (threadIdx.x % 2 == 0) {
            C[row * N + col] = value;         // operation 1
        } else {
            C[row * N + col] = 2.0f * value; // operation 2
        }
    }
}

// --------------------------------------
// Kernel without control divergence
__global__ void matMulNoConditional(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }

        // No conditional
        C[row * N + col] = value;
    }
}

// --------------------------------------
int main() {
    size_t bytes = N * N * sizeof(float);

    // Host memory
    float *h_A = new float[N*N];
    float *h_B = new float[N*N];
    float *h_C = new float[N*N];

    // Initialize matrices
    for (int i = 0; i < N*N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    // -----------------------
    // Warmup (do not count)
    matMulConditional<<<blocks, threads>>>(d_A, d_B, d_C, N);
    matMulConditional<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // -----------------------
    // Conditional kernel benchmark
    float total_cond = 0.0f;
    std::cout << "Running conditional kernel..." << std::endl;

    for (int r = 0; r < runs; r++) {
        cudaEventRecord(start);
        matMulConditional<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_cond += ms;
        std::cout << "Run " << r+1 << ": " << ms << " ms" << std::endl;
    }

    std::cout << "Average conditional kernel time: " << (total_cond / runs) << " ms\n" << std::endl;

    // -----------------------
    // Warmup for non-conditional kernel
    matMulNoConditional<<<blocks, threads>>>(d_A, d_B, d_C, N);
    matMulNoConditional<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // -----------------------
    // Non-conditional kernel benchmark
    float total_no_cond = 0.0f;
    std::cout << "Running non-conditional kernel..." << std::endl;

    for (int r = 0; r < runs; r++) {
        cudaEventRecord(start);
        matMulNoConditional<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_no_cond += ms;
        std::cout << "Run " << r+1 << ": " << ms << " ms" << std::endl;
    }

    std::cout << "Average non-conditional kernel time: " << (total_no_cond / runs) << " ms" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
