#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

void check_error(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void vec_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {

    float *h_a, *h_b, *h_c; // Host pointers.
    float *d_a, *d_b, *d_c; // Device pointers.
    int n = 1024;
    size_t size = n * sizeof(float);

    // Allocate memory on the host.
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);

    // Initialize host arrays.
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // Allocate memory on the device.
    check_error(cudaMalloc((void **)&d_a, size));
    check_error(cudaMalloc((void **)&d_b, size));
    check_error(cudaMalloc((void **)&d_c, size));

    // Copy data from host to device.
    check_error(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    check_error(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch the kernel.
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vec_add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    check_error(cudaGetLastError());
    check_error(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify the result.
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %f + %f != %f\n", i, h_a[i], h_b[i], h_c[i]);
        }
    }

    printf("Vector addition completed successfully.\n");

    // Free memory.
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
