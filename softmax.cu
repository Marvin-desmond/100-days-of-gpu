#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>

__global__ void maxKernel(float *x, float *o, int n) {
    unsigned int i = threadIdx.x * 2;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
       if (threadIdx.x % stride == 0 && i < n) {
         x[i] = x[i] > x[i + stride] ? x[i] : x[i + stride];
       }
       __syncthreads();
    }
    if (threadIdx.x == 0){
        *o = x[0];
    }
}

__global__ void sumKernel(float *x, float *m, float *den, int n) {
    unsigned int i = threadIdx.x * 2;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
       if (threadIdx.x % stride == 0 && (i + stride) < n) {
        if (stride == 1) {
           x[i] = exp(x[i] - *m) + exp(x[i + stride] - *m);
        } else {
           x[i] += x[i + stride];
        }
       }
       __syncthreads();
    }
    if (threadIdx.x == 0){
        *den = x[0];
    }
}

__global__ void divKernel(float *x, float *m, float *den, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        x[idx] = exp(x[idx] - *m) / *den;
    }
}

void softMax(float *x, float *max, float *den, float *out, int w) {
    int THREADS = 32;
    int size = w * sizeof(float);
    int scalar = sizeof(float);
    float *x_d, *x2_d, *max_d, *den_d, *out_d;
    cudaMalloc((void**)&x_d, size);
    cudaMalloc((void**)&x2_d, size);
    cudaMalloc((void**)&max_d, scalar);
    cudaMalloc((void**)&den_d, scalar);
    cudaMalloc((void**)&out_d, size);
    
    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice); // for computing max
    cudaMemcpy(x2_d, x_d, size, cudaMemcpyDeviceToDevice); // for computing sum
    cudaMemcpy(out_d, x_d, size, cudaMemcpyDeviceToDevice); // for computing final output

    maxKernel<<<ceil(w/(float)THREADS), THREADS>>>(x_d, max_d, w);
    sumKernel<<<ceil(w/(float)THREADS), THREADS>>>(x2_d, max_d, den_d, w);
    divKernel<<<ceil(w/(float)THREADS), THREADS>>>(out_d, max_d, den_d, w);
    cudaMemcpy(max, max_d, scalar, cudaMemcpyDeviceToHost);
    cudaMemcpy(den, den_d, scalar, cudaMemcpyDeviceToHost);
    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);
    cudaFree(max_d);
    cudaFree(den_d);
    cudaFree(x_d);
    cudaFree(x2_d);
}

int main() {
    int w = 10;
    float max_val = 0.0f;
    float den = 0.0f;
    float sum_out = 0.0f;
    float x[w] = {0.6964, 0.6538, 0.4852, 0.6394, 0.6194, 0.1290, 0.2548, 0.6966, 0.9732, 0.9298};
    float out[w];
    softMax(x, &max_val, &den, out, w);
    for (int i = 0; i < w; i++){
        printf("out at %d: %f\n", i, out[i]);
        sum_out += out[i];
    }
    printf("result value: %f\n", sum_out);
}

