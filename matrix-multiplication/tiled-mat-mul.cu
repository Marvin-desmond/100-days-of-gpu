#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define R1 5
#define C1 10
#define R2 10
#define C2 8

# define TILE_WIDTH 16

void initMatrix(int **A, int start_idx, int R, int C);
void flattenMatrix(int **M, int *M_flat, int R, int C);

void checkLastError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void tiledMatMulKernel(int *A, int *B, int *C, int M, int N, int P) {
    __shared__ int Ms[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Ns[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int tx = threadIdx.x;
    int by = blockIdx.y; int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    int pixel_value = 0;
    for (int i = 0; i < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++i){
        Ms[ty][tx] = A[Row * N + i * TILE_WIDTH + tx];
        Ns[ty][tx] = B[(i * TILE_WIDTH + ty) * P + Col];
        __syncthreads();
        for (int j = 0; j < TILE_WIDTH; ++j) {
            pixel_value += Ms[ty][j] * Ns[j][tx];
        }
        __syncthreads();
    }
    if (Row < M && Col < P) {
       C[Row * P + Col] = pixel_value;
   }
}

void matMul(int* A, int* B, int *C, int M, int N, int P) {
    int *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, M * N * sizeof(int));
    cudaMalloc((void**)&B_d, N * P * sizeof(int));
    cudaMalloc((void**)&C_d, M * P * sizeof(int));
    cudaMemcpy(A_d, A, M*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*P*sizeof(int), cudaMemcpyHostToDevice);
    // specifying grid size and block size
    dim3 dimGrid(1., 1., 1.);
    dim3 dimBlock(16., 16., 1.);
    tiledMatMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, P);
    cudaDeviceSynchronize();
    cudaMemcpy(C, C_d, M*P*sizeof(int), cudaMemcpyDeviceToHost);
    checkLastError();
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char **argv) {
   // dynamically allocate memory for matrix A 
   int **A = (int **)malloc(R1 * sizeof(int*));
   for (int i = 0; i < R1; i++){
     A[i] = (int*)malloc(C1*sizeof(int));
   }
   // dynamically allocate memory for matrix B
   int **B = (int**)malloc(R2*sizeof(int*));
   for (int i = 0; i < R2; i++){
    B[i] = (int*)malloc(C2*sizeof(int));
   }
   // initialize data as if we did arange then reshape in pytorch
   initMatrix(A, 0, R1, C1);
   initMatrix(B, R1 * C1 + 1, R2, C2);
   printf(" start for B: %d\n", R1 * C1 + 1);
   // Since CUDA expects a vector in a row-major format, 
   // let's flatten the input matrices to vector variables before sending the 
   // data to the GPU
   int *A_flat = (int*)malloc(R1*C1*sizeof(int));
   int *B_flat = (int*)malloc(R2*C2*sizeof(int));
   flattenMatrix(A, A_flat, R1, C1);
   flattenMatrix(B, B_flat, R2, C2);
   // declare the resulting matrix for holding the results
   int *C_flat = (int*)malloc(R1*C2*sizeof(int));
   matMul(A_flat, B_flat, C_flat, R1, C1, C2);
   // convert it back to matrix
   int **C = (int**)malloc(R1*sizeof(int*));
   for (int i = 0; i < R1; i++){
     C[i] = (int*)malloc(C2*sizeof(int));
   }
   // allocation to C matrix
   for (int i = 0; i < R1; i++){
    for (int j = 0; j < C2; j++){
        C[i][j] = C_flat[i * C2 + j];
    }
   }
   // let's print the values of the final matrix
   for (int i = 0; i < R1; i++){
    for (int j = 0; j < C2; j++){
        printf("%d, ", C[i][j]);
    }
    printf("\n");
   }

   free(A_flat);   
   free(B_flat);   
   free(C_flat);   
   free(A);
   free(B);
   free(C);
}

void initMatrix(int **A, int start_idx, int R, int C)
{
    for (int i = 0; i < R; i++){
        for (int j = 0; j < C; j++){
          A[i][j] = (i * C + j ) + start_idx;
        }
    }
}

void flattenMatrix(int **M, int *M_flat, int R, int C){
   for (int i = 0; i < R; i++){
      for (int j = 0; j < C; j++){
        M_flat[i * C + j] = M[i][j];
      }
   }
}

