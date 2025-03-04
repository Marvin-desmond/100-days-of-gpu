#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define R1 5
#define C1 10
#define R2 10
#define C2 8

void initMatrix(int **A, int start_idx, int R, int C);
int main(int argc, char **argv) {
   // dynamically allocate memory for matrix A 
   int **A = (int **)malloc(R1 * sizeof(int*));
   for (int i = 0; i < C1; i++){
     A[i] = (int*)malloc(C1*sizeof(int));
   }
   // dynamically allocate memory for matrix B
   int **B = (int**)malloc(R2*sizeof(int*));
   for (int i = 0; i < C2; i++){
    B[i] = (int*)malloc(sizeof(int));
   }
   // initialize data as if we did arange then reshape in pytorch
   initMatrix(A, 0, R1, C1);
   initMatrix(B, R1 * C1 + 1, R1, C1);
   
   // Since CUDA expects a vector in a row-major format, 
   // let's flatten the input matrices before sending the 
   // data to the GPU
   free(A);
   free(B);
}

void initMatrix(int **A, int start_idx, int R, int C)
{
    for (int i = 0; i < R; i++){
        for (int j = 0; j < C; j++){
          A[i][j] = (i * C + j ) + start_idx;
        }
    }
}
