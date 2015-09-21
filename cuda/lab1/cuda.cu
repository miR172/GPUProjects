#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>

#define ITERATION 500
#define BLKSIZE 512 

typedef unsigned long long bint;

float * allocate(bint n){
  bint size = (n+1)*(n+1);
  float *m = (float *)calloc(size, sizeof(float));

  bint i;
  for (i=0; i<n+1; i++){
    m[i] = 80;
    m[size-1-i] = 80;
    m[(n+1)*i] = 80;
    m[(n+1)*(i+1)-1] = 80;
    m[i] = (i >=10 && i<=30) ? 150 : m[i];
  }

  return m;
}

float avg(float *m, bint dim){
  bint size = dim*dim;
  float sum = 0;
  bint i;
  for (i=0; i<size; i++){
    sum += m[i];
    //if (i % dim==0)
    //  printf("\n");
    //printf("%f ", m[i]);
  }
  //printf("\n");
  return sum/size;
}


__global__ void simulateKernel(float *s, float *d, bint dim){
  //dim is one side length of matrix
  bint tid = threadIdx.x + blockIdx.x * blockDim.x;
  bint i = tid + dim + 1 + 2*(tid/(dim-2));
  if (i < dim*(dim-1)-1)
    d[i] = (s[i-1] + s[i+1] + s[i-dim] + s[i+dim]) / 4;
}


int main(int argc, char *argv[]){

  if (argc < 2){
    printf("Please indicate matrix size.\n");
    exit(0);
  }
  bint n = atoi(argv[1]);

  float *m = allocate(n);

  //float mean = avg(m, n+1);
  //printf("%f===>",mean);


  // allocation and copy to DEVICE
  float * a, *b;
  bint mem = (n+1)*(n+1)*sizeof(float);
  cudaMalloc((void **)&a,  mem);
  cudaMalloc((void **)&b,  mem);
  cudaMemcpy(a, m, mem, cudaMemcpyHostToDevice);
  cudaMemcpy(b, m, mem, cudaMemcpyHostToDevice);

  // call kernel function
  bint gridSize = ((n+1)*(n+1) % BLKSIZE == 0)? (n+1)*(n+1)/BLKSIZE : (n+1)*(n+1)/BLKSIZE+1;
  int i;
  for (i=0; i<ITERATION/2; i++){
    simulateKernel<<<gridSize, BLKSIZE>>>(a, b, n+1);
    simulateKernel<<<gridSize, BLKSIZE>>>(b, a, n+1);
  }
  if (ITERATION%2 !=0){
    simulateKernel<<<gridSize, BLKSIZE>>>(a, b, n+1);
    cudaMemcpy(m, b, mem, cudaMemcpyDeviceToHost);
  }
  else{
    cudaMemcpy(m, a, mem, cudaMemcpyDeviceToHost);
  }

  //mean = avg(m, n+1);
  //printf("%f\n", mean);

  free(m);
  cudaFree(a);
  cudaFree(b);

  return 0;
}
