#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct __SMALL_INT{
  unsigned int i:4;
}small_int;

__constant__ char data[4][3];

__global__ void solve(int ** result){
 __shared__ small_int chrs[4][2];
 int i = threadIdx.x;
 int j;
 for (j=0;j<2;j++){
 chrs[i][j] = data[i][j];
 result[i][j] = atoi(&(chrs[i][j]));
 }
}

int main(){
  char d[4][3] = {"ab", "cd", "ef", "gh"};
  int i;
  for(i=0;i<4;i++){printf("%d:  %s", i, d[i]); }
  cudaMemcpyToSymbol(data, d, sizeof(char)*12);
  int ** result;
  cudaMalloc(result, sizeof(int)*8);
  solve<<<1, 4>>>(result);
  int r[4][2];
  cudaMemcpy(r, result, sizeof(int)*8, cudaMemcpyDeviceToHost);
  int j;
  for(i=0;i<4;i++){
    for (j=0;j<4;j++){
      printf("%d ", r[i][j]);
    }
    printf("\n");
  }
  return 0;
}
