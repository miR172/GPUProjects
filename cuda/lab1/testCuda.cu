#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


__global__ 
void testKernel(int *s, int dim){
  s[400] = 9;
}


int main(int argc, char *argv[]){

  if (argc < 2){
    printf("Please indicate matrix size.\n");
    exit(0);
  }

  int n = atoi(argv[1]);

  int * tm = (int *)calloc((n+1)*(n+1), sizeof(int));

  int j;
  for (j=0; j<(n+1)*(n+1); j++){
    printf("%d ", tm[j]);
  }
  printf("\n");

  int *dev;
  cudaMalloc((void **)&dev,  (n+1)*(n+1)*sizeof(int));
  cudaMemcpy(dev, tm, (n+1)*(n+1), cudaMemcpyHostToDevice);

  testKernel<<<2, 128>>>(tm, n+1);
  int *newtm = (int *)calloc((n+1)*(n+1),sizeof(int));
  cudaMemcpy(newtm, dev, (n+1)*(n+1), cudaMemcpyDeviceToHost);
  for (j=0; j<(n+1)*(n+1); j++){
    printf("%d ", newtm[j]);
  }
  printf("\n");
  free(tm);
  cudaFree(dev);
  free(newtm);
  return 0;
}
