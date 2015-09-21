#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>

#define TIME 500 //# of iterations
#define BLKSIZE 24
#define DEBUG(s) {printf("peek "); printf(s); printf("\n");}
//#define DEBUG(s)

typedef unsigned long long bint;


__global__ void simulate(float *src, float* des, bint dim){

  __shared__ float add[TIME+1][BLKSIZE];

  //x, y location of thread - to MEM space
  bint x =  threadIdx.x; 
  bint y =  threadIdx.y + blockIdx.x*blockDim.y;
  bint id = threadIdx.x*(dim-2) + threadIdx.y + blockIdx.x*blockDim.y;
  float v = src[id]/4; 
  
  //initialize
  if (x>0){
    add[x][y] = 0;
  }

  __syncthreads();


  //load each v to up, left, right, down positions
  if (x < TIME)
    add[x+1][y] = v;

  if (x > 0)
    add[x-1][y] = v;

  if (y%BLKSIZE > 0) //has sth on left
    add[x][y%BLKSIZE-1] = v;
  else if (y > 0)
    des[id-1] = v; //global

  if (y%BLKSIZE < BLKSIZE-1) //has sth on right
    add[x][y%BLKSIZE+1] = v;
  else if (y < dim-3)
    des[id+1] = v; //global

  __syncthreads();

  // GMT once for all
  if ((x > 0) && (y < dim-2))
    des[id] += add[x][y];
  
}

__global__ void assembly(float *d1, float *d2, float *m, bint dim){
  __shared__ float tmp[TIME+1];

  bint x =  threadIdx.x; 
  bint y =  threadIdx.y;
  bint id1 = threadIdx.x*(dim-2) + threadIdx.y;
  bint id2 = (threadIdx.x+1)*(dim-2) - threadIdx.y - 1; //upright box

  tmp[x] = d2[x]; //global load to shared

  __syncthreads();

  // GMT
  m[y] = tmp[y];
  if (y < TIME){
    d1[id1] += tmp[y+1];
    d1[id2] += tmp[y+1];
    d2[id1] += tmp[y+1];
    d2[id2] += tmp[y+1];
    m[dim-2-y] = tmp[y+1];
  }

}


float * config(bint dim){

  //allocate on host and initialize
  float *bar1 = (float *)calloc((dim-2)*(TIME+1), sizeof(float)); //side with 150
  float *bar2 = (float *)calloc((dim-2)*(TIME+1), sizeof(float)); //side all 80

  bint p;
  for (p=0; p < dim-2; p++){
    bar1[p] = 80;
    bar2[p] = 80;
    if ((p>=10) && (p<=30)){
      bar1[p] = 150;
    }
  }

  //config kernel

  dim3 blkdim;
  blkdim.x = TIME+1;
  blkdim.y = BLKSIZE;
  bint griddim = ceil((double)(dim-2)/BLKSIZE);

  //allocate on kernel
  bint mem = (dim-2)*(TIME+1)*sizeof(float);
  float *src1, *des1;
  cudaMalloc((void **)&src1,  mem);
  cudaMalloc((void **)&des1,  mem);
  cudaMemcpy(src1, bar1, mem, cudaMemcpyHostToDevice);
  cudaMemcpy(des1, bar1, mem, cudaMemcpyHostToDevice);

  float *src2, *des2;
  cudaMalloc((void **)&src2,  mem);
  cudaMalloc((void **)&des2,  mem);
  cudaMemcpy(src2, bar2, mem, cudaMemcpyHostToDevice);
  cudaMemcpy(des2, bar2, mem, cudaMemcpyHostToDevice);

  DEBUG("loaded")

  free(bar1);
  free(bar2);

  //launch
  bint i;
  for (i=0; i<TIME; i++){
    if (i%2==0){
      simulate<<<griddim, blkdim>>>(src1, des1, dim);
      simulate<<<griddim, blkdim>>>(src2, des2, dim);
    }else{
      simulate<<<griddim, blkdim>>>(des1, src1, dim);
      simulate<<<griddim, blkdim>>>(src2, des2, dim);
  }}

  // clean up
  float *d1, *d2;
  if (TIME%2==0){ //result in src
    cudaFree(des1);
    cudaFree(des2);
    d1 = src1;
    d2 = src2;
  }
  else{
    cudaFree(src1);
    cudaFree(src2);
    d1 = des1;
    d2 = des2;
  }
  DEBUG("simulated")

  //assembly
  float *mid;
  cudaMalloc((void **)&mid, dim*sizeof(float)); //result for middle lines

  dim3 blk;
  blk.x = TIME+1;
  blk.y = (TIME%32==0) ? TIME : TIME+32-TIME%32; //first 32n >= TIME
  assembly<<<1, blk>>>(d1, d2, mid, dim);
  DEBUG("assemblied")

  /*
  //cpu assembly
  float *m = (float *)malloc(dim*dim*sizeof(float));

  bint unit = (dim-2)*sizeof(float);
  for (i=0; i<TIME+1; i++){
    m[i*dim] = 80;
    cudaMemcpy(&m[i*dim+1], &d1[i*(dim-2)], unit, cudaMemcpyDeviceToHost);
    m[(i+1)*dim-1] = 80;
  }
  cudaFree(d1);
      //done line 0 to TIME

  float *middlelines = (float *)malloc(dim*sizeof(float));
  cudaMemcpy(middlelines, mid, dim*sizeof(float), cudaMemcpyDeviceToHost);
  for (i=TIME+1; i<dim-1-TIME; i++){
    middlelines[i] = 0; //no temperature for the middle region
  }
  middlelines[dim-1] = 80;
  for (i=TIME+1; i<dim-1-TIME; i++){
    memcpy(&m[i*dim], middlelines, dim*sizeof(float));
  }
  cudaFree(mid);
      //done for TIME+1...dim-TIME-1

  for (i=dim-1-TIME; i<dim; i++){
    m[i*dim] = 80;
    cudaMemcpy(&m[i*dim+1], &d2[i*(dim-2)], unit, cudaMemcpyDeviceToHost);
    m[(i+1)*dim-1] = 80;
  }

  cudaFree(d2);
      //done dim-TIME-1...dim-1

  return m;
  */
  cudaFree(d1);
  cudaFree(d2);
  cudaFree(mid);

  return NULL;
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


int main(int argc, char *argv[]){
        //getDeviceProp();

  if (argc < 2){
    printf("Please indicate matrix size.\n");
    exit(0);
  }
  bint n = atoi(argv[1]);

  float *x = config(n+1);
  if (x != NULL){
    float mean = avg(x, n+1);
    printf("peek mean: %f\n", mean);
    free(x);
  }

  return 0;
}
