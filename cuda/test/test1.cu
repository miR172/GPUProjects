# include <cuda.h>
# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <curand_kernel.h>
# define N 10
# define CUDA_ERROR_CHECK(error) {\
           e = error; \
           if (e != cudaSuccess){ \
             printf("%s\n", cudaGetErrorString(e)); \
             exit(0); \
           }    \
         }

__constant__ int a[2][2];


__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
} 

__global__ void generate( curandState* globalState ) 
{
    int ind = threadIdx.x;
    curandState localState = globalState[ind];
    int i = 0;
    while (i<10){
    float RANDOM = curand_uniform( &localState );
    i++;
    }
    globalState[ind] = localState; 
}

int main( int argc, char** argv) 
{
      cudaError_t e;
  int ** c_a = (int **)malloc(sizeof(int*)*2);
  c_a[0] = (int*)malloc(sizeof(int)*2);
  c_a[1] = (int*)malloc(sizeof(int)*2);
  c_a[0][0] = 1;
  c_a[0][1] = 2;
  c_a[1][0] = 4;
  c_a[1][1] = 8;
  printf("c_a: [%d][%d] [%d][%d]\n", c_a[0][0], c_a[0][1], c_a[1][0], c_a[1][1]);
  CUDA_ERROR_CHECK(cudaMemcpyToSymbol(a, c_a, 2*2*sizeof(int)))
  dim3 tpb(N,1,1);
  curandState* devStates;
  cudaMalloc ( &devStates, N*sizeof( curandState ) );
    
  // setup seeds
  setup_kernel <<< 1, tpb >>> ( devStates, time(NULL) );
  
  // generate random numbers
  generate <<< 1, tpb >>> ( devStates );

  return 0; 
}
