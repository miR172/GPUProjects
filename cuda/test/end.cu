#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void solve(){
 __shared__ int going;
 int id = threadIdx.x;
 going = -1;
 while (going < 0){
   if (id==7){
     going = id;
     return;
   }
 }
}

int main(){
  solve<<<1, 32>>>();
  return 0;
}
