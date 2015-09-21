#include <stdio.h>
#include <math.h>
int main(){
  unsigned long long dim = 6;
  unsigned long long r;
  int x = 15;
  r = x + dim + 1 + 2*(x/(dim-2));
  printf(" id %d in dim %llu is %llu\n", x, dim, r);
}
