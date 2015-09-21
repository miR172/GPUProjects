#include <stdio.h>
#include <stdlib.h>

#define ITERATION 500

typedef unsigned long long bint;

float ** allocate(bint n){
  float **m = (float **)calloc(n+1, sizeof(float*));

  bint i;
  for (i=0; i<n+1; i++)
    m[i] = (float *)calloc(n+1, sizeof(float));

  //initialize
  for (i=0; i<n+1; i++){
    m[0][i] = 80;
    m[n][i] = 80;
    m[i][0] = 80;
    m[i][n] = 80;
  }

  for (i=10; i<=30; i++)
    if (i<n+1)
      m[0][i] = 150;

  return m;

}

void simulate(float **s, float **d, int dim){
  int i,j;
  for (i = 1; i<dim-1; i++){
    for (j = 1; j<dim-1; j++){
      d[i][j] = (s[i-1][j]+s[i+1][j]+s[i][j-1]+s[i][j+1])/4;
    }
  }
  return;
}

float avg(float **m, int dim){
  float sum = 0;

  int i,j;
  for (i=0; i<dim; i++){
    for (j=0; j<dim; j++){
      sum += (float)m[i][j];
      //printf("%f ", m[i][j]);
    }
    //printf("\n");
  }
  //printf("\n");
  return sum/dim/dim;
}


int main(int argc, char*argv[]){
  if (argc < 2){
    printf("Please indicate matrix size.\n");
    exit(0);
  }

  bint n = atoi(argv[1]);
  //printf("size (%llu+1)*(%llu+1) ; ", n, n);

  float **a = allocate(n);
  float **b = allocate(n);

  //float mean = avg(a, n+1);
  //printf("%f===>", mean);

  //simulate
  int i;
  for (i=0; i<ITERATION/2; i++){
    simulate(a, b, n+1);
    simulate(b, a, n+1);
  }
  if (ITERATION%2 !=0){
    simulate(a, b, n+1);
    //mean = avg(b, n+1);
    //printf("%f\n", mean);
  }
  else{
    //mean = avg(a, n+1);
    //printf("%f\n", mean);
  }

  //free
  bint ii;
  for (ii=0; ii<n+1; ii++){
    free(a[ii]);
    free(b[ii]);
  }

  free(a);
  free(b);

  return 0;
}
