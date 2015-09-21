# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>
# include "sudoku.h"

# define N 10 //population size
# define BLKN 1 //grid size
# define M_RATE 0.6 //mutation rate

void free_exit(FILE *f, FILE *sf, int ** pool, char * fname_out, sudoku_puzzle *s);

void shuffle(int * a, int n);

int ** scan_puzzle(sudoku_puzzle *s, int pool_size);

int main(int argc, char *argv[]){

  FILE *f, *sf;
  f = fopen(argv[1], "r");

  if (f==NULL){
    DEBUG("usage: sudokusolver filename.in");
    exit(0);
  }

  int fname_size = strlen(argv[1]);
  char *fname_out = malloc(fname_size+1);
  char *temp = ".out";
  memcpy(fname_out, argv[1], fname_size-3);
  memcpy(fname_out+fname_size-3, temp, 4);

  sf = fopen(fname_out, "w");
  if (sf==NULL){
    DEBUG("unable to create output file\n");
    exit(0);
  }

  sudoku_puzzle *s = ini_puzzle(0, f);
  int ** pool = scan_puzzle(s, N);

  /*
  cudaError_t error;

  // constant mem
  __constant__ int map2chr[9][9]; //position on pannel to chromosome index
  __constant__ int colpre[9][9]; //column present bits
  __constant__ int rowpre[9][9]; //row present bits, val-1 --> 1 or 0
  __constant__ int blk2chr[9][2]; //blk2chr[i] is [i_start, i_length] of i th block


  error = cudaMemcpyToSymbol(blk2chr, s->blk2chr, 9*2*sizeof(int), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess){ free_exit(f, sf, pool, fname_out, s);}
  printf("start index for blks on chr on constant\n");

  error = cudaMemcpyToSymbol(map2chr, s->map2chr, 9*9*sizeof(int), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess){ free_exit(f, sf, pool, fname_out, s);}

  error = cudaMemcpyToSymbol(colpre, s->colpre, 9*9*sizeof(int), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess){ free_exit(f, sf, pool, fname_out, s);}

  error = cudaMemcpyToSymbol(rowpre, s->rowpre, 9*9*sizeof(int), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess){ free_exit(f, sf, pool, fname_out, s);}

  printf("9*9 2D panel on constant: \n\tpannel_position --mapping--> chromosome_index\n\tpresent bits by column & row\n");

  // global mem  
  int ** pp;
  int pitch = 0;
  int ** final;

  error = cudaMallocPitch((void **) pp, &pitch, s->chr_size*sizeof(int), N);
  if (error != cudaSuccess){ free_exit(f, sf, pool, fname_out, s);}

  error = cudaMemcpy2D(pp, pitch, pool, s->chr_size*sizeof(int), s->chr_size*sizeof(int), N);
  if (error != cudaSuccess){ free_exit(f, sf, pool, fname_out, s);}

  error = cudaMalloc((void **) final, s->chr_size*sizeof(int));
  if (error != cudaSuccess){ free_exit(f, sf, pool, fname_out, s);}

  printf("allocated %d*%d 2D population pool, pitch = %d\n", s->chr_size, N, pitch);
  
  // init with random numbers, curand library
  curandState * allStates;
  cudaMalloc(&allStates, N*sizeof(curandState));
  ini_device<<<BLKN, 3*N>>>(allStates, time(NULL));
  */
  // call solve_kernel
   
  export_sudoku(s, sf);

  free_exit(f, sf, pool, fname_out, s);
  return 0;
}

void free_exit(FILE *f, FILE *sf, int ** pool, char * fname_out, sudoku_puzzle *s){
  fclose(f);
  fclose(sf);

  int i = 0;
  for (;i<N;i++){ free(pool[i]); }
  free(pool);

  free(fname_out);
  clean_sudoku(s);
  exit(0);
}


void shuffle(int * array, int size){
  if (size>1){
    int i=0;
    for (; i<size-1; i++){
      int j = i + rand()/(RAND_MAX/(size-i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

int ** scan_puzzle(sudoku_puzzle *s, int pool_size){
  // scan and generate arrays
  int i, j;
  int ** map = malloc(sizeof(int *)*9);
  int ** map_chr = malloc(sizeof(int *)*9);
  int chr_size = 0;

  for (i=0; i<9; i++){
    // for each block

    int all[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int n = 0; // missing in this blocks

    for (j=0; j<9; j++){
      if (s->blocks[i][j].on){ // present
        all[s->blocks[i][j].val-1] = 0; // set present to 0
      }else{ // missing
        int r = s->blocks[i][j].i;
        int c = s->blocks[i][j].j;
        s->map2chr[r][c] = chr_size + n;
        n++;
      }
    }

    map[i] = malloc(sizeof(int)*n); // missing
    map_chr[i] = malloc(sizeof(int)*2); //start, length
    map_chr[i][0] = i==0?0:map_chr[i-1][0] + map_chr[i-1][1];
    map_chr[i][1] = n; 
    chr_size += n;

    int k = 0;
    for (j=0; j<9; j++){
      if (all[j] != 0){ // non-0->missing
        map[i][k] = all[j];
        k++;
      }
    }
  }
                                                               
  // randomize initial chromosomes pool

  srand(time(NULL));
  int ** pool = malloc(sizeof(int*)*pool_size);
  for (i=0; i<N; i++){ 
    pool[i] = malloc(sizeof(int)*chr_size);
    for (j=0; j<9; j++){
      // printf("block %d size %d chrstart %d\n", j, map_chr[j][1], map_chr[j][0]);
      shuffle(map[j], map_chr[j][1]);
      memcpy(pool[i]+map_chr[j][0], map[j], map_chr[j][1]*sizeof(int));
    }
  }

  s->map = map;
  s->blk2chr = map_chr;
  s->chr_size = chr_size;

  // debug
  for(i=0;i<pool_size;i++){
    printf("chromosome %d: ",i);
    for(j=0;j<chr_size;j++){
      printf("%d",pool[i][j]);
    }
    printf("\n");
  }
  for (i=0;i<9;i++){
    for (j=0;j<9;j++){
      printf("%d", s->map2chr[i][j]);
    }
    printf("\n");
  }
  printf("==>map2chr\n\n");

  for (i=0;i<9;i++){
    for (j=0;j<9;j++){
      printf("%d", s->colpre[i][j]);
    }
    printf("\n");
  }
  printf("==>colpre\n\n");

  for (i=0;i<9;i++){
    for (j=0;j<9;j++){
      printf("%d", s->rowpre[i][j]);
    }
    printf("\n");
  }
  printf("==>rowpre\n\n");

  for (i=0;i<9;i++){
    printf("blk %d start %d length %d", 
           i, s->blk2chr[i][0], s->blk2chr[i][1]);
    printf("\n");
  }
  printf("==>blk2chr\n\n");

  // debug


  return pool;
}








