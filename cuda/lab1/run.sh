#!/bin/bash

# gcc sequential.c -o sequential
# nvcc cuda.cu -o cuda
# nvcc cuda-optimized.cu -o ocuda

#rm result.txt

#for SIZE in 100 500 1000 10000
#do
#  time ./sequential $SIZE #>> result.txt
#  echo $SIZE #>>result.txt
#  echo >> result.txt
#done

#echo "-----------GPU-------------"

for SIZE in 100 500 1000 10000 100000 1000000
do
  time ./cuda $SIZE 
#  echo "optimized"
#  time ./ocuda $SIZE 
  echo $SIZE 
done

