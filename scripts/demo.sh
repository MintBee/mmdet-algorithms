#!/bin/bash

algorithms=('detecto-rs' 'mask2former' 'solo' 'solov2' 'yolact')

for alg in "${algorithms[@]}"
do
  python ../demo.py $alg
done
