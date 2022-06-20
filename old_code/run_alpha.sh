#!/bin/bash

ml modenv/hiera
ml fosscuda/2020b
ml scikit-learn/0.23.2
ml TensorFlow/2.4.1

python3 /lustre/ssd/ws/s8979104-workspace-1/network_cnn_dense.py $1 $2 $3 $4 $5 $6


