#!/bin/bash

ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4
ml TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4

python3 /lustre/ssd/ws/pwinkler-Pbopt/network_cnn_dense.py $1 $2 $3 $4 $5 $6


