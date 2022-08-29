#!/bin/bash

# Fetch all groups
PARTITIONS="gpu2 hpdlf alpha"
WB_GROUPS="srun_mpi_big srun_nccl_big srun_mpi srun_nccl"
# iterate over GROUPS
for WB_GROUP in $WB_GROUPS; do
    for PARTITION in $PARTITIONS; do
        # iterate over PARTITIONS
        echo "Fetching $WB_GROUP/$PARTITION"
        python runs_store.py orlopau/$PARTITION $WB_GROUP
    done
done