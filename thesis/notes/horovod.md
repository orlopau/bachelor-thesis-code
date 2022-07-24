# Horovod

* Gradients aggregated via allreduce
* Gradient compression used to reduce network load
* OpenMPI 4.1.1 doesn't work with horovod
* OpenMPI 3.1.6 works

trick damit horovodrun auf slurm klappt: 

* horovod kann nur TCP über MPI nicht RDMA -> siehe doku
* horovod auf gpu2 über infiniband crashed am ende (bug in openmpi)
* vergleich infiniband <-> normales ethernet -> runs in gruppe network
* kleine batch size -> viele gradient updates (da pro batch) -> latenz des netzwerks!