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

* benchmark (https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#task-0-communication-backend-raw-performance) selbst implementieren mit horovod allreduce


* skaliert linear aber nicht perfekt
* liegt wahrscheinlich daran dass der Anteil der Zeit der Step Synchronisierung immer weiter ansteigt, d.h. die eigentliche Epoche wird kürzer, aber der Anteil der Step Synchronisierung steigt???

* horovod funktioniert nicht mit der aktuellsten NCCL version