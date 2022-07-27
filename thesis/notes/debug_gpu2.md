Funktioniert mit `WANDB_MODE=disabled mpirun -N 3 -bind-to none --oversubscribe -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_IB_DISABLE=0 -x NCCL_SHM_DISABLE=0 -x NCCL_P2P_DISABLE=1 -x NCCL_P2P_LEVEL=NVL  -mca pml ob1 -mca btl ^openib python code/test.py`.

GPUs sind teilweise mit PIX verbunden, dies funktioniert nicht mit NCCL (Grund unbekannt!), deshalb P2P deaktivieren.

Vllt noch mal bandwidth test: https://stackoverflow.com/questions/69693950/error-some-nccl-operations-have-failed-or-timed-out

```
        GPU0    GPU1    GPU2    GPU3    mlx4_0  CPU Affinity    NUMA Affinity
GPU0     X      PIX     SYS     SYS     SYS     0-11    0
GPU1    PIX      X      SYS     SYS     SYS     0-11    0
GPU2    SYS     SYS      X      PIX     PHB     12-23   1
GPU3    SYS     SYS     PIX      X      PHB     12-23   1
mlx4_0  SYS     SYS     PHB     PHB      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```