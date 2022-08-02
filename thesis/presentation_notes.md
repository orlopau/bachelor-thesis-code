* welcome to interim presentation of bachelor thesis: Parallelization of GPUs based on Horovod – A Scaling and Performance Analysis based on an Application from Material Sciences
* ml models, especially NN models increase in size
  * example: Language models, almost 10 times more parameters per year
  * increased training time
  * problematic even for smaller models for HP tuning
  * data or model won't fit onto single gpu
* using multiple gpus over multiple nodes for training
  
* **NEXT**
  
* multiple approaches for parallelism
* data parallelism, useful when whole model fits in memory
  * worker is a single gpu, tpu or cpu
  * data is split into partitions
  * each worker has a single partition
  * whole forward / backward pass can be computed in parallel
  * synchronization of gradients when parameters are updated
  * synchronizations per epoch is equal to batches per epoch -> normally, one parameter update per batch
* **NEXT**
* model parallelism, useful for models that don't fit in memory
  * each worker holds all data, but only part of model
  * model split into partitions
  * parallelism is highly dependant on network topology
  * communication happens everytime a neuron output is an input to another neuron compuation
  * example: bold black lines neuron outputs, must be copmmunicated between workers
  * neurons without bold line as input can be computed in parallel
* **NEXT**
* choosing model vs data parallelism
* what is communicated? parameter gradients and neuron outputs
* decide by what is more expensive to compute
* example: 
  * small network, small amount of parameters but big dataset: high amount of computing time per parameter
  * many passes over same parameters -> high amount of computing time per parameter
  * big network, same size dataset: passes take longer, higher amount of computing time per parameter
* data parallelism easier to implement, model parallelism must take into account network topology
* model parallelism only choice when model too large for gpu memory
* in summary: for models that fit into memory -> data parallelism
* model too large for gpu -> model parallelism

* **NEXT**
* many frameworks for parallel training of NNs
* easy to use, efficient: horovod
* created by uber 2017 to solve in-house problems with TF distributed
* tf distributed uses parameter servers to average computed gradients
* development difficult, more HPs to optimize: number of parameter servers
* extra boilerplate code to setup parameter servers
* slow due to bottleneck parameter servers -> communication of all workers with parameter server to synchronize gradients
* originally written for tf
* now supports most major frameworks
* easy setup for data parallelism -> 6 changed lines in PyTorch
* model parallelism not supported, can be achieved with custom implementation
* inspired by fork of tf by baidu, using ring-allreduce algorithm for gradient synchronization
* ring-allreduce -> workers arranged in ring, speed limited by slowest connection in ring
* bandwidth-optimal but not latency optimal 
* ring-allreduce implemented in mpi and nccl
* 88% efficient in scaling training of Inception V3, ResNet-101

* **NEXT**
* Practical example from material sciences
* prediction of material parameters from strain stress curves
* labelled datapoints
* each point has 2 curves, strain and stress with 200 points
* flag indicating if values were padding
* label: material parameters G0 and a
* output was normalized to range 0-1
* accuracy is mse of g0 and a
* 1M labelled data points
* ~4GB size
* **NEXT**
* simple CNN, 2 conv layers with max pooling, 2 fully connected layers
* HPs optimized using automatic HP tuning via OmniOpt
* small network, only 153tsd parameters, 0.61MB size
* batch size and LR given as baseline
* large dataset, small network that easily fits into memory -> data parallelism

**NEXT**
* training on different partitions (alpha,gpu2,hpdlf)
* nccl and mpi used
* nccl advantages to mpi:
  * nccl uses NVLink, PCIe and shared memory to directly communicate between GPUs
  * no copy to cpu memory needed when gpus are connected by NVLink or PCI
  * automatic detection of topology
  * uses network interfaces for inter-node communication
  * Sockets and remote direct memory access
  * compatible with mpi
* **NEXT**
* resources used for training
* nodes used exclusively for reproducability of runtimes and system metrics
* number cpus used for training equals max cpus div by num gpus
* example: 8 gpus, 48 cpus (alpha), 6 cpus per gpu -> 6 cpus per task
* all memory on node used (exclusive anyways)
* variable number of used GPUs
* NCCL used by default, mpi as comparison

**NEXT**
* speedup and efficiency of distributed training
* gpus distributed in sequence
* example: alpha: 1 node used until 8 gpus allocated, then next node allocated
* 10 gpus -> 8 on node 1, 2 on node 2
* sequential distribution, because inter-node communication is more expansive than intra node
* epoch time vs number gpus
* in absolute terms alpha fastest, to be expected, best performing gpus
* gpu2 faster than hpdlf interesting, k80 slower than 1080 ti in benchmarks
* good scaling, gpu2 nearly 8x reduction in epoch time for 8 gpus
* **NEXT**
* speedup/efficiency calculated in relation to base run on alpha without horovod
* overhead of horovod on one gpu is approximately 17%
* alpha best scaling behaviour, gpu2 scales better than hpdlf
* alpha bump at 9 gpus, when extra node is used
* hpdlf bump each 3 gpus, 3 gpus per node
* scaling approximately linear
* dotted lines show fitted regression line
* 0.45 alpha, ...
* mpi similar characteristics, small speedup due to using nccl
* data transmitted is small
* gpu2 no mpi because infiniband did not work with MPI, skewed results
* **not close to the promised 88% scaling efficiency**
* why?
* **NEXT**
* profiling of training loop
* hpdlf run, 2 nodes, 6 gpus
* 28% of time used for gradient synchronization
* data loading no bottleneck, 2% of time
* data is not on hard drive but in ram
* could even fit into gpu memory -> possible optimization, small gain
* why gradient synchronization time so large?
* **NEXT**
* overhead for communication
* plot single synchronization time vs number of gpus in ms
* hpdlf shows abrupt jumps in synchronization overhead when new node is used
* overall, synchronization time increases with number of gpus used
* steeper gradient for hpdlf, higher communication costs
* flatter curve for hpdlf and alpha
*  **NEXT**
*  number of synchronizations per epoch directly related to number of batches per epoch
*  number of batches adjusted by batch size
*  on 2 workers, even 1ms communication overhead results in 5s overhead per epoch -> 15% of training time
*  **NEXT**
*  But, number of batches decreases with number of gpus used
*  Data is split, less batches per epoch -> batches calculated synchronously
*  Less synchronizations
*  Bacthes per epoch dotted line
*  Correlated to synchronization time per epoch
*  -> number of batches decreases more than duration per synchronization increases per additional gpu
*  **NEXT** 
*  MPI more efficient for single node
*  nccl more efficient intra node
*  jumps for mpi when new another node is used
*  -> normally, nccl should be faster beause it uses inter-GPU communication like NVLink or PCIe
*  not as efficient for small tensors, see benchmark
*  **NEXT** System Usage
*  Low GPU usage due to:
   *  small batch size
   *  small network size
   *  cpu-gpu communication overhead significant
*  Interesting effect for NCCL:
   *  Using NCCL, the gpu usage jumps when the first additional node is used, i.e. when inter-node communication is used
   *  Has no impact on speedup
   *  Not visible in GPU power usage
   *  Maybe artifact of NCCL waiting for communication
*  **NEXT**
*  Jump in GPU Usage not visible for MPI
*  GPU Usage decreases as number of GPUs increases, possibly due to gradient synchronization overhead
*  No GPU usage when gradients are synchronized via MPI
*  **NEXT**
*  Optimizations
*  Low GPU Usage, no increase in network size expected
*  Increasing batch size
   *  Performance profits from higher GPU usage due to better internal parallelization -> bigger matrices, efficiently parallelized on gpus
   *  Less synchronizations per epoch, lower synchronization overhead
   *  But: worse accuracy with the same HPs, new HP search, time for HP search
   *  Possible generalization gap when very large batch sizes are used
   *  Good accuracy on training, but bad accuracy on test set
*  Move dataset into GPU memory
   *  Enough space on GPU for whole datset, smallest GPU memory on hpdf is 11GB
   *  But probably only 2% maximum performance gain -> as seen when Profiling
*  **NEXT**
*  Test batch size increase on 1 GPU to find optimal batch size
*  sharp decrease in time per epoch, until batch size 700, then time levels off
*  same for all partitions used
*  speedup factor of 6 to 7 for batch size 700 on 1 gpu
*  better speedup than using 10 GPUs with effective batch size 750, but only speedup of factor 5
*  **NEXT**
*  worse accuracy on test set, baseline is blue line in first diagram
*  logarithmic scale
*  currently, accuracy of batch size 700 is order of magnitude worse than 75
*  increase LR, because effective batch size is batch size multiplied by gpu count gradients are averaged
*  -> less gradient updates per epoch, because less batches per epoch
*  goal: same learning progression and accuracy per epoch, fully benefit from speedup
*  better prediction of the true gradient due to more sample in batch should allow higher LR
*  **NEXT**
*  same data, but plotted vs time passed since start
*  looks fairly similar for all batches
*  more epochs but no speedup w.r.t. test accuracy
*  same accuracy after time
*  goal: faster convergence
*  **NEXT**
*  next steps
*  train same time not epochs with increased batch size
*  adjust learning rate according to heuristic, e.g. scale learning linearly with batch size
*  optimize hp for large batches and distributed training
