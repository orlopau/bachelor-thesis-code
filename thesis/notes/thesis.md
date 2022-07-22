1. Neural Networks
2. Distributed Training of NN
    1. Model parallelism
    2. Data parallelism
    3. Pipelining
    4. Hybrid Parallelism
    5. Parameter/Gradient compression
    6. Frameworks
       1. PyTorch
       2. DeepSpeed
       3. TensorFlow
       4. Horovod
       5. Ray
       6. Fairscale
       7. 
3. Example from Material Science
4. Taurus
5. Horovod on Taurus
   1. Scaling analysis
      1. Batch size optimization
   2. Performance analysis

* speedup between: -> amdahls law graph
  * worker number for dataloader
  * pin memory
  * data on gpu
  * mixed precision
  * multi gpu
  * multi node
* batch size for test calculation: max out size of GPU mem
* put all data into GPU mem instead of host mem to avoid moving it to gpu