# Parallelization of GPUs based on Horovod - an Analysis of Scaling and Performance Based on an Application from Material Sciences

This repository contains the accompanying code for the thesis. 


It contains the neural network code for the thesis, as well as the scripts to
run the experiments on an HPC cluster using SLURM. The original problem comes
from the field of material sciences [1], the dataset was enhanced with
artificial data in [2].

Code to generate the plots from data is also included.

## References

[1] Matthias Zscheyge, Robert Böhm, Andreas Hornig, Johannes Gerritzen, and Maik
    Gude. Rate-dependent non-linear mechanical behaviour of continuous
    fibre-reinforced thermoplastic composites – experimental characterisation
    and viscoelastic-plastic damage modelling. Materials & Design, 193:108827,
    2020

[2] Peter Winkler, Norman Koch, Andreas Hornig, and Johannes Gerritzen. Omniopt
    – a tool for hyperparameter optimization on HPC. In Heike Jagode, Hartwig
    Anzt, Hatem Ltaief, and Piotr Luszczek, editors, High Performance Computing,
    pages 285–296, Cham, 2021. Springer International Publishing.
