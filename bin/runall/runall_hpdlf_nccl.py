from runall import run

def _sbatch_generator(config):
    prefix = f"""\
#!/bin/bash

# alpha: 48 cores, 8 gpus, ~8GB per core, 10312M -> 6 cores per gpu
# gpu2: 24 cores, 4 gpus, ~2.5G per core, 2583M -> 6 cores per gpu
# hpdlf: 12 cores, 3 gpus, 7916M per core, 3 gpus, 4 cores per GPU

#SBATCH --nodes={config["nodes"]}
#SBATCH --ntasks-per-node={config["gpus"]}
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --gres="gpu:{config["gpus"]}"
#SBATCH --time=4:00:00
#SBATCH --exclusive=user
#SBATCH -p hpdlf
#SBATCH -o /lustre/ssd/ws/s8979104-horovod/sbatch/hpdlf/sbatch_%j.log
#SBATCH -J runall_hpdlf_nccl
{"" if config["nodelist"] is None else f"#SBATCH --nodelist {config['nodelist']}"}

WS_PATH=/lustre/ssd/ws/s8979104-horovod
VENV=$WS_PATH/venv_hvd_nccl

source $VENV/bin/activate
"""

    if config["sequential"]:
        return prefix + f"""\
for i in `seq 1 {int(config["gpus"]) * int(config["nodes"])}`;
do
    mpirun -np $i \\
        -bind-to none -map-by slot \\
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_P2P_DISABLE=0 -x NCCL_IB_DISABLE=1 {"-x HOROVOD_TIMELINE=$WS_PATH/timeline.data" if config["timeline"] else ""}\\
        -mca pml ob1 -mca btl ^openib --mca btl_tcp_if_include ib0,ib1 \\
        $VENV/bin/python -u {config["script_path"]} --data $WS_PATH/data --dist --group {config["group"]} \\
        --project hpdlf {"" if config["name"] is None else f"--name {config['name']}"}
done
"""

    else:
        return prefix + f"""\
mpirun -N {config["gpus"]} \\
    -bind-to none -map-by slot \\
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_P2P_DISABLE=1 -x NCCL_IB_DISABLE=1 {"-x HOROVOD_TIMELINE=$WS_PATH/timeline.data" if config["timeline"] else ""}\\
    -mca pml ob1 -mca btl ^openib --mca btl_tcp_if_include ib0,ib1 \\
    $VENV/bin/python -u {config["script_path"]} --data $WS_PATH/data --dist --group {config["group"]} \\
    --project hpdlf {"" if config["name"] is None else f"--name {config['name']}"}
"""

run(_sbatch_generator)