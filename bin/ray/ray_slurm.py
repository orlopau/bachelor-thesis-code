#!/usr/bin/env python3

# A ray cluster is not easily deployable by slurm, because slurm by default can only deploy the same
# process on each node.
# Ray requires a head node and worker nodes, communicating over TCP/IP, thus requiring IP addresses and port numbers.
#
# To accomplish this, we retrieve the IP addresses of the allocated nodes, use node[0] as a head node,
# then start the worker nodes.
# In total, this script will spawn N+1 tasks, where N is the number of nodes.
#
# This script must be called with an available allocation, e.g. via salloc or sbatch.
#
# The first argument is executed as a command on the head node of the cluster.

import ray
import subprocess
import os
from time import sleep
import atexit
import sys


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


try:
    import ray
except ModuleNotFoundError:
    print("module ray not found, cant spawn ray cluster")


def subprocess_output_list(cmd):
    return subprocess.check_output(cmd, shell=True, timeout=120).decode().splitlines()


def subprocess_output(cmd):
    return subprocess.check_output(cmd, shell=True, timeout=120).decode().strip()


def cleanup():
    print(bcolors.HEADER + "cleaning up all steps")
    # dont kill *.extern step (usually the ssh session when interactive) and *.batch step (batch host job)
    step_ids = list(filter(lambda s: 'extern' not in s and 'batch' not in s,
                           subprocess_output_list("squeue --me -s -h -o %i")))
    print(f"killing jobs: {step_ids}")
    if len(step_ids) > 0:
        print(subprocess_output(f"scancel {' '.join(step_ids)}"))
    print(bcolors.ENDC)


# kill all step processes on exit
atexit.register(cleanup)

# retrieve node hostnames
hostnames = subprocess_output_list(
    "scontrol show hostnames $SLURM_JOB_NODELIST")

# get head node ip address
head_hostname = hostnames[0]
head_ip = subprocess_output(
    f"srun -N 1 --ntasks=1 -w {hostnames[0]} hostname -i")
head_port = 6379
head_address = f"{head_ip}:{head_port}"

print(f"head node: {hostnames[0]}, ip: {head_ip}")

slurm_cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
# TODO get gpus from gres, gpus per task doesnt work on taurus

slurm_gpus_per_task = len(os.environ.get("SLURM_JOB_GPUS").split(','))
print(f"N_GPUS: {slurm_gpus_per_task}")
# only allocations where each node has the same number of tasks are allowed, i.e. SLURM_TASKS_PER_NODE must have a form like "2(x5)"
slurm_tasks_per_node = int(os.environ.get(
    'SLURM_TASKS_PER_NODE').split('(')[0])

cmd_ray_head = f"ray start --head --node-ip-address={head_ip} --port={head_port} --num-cpus {slurm_cpus_per_task} --num-gpus {slurm_gpus_per_task} --block"
p_head = subprocess.Popen(
    f"srun -N 1 -J ray_head --ntasks=1 -w {head_hostname} {cmd_ray_head}", shell=True)
print("started ray head")

for i, host in enumerate(hostnames[1:]):
    cmd_ray_worker = f"ray start --address {head_address} --num-cpus {slurm_cpus_per_task} --num-gpus {slurm_gpus_per_task} --block"
    subprocess.Popen(
        f"srun -N 1 -J ray_worker_node_{i} --ntasks=1 -w {host} {cmd_ray_worker}", shell=True)
    print(f"started worker {i} on {host}")

ray_addr = f"ray://{head_ip}:10001"
print(f"ray address: {ray_addr}")

print("waiting for cluster...", end="")

ray.init(address=ray_addr)
while len(ray.nodes()) < len(hostnames):
    print(".", end="")
    sleep(2)

print(bcolors.OKGREEN + "\ncluster setup finished" + bcolors.ENDC)

# for some unknown reasson, we can not run another job via srun on the head node
# subprocess.run(
#     f"srun --ntasks=1 -w {head_hostname} --exclusive --overlap --cpus-per-task={slurm_cpus_per_task} {sys.argv[1]}")
# srun_cmd = f"srun -J ray_task --ntasks=1 -w {head_hostname} --exclusive --overlap --cpus-per-task={slurm_cpus_per_task} {sys.argv[1]}"
# ssh_cmd = f"ssh -o StrictHostKeyChecking=no {head_ip} {srun_cmd}"
# print(f"running command via ssh\n{bcolors.OKGREEN}{ssh_cmd}{bcolors.ENDC}")
# subprocess.run(f"{ssh_cmd} {sys.argv[1]}", shell=True, check=True)

current_host = subprocess_output("hostname")
print(
    f"{bcolors.HEADER}Running cmd on {current_host}: {sys.argv[1]}{bcolors.ENDC}")

# prevent wrong ordering on shell redirection
sys.stdout.flush()
subprocess.run(sys.argv[1], shell=True, check=True)
