import math
import stress
from utils import distributed
from utils import args
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.wandb import (
    WandbTrainableMixin,
    wandb_mixin,
)
import torch
import wandb
import os
import numpy as np

data_dir = args.get_args().data


@wandb_mixin
def train(config):
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    # merge configs and create runnable
    c = config["run"]
    device = torch.device(f"cuda")

    (model, optimizer, scheduler) = stress.create_model(c)
    (dataset_train, dataset_test, y_max, y_min) = stress.create_datasets(data_dir)
    (loader_train, loader_test) = distributed.create_loaders(dataset_train, dataset_test, c["batch_size"],
                                                             c["batch_size"], c["workers"], c["pin_memory"])
    runnable = stress.StressRunnable(model, optimizer, loader_train, loader_test, device, y_max, y_min,
                                     scheduler)
    runner = distributed.Runner()
    runner.runnable = runnable

    def hook(res):
        tune.report(acc_test=res["acc_test"], acc_train=res["acc_train"])
        wandb.log({**res, "gpu": os.environ["CUDA_VISIBLE_DEVICES"]})

    runner.epoch_hooks.append(hook)
    runner.start_training(c["epochs"])


ss = {
    "run": {
        **stress.config,
        "batch_size": tune.grid_search(np.arange(75, 5000, 100, dtype=int).tolist()),
        "lr": tune.sample_from(lambda spec: math.sqrt(spec.config["run"]["batch_size"] / 75) * 4.7331e-04),
        "epochs": 100,
    },
    "slurm": distributed.slurm_meta(),
    "wandb": {
        "project": args.get_args().project,
        "settings": wandb.Settings(_stats_sample_rate_seconds=0.5, _stats_samples_to_average=2),
        "dir": "/lustre/ssd/ws/s8979104-horovod/data/wandb",
        "group": "batch_sqrt",
    },
}

print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

try:
    ray.init(address="auto")
except Exception:
    print("error initializing ray")

reporter = tune.CLIReporter(max_progress_rows=16,
                            print_intermediate_tables=True,
                            sort_by_metric=True,
                            infer_limit=20)
analysis = tune.run(
    train,
    config=ss,
    resources_per_trial={
        "cpu":
            int(distributed.slurm_meta()["SLURM_CPUS_PER_TASK"]) /
            len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
        "gpu":
            1,
    },
    num_samples=1,
    # scheduler=ASHAScheduler(max_t=80, grace_period=30),
    progress_reporter=reporter,
    metric="acc_test",
    mode="min",
    fail_fast=True,
    verbose=3)

best_experiment = analysis.get_best_trial()
print(best_experiment)
