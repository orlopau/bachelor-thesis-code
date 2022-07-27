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

data_dir = args.get_args().data


@wandb_mixin
def train(config):
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    # merge configs and create runnable
    c = {**stress.config, **config}
    device = torch.device(f"cuda")
    runnable = stress.create_runnable(c, device, data_dir)

    runner = distributed.Runner(device, data_dir)
    runner.set_runnable(runnable, batch_size=c["batch_size"], workers=c["workers"])

    def hook(res):
        tune.report(acc_test=res["acc_test"], acc_train=res["acc_train"])
        wandb.log(res)

    runner.epoch_hooks.append(hook)
    runner.start_training(config["epochs"])


ss = {
    "batch_size": tune.grid_search([x*50 for x in range(1, 20)]),
    "wandb": {
        "project": "hpdlf",
        "settings": wandb.Settings(_stats_sample_rate_seconds=0.5, _stats_samples_to_average=2),
        "dir": "/lustre/ssd/ws/s8979104-horovod/data/wandb",
        "group": "bs_speedup",
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
        "cpu": 6,
        "gpu": 1
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
