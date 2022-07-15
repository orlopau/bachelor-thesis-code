import stress
from utils import distributed
from utils import args
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import torch

data_dir = args.get_args().data


def train(config):
    # merge configs and create runnable
    c = {**stress.config, **config}
    device = torch.device("cuda:0")
    runnable = stress.create_runnable(c, device, data_dir)

    runner = distributed.Runner(device, data_dir)
    runner.set_runnable(runnable, batch_size=c["batch_size"], workers=c["workers"])
    runner.epoch_hooks.append(lambda res: tune.report(acc_test=res["acc_test"], acc_train=res["acc_train"]))
    runner.start_training(50)


ss = {
    "lr": tune.grid_search([1e-7, 1e-6, 1e-5, 1e-4]),
    "lr_max": tune.sample_from(lambda spec: spec.config.lr * 100),
    "batch_size": 7500,
    "log_sys_usage": True
}

try:
    ray.init(address="auto")
except Exception:
    print("error initializing ray")

reporter = tune.CLIReporter(max_progress_rows=16,
                            print_intermediate_tables=True,
                            sort_by_metric=True,
                            infer_limit=20)
analysis = tune.run(train,
                    config=ss,
                    resources_per_trial={
                        "cpu": 4,
                        "gpu": 1
                    },
                    num_samples=1,
                    progress_reporter=reporter,
                    metric="acc_test",
                    mode="min",
                    fail_fast=True,
                    verbose=3)

best_experiment = analysis.get_best_trial()
print(best_experiment)
