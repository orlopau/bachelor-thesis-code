import pathlib
import pickle
import sys
import wandb
import pandas as pd

api = wandb.Api(timeout=300)
max_samples = 999999


def get_named_run(project, name):
    runs = list(filter(lambda x: x.name == name, api.runs(project)))
    if len(runs) != 1:
        raise Exception("No base run found, must have name 'base'")

    return runs[0]


def exclude_map_keys(map, keys):
    return {k: map[k] for k in map.keys() if not any(e in k for e in keys)}


def run_to_dict(run, exclude_keys=["gradients", "parameters"]):
    summary = exclude_map_keys(run.summary, exclude_keys)
    # delete wandb from summary, because it is not serializable
    if "_wandb" in summary:
        del summary["_wandb"]

    config = run.config
    history = pd.DataFrame.from_dict([exclude_map_keys(h, exclude_keys) for h in run.scan_history()])
    history_metrics = run.history(samples=max_samples, stream="system", pandas=True)

    return {
        "id": run.id,
        "summary": summary,
        "config": config,
        "history": history,
        "history_metrics": history_metrics
    }


def dump_runs(runs, path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "wb") as f:
        pickle.dump(runs, f)


def load_runs(path):
    p = pathlib.Path(path)
    with open(p, "rb") as f:
        return pickle.load(f)


def run_to_reduced_dict(run):
    config = run["config"]
    summary = run["summary"]
    history = run["history"]
    history_metrics = run["history_metrics"]

    history_metrics_start = int(summary["_runtime"] - summary["time"]) + 5
    history_metrics = history_metrics.iloc[history_metrics_start:-5]

    return {
        "epoch":
            summary["epoch"],
        "nodes":
            int(config["slurm"]["SLURM_NNODES"]) if "slurm" in config else 1,
        "gpus":
            config["horovod"]["size"] // int(config["slurm"]["SLURM_NNODES"]) if "horovod" in config else 1,
        "size":
            config["horovod"]["size"] if "horovod" in config else 1,
        "batch_size":
            config["run"]["batch_size"] if "run" in config else config["batch_size"],
        "history":
            history,
        "history_metrics":
            history_metrics
    }


def reduce_run(run):
    config = run["config"]
    summary = run["summary"]
    history = run["history"]
    history_metrics = run["history_metrics"]

    history = history.rename(columns={"time_step": "time_sync_sum", "time_step_avg": "time_sync"})

    mean_metrics = ["time_sync", "time_sync_sum"]
    min_metrics = ["acc_test", "acc_train", "time_train", "time_epoch", "time_train", "time_test"]

    entry = {
        "epoch":
            summary["epoch"],
        "nodes":
            int(config["slurm"]["SLURM_NNODES"]) if "slurm" in config else 1,
        "gpus":
            config["horovod"]["size"] // int(config["slurm"]["SLURM_NNODES"]) if "horovod" in config else 1,
        "size":
            config["horovod"]["size"] if "horovod" in config else 1,
        "batch_size":
            config["run"]["batch_size"] if "run" in config else config["batch_size"],
        **history[mean_metrics][4:].add_suffix("_mean").median().to_dict(),
        **history[min_metrics][4:].add_suffix("_min").median().to_dict(),
        **history[min_metrics][4:].add_suffix("_act_min").min().to_dict(),
    }

    gpu_index = 0
    if "gpu" in summary:
        gpu_index = summary["gpu"]

    history_metrics_start = int(summary["_runtime"] - summary["time"]) + 5
    history_metrics = history_metrics.iloc[history_metrics_start:-5]
    entry["gpu_power"] = history_metrics[f"system.gpu.{gpu_index}.powerPercent"].mean()
    entry["gpu_usage"] = history_metrics[f"system.gpu.{gpu_index}.gpu"].mean()
    entry["gpu_mem"] = history_metrics[f"system.gpu.{gpu_index}.memoryAllocated"].mean()
    entry["gpu_mem_usage"] = history_metrics[f"system.gpu.{gpu_index}.memory"].mean()

    return entry


def reduce_runs(runs):
    """
    Reduces a run list into a single dataframe by reducing the history data of each run.
    """
    df = pd.DataFrame.from_dict([reduce_run(run) for run in runs])
    return df


if __name__ == "__main__":
    # dump runs in a group to a file
    runs = api.runs(sys.argv[1], {"group": sys.argv[2]}, per_page=500)

    path = (pathlib.Path(__file__).parent.parent / "data/runs" / sys.argv[1] /
            (sys.argv[2] + ".pickle")).resolve()

    print(f"dumping to {path}...")
    run_dicts = [run_to_dict(r) for r in runs]
    print(f"fetched {len(run_dicts)} runs")
    dump_runs(run_dicts, path)