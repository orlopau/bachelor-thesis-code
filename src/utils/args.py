import argparse

_args = None


def get_args():
    if _args is not None:
        return _args

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        help="path to data dir",
                        default="/home/paul/dev/bachelor-thesis-code/src/data")
    parser.add_argument("--small", help="specifiy if a small dataset should be used", action="store_true")
    parser.add_argument("--log-dir", help="sub dir under --data for logging results", default="logs")
    parser.add_argument("--no-test",
                        help="specify if no test acc should be calculated after each epoch",
                        action="store_true")
    parser.add_argument("--single-gpu",
                        help="specify if only a single gpu but multiple horovod processes should be used",
                        action="store_true")
    parser.add_argument("--group", help="group", default="run_group")
    parser.add_argument("--project", help="project", default="test")
    parser.add_argument("--name", help="name")
    parser.add_argument("--tprof", action="store_true")
    parser.add_argument("--dlprof", action="store_true")
    parser.add_argument("--lrf", help="lr finder", action="store_true")
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--cprof", action="store_true")
    parser.add_argument("--scorep", action="store_true")
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args()
    return args

