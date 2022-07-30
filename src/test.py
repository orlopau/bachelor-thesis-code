import torch
import horovod.torch as hvd

hvd.init()
torch.cuda.set_device(hvd.local_rank())

device = torch.device(f"cuda:{hvd.local_rank()}")

t = torch.rand(1, 2, device=device)
print(t)
print(hvd.allreduce(t))

print("finished!")