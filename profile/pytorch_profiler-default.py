import torch
from torch.profiler import profile, record_function, ProfilerActivity


# ## Default way to use profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        a = torch.square(torch.randn(10000, 10000).cuda())

prof.export_chrome_trace("pytorch_profiler/default_trace.json")
