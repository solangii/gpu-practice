import torch

a = torch.tensor([1., 2., 3.])

print(torch.square(a))
print(a ** 2)
print(a * a)

def time_pytorch_function(func, input):
    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize() # wait until finish current job
    return start.elapsed_time(end)

b = torch.randn(10000, 10000).cuda()

def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

time_pytorch_function(torch.square, b)
time_pytorch_function(square_2, b)
time_pytorch_function(square_3, b)

print("=============")
print("Profiling torch.square")
print("=============")

# Now profile each function using pytorch profiler
with torch.profiler.profile() as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

with torch.profiler.profile() as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

with torch.profiler.profile() as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


"""
tensor([1., 4., 9.])
tensor([1., 4., 9.])
tensor([1., 4., 9.])
=============
Profiling torch.square
=============
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------
                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------
           aten::square         0.21%       2.987ms       100.00%        1.402s        1.402s             1
              aten::pow        99.78%        1.399s        99.79%        1.399s        1.399s             1
      aten::result_type         0.00%      31.000us         0.00%      31.000us      15.500us             2
       aten::empty_like         0.00%      14.000us         0.00%      45.000us      45.000us             1
    aten::empty_strided         0.00%      31.000us         0.00%      31.000us      31.000us             1
         aten::can_cast         0.00%       0.000us         0.00%       0.000us       0.000us             1
               aten::to         0.00%       2.000us         0.00%       2.000us       2.000us             1
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.402s

=============
Profiling a * a
=============
---------------  ------------  ------------  ------------  ------------  ------------  ------------
           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
---------------  ------------  ------------  ------------  ------------  ------------  ------------
      aten::mul        98.87%     962.000us       100.00%     973.000us     973.000us             1
    aten::empty         1.13%      11.000us         1.13%      11.000us      11.000us             1
---------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 973.000us

=============
Profiling a ** 2
=============
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------
                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------
              aten::pow        73.68%      70.000us       100.00%      95.000us      95.000us             1
      aten::result_type        11.58%      11.000us        11.58%      11.000us       5.500us             2
       aten::empty_like         5.26%       5.000us        13.68%      13.000us      13.000us             1
    aten::empty_strided         8.42%       8.000us         8.42%       8.000us       8.000us             1
         aten::can_cast         0.00%       0.000us         0.00%       0.000us       0.000us             1
               aten::to         1.05%       1.000us         1.05%       1.000us       1.000us             1
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 95.000us
"""
