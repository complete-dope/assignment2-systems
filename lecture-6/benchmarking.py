# here we are doing benchmarking for our codebase

from typing import Callable
import torch 
import time
import torch.nn as nn

# this is a decorator function
def benchmark(desc:str = "",warmup_steps:int =3)->Callable:
    print(desc)
    def _wrapper(call:Callable):
        def _run(): 
            for _ in range(warmup_steps):
                call()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            times = []
            for _ in range(10):
                start_time = time.time()
                call()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time =  time.time() 

                times.append(end_time - start_time)
            
            return sum(times)/len(times)
        return _run
    return _wrapper


def benchmark_function(desc:str,  call:Callable,warmup_steps = 3)->Callable:
    print(desc)
    for _ in range(warmup_steps):
        call()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(10):
        start_time = time.time()
        call()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time =  time.time() 

        times.append(end_time - start_time)
    
    return sum(times)/len(times)


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'

    return device


class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x

def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    # Define a model (with random weights)
    model = MLP(dim, num_layers).to(get_device())
    # Define an input (random)
    x = torch.randn(batch_size, dim, device=get_device())
    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        for _ in range(num_steps):
            # Forward
            y = model(x).mean()
            # Backward
            y.backward()
    return run


def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x, y)


if torch.cuda.is_available():
    dims = (1024, 2048, 4096, 8192, 16384)  # @inspect dims
else:
    dims = (1024, 2048)  # @inspect dims

matmul_results = [] 
# for dim in dims:
#     # @ inspect dim
#     result = benchmark_function(f"matmul(dim={dim})", run_operation2(dim=dim, operation=lambda a, b: a @ b))
#     matmul_results.append((dim, result))  # @inspect matmul_results

# print(matmul_results)

## profiler tells where the exact time is spent in the whole code block ! 
import torch 
from torch.profiler import profile, ProfilerActivity, record_function

def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    print('\n\n'+description)
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Run the code with the profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # Output stack trace for visualization
            with_stack=with_stack,
            # Needed to export stack trace for visualization
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    
    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    #text(f"## {description}")
    #text(table, verbatim=True)
    # Write stack trace visualization
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")
    return table

# sleep_function = lambda : time.sleep(50 / 1000)
# sleep_profile = profile("sleep", sleep_function) 
# print(sleep_profile)

# add_function = lambda a,b: a+b
# add_profile = profile("add", run_operation2(dim=2048, operation=add_function))
# print(add_profile)

# matmul_function = lambda a, b: a @ b
# matmul_profile = profile("matmul", run_operation2(dim=2048, operation=matmul_function))
# print(matmul_profile)


# gelu_function = lambda a, b: torch.nn.functional.gelu(a + b)
# gelu_profile = profile("gelu", run_operation2(dim=2048, operation=gelu_function))
# print(gelu_profile)

def manual_gelu(x:torch.Tensor):
    return 0.5 * x * (1+torch.nn.functional.tanh((2/torch.pi)**0.5 * (x + 0.044715*x*x*x)))

def pytorch_gelu(x: torch.Tensor):
    # Use the tanh approximation to match our implementation
    return torch.nn.functional.gelu(x, approximate="tanh")


x = torch.tensor([1.])

# now use the manual GeLU and the torch GeLU

y1 = pytorch_gelu(x)
y2 = manual_gelu(x)

print(y1,y2)
assert torch.allclose(y1, y2)

manual_gelu_profile= profile('manual_gelu', run_operation1(16384, manual_gelu))
print(manual_gelu_profile)


pytorch_gelu_profile = profile('pytorch_gelu', run_operation1(16384 , pytorch_gelu))
print(pytorch_gelu_profile)

manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time

if manual_time is not None and pytorch_time is not None:
    print('pytorch implementation is almost 10x faster that the manual one')

else:
    print("Could not compare times - benchmark results were None")



import triton
import triton.language as tl

def triton_gelu(x:torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    # allocate output tensor 
    y = torch.empty_like(x)

    num_elements= x.numel()
    block_size = 1024 # No. of threads in a block
    num_blocks = triton.cdiv(num_elements, block_size)
    
    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)
    return y

@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE:tl.constexpr):
    # input is at : x_ptr, output is at y_ptr 
    # the index of the threadBlock here we find that as pid
    # only main difference is triton works on vectors, whereas cuda code works on indexes of threads
    # Here I am working with blocks
    #        block-0     |      block-1      |      block-2
    #               BLOCK-SIZE          BLOCK-SIZE
    
    pid = tl.program_id(axis=0)  
    block_start = pid * BLOCK_SIZE

    #offset is a thread block , that tells these many threads are required   
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < num_elements

    # Read
    x = tl.load(x_ptr + offsets ,mask = mask) # load this whole vector of values in the vector x


    # Approx gelu
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    tl.store(y_ptr+offsets, y , mask = mask)


pytorch_triton_gelu = profile('triton_gelu', run_operation1(16384 , triton_gelu))
print(pytorch_triton_gelu) # this is faster than the torch implementation 

# GeLU is a nice element-wise operation but problem comes in aggregate operations ( operations that requires aggregation (sum , multiplication of all the values inside an array) that causes the whole problem)

# compiled method for gelu 
compiled_gelu = torch.compile(manual_gelu)
compiled_time = profile("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu)) 
print(compiled_time)
'''
# so we have seen 5 methods till now for writing kernels 

manual methods
pytorch method
cuda kernel method 
triton method 
compiled method 

'''
