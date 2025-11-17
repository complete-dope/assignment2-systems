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