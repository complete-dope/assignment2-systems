ATen: A Tensor library ( this library is foundation of pytorch and has many tensor operation in it)

Cutlass (cuda template for linear algebra subroutines): This is a library written by nvidia folks that has code in ptx lang and is highly efficient for linear algebra 

Fused kernel is made only if the generated PTX / SASS shows fusion

To understand / see what went wrong in your CUDA kernel you need to actually go out and use the `os.environ['CUDA_LAUNCH_BLOCKING'] = 1` this helps in debugging the cuda kernels ! 




Assignment 2 :1.1.5

Modern gpu contains specialised GPU cores for accelerating matrix multiplication at lower precision.

we do loss scaling when doing ops in lower precision like fp16, bf16 , so loss scaling is simply that loss is multiplied by a scaling factor

Furthermore, FP16 has a lower dynamic range than FP32, which can lead to overflows that manifest as a NaN loss. Full bfloat16 training is generally more stable (since BF16 has the same dynamic range as FP32), but can still affect final model performance compared to FP32

in pytorch this mixed precision training is done using torch.autocast 
matrix multiplications are done in low precisions but accumulation and reduction operations are done in the full float32 

```bash
with torch.autocast(device = 'cuda', dtype = dtype):
    y = model(x)
```

As alluded to above, it is generally a good idea to keep accumulations in higher precision even if the
tensors themselves being accumulated have been downcasted. ( so operations like reduction and accumulation)


so underlying values in fp16,  should do operations and report back to fp32 so avoid getting precision errors 




So there are 2 things to profile in a code, one is memory profiler and other one is time profiler ( time profiler we have seen with the benchmarking and torch.profiler)

Recording / Getting CUDA memory history :  `torch.cuda.memory._record_memory_history`


> DO THE REQUIRED MEMORY PROFILING AND SEE WHERE ALL CAN BE SAVE THE MEMORY BANDWIDTH 

Continue with 1.3.1 that is the example on Weighted Sum on how to implement that using the triton's kernels ! 


