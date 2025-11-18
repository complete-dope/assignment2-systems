ATen: A Tensor library ( this library is foundation of pytorch and has many tensor operation in it)

Cutlass (cuda template for linear algebra subroutines): This is a library written by nvidia folks that has code in ptx lang and is highly efficient for linear algebra 

Fused kernel is made only if the generated PTX / SASS shows fusion

To understand / see what went wrong in your CUDA kernel you need to actually go out and use the `os.environ['CUDA_LAUNCH_BLOCKING'] = 1` this helps in debugging the cuda kernels ! 




