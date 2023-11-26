1. Compile both OMP and CUDA versions of your selected benchmarks. Do you need to make any changes in Makefile?
   
   We need to modify the rodinia_3.1/cuda/lud/cuda/Makefile if we want to run the lud.
   
   ```verilog
   remove: -arch=sm_13 \
   in  NVCCFLAGS +=  section
   ```
   
   For rodinia_3.1/cuda/b+tree/Makefile,
   
   ```verilog
   remove line CUDA_FLAG = -arch sm_20 and all variables related to CUDA_FLAG
   ```

2. Ensure the same input problem is used for OMP and CUDA versions. Report and compare their execution time.
   
   ![](C:\Users\Zhou\AppData\Roaming\marktext\images\2023-11-11-17-06-32-image.png)
   
   v

3. Do you observe expected speedup on GPU compared to CPU? Why or Why not?

     There's speedup for lud, 27.45ms is reduced to 3.21ms when running on gpu.

     For b+tree, 0.877602994442s for gpu and 0.017s for cpu.

     Therefore, not all algorithms could benefit from parallelism.
