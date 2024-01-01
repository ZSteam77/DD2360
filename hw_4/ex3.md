## EX3

1. Run the program with different dimX values. For each one, approximate the FLOPS (floating-point operation per second) achieved in computing the SMPV (sparse matrix multiplication). Report FLOPS at different input sizes in a FLOPS. What do you see compared to the peak throughput you report in Lab2?
   
   Considering the given algorithm.
   
   ```verilog
   for i from 1 to nsteps:
      tmp = A * temp
      temp = alpha * tmp + temp 
      norm = vectorNorm(tmp) 
      if tmp < 1e-4 : 
         break
   end
   ```
   
   ```verilog
    tmp = A * temp
   ```
   
   Requires 3*dimX - 6 FPs
   
   ```verilog
     temp = alpha * tmp + temp
   ```
   
   Requires 2*dimX FPs
   
   Norm operation requires 2*dimX FPs
   
   In total there're 7*dimX - 6 FPs per step.
   
   With increasing dimX, Flops would obviously increase to a peak value.

2. Run the program with dimX=128 and vary nsteps from 100 to 10000. Plot the relative error of the approximation at different nstep. What do you observe?
   
   ![](D:\Homework\KTH\DD2360\DD2360\hw_4\pic1.png)
   
   The y-axis is the error, x-axis is the nstep.
   
   The error falls as nstep increases.

3. Compare the performance with and without the prefetching in Unified Memory. How is the performance impact?

With prefetching: ![](D:\Homework\KTH\DD2360\DD2360\hw_4\pic2.png)

Without Prefetching:

![](D:\Homework\KTH\DD2360\DD2360\hw_4\pic3.png)

As nstep and runtime increases, the improvement in performance of having a prefetching is more obvious.
