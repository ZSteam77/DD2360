## EX3

1. Considering the given algorithm.
   
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
   
   With increasing dimX, Flops would obviously increase to a peak value,

2. 
