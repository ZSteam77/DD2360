## EX1

1. Assume X=800 and Y=600. Assume that we decided to use a grid of 16X16 blocks. That is, each block is organized as a 2D 16X16 array of threads. How many warps will be generated during the execution of the kernel? How many warps will have control divergence? Please explain your answers.
   
   Ans:
   
   ```verilog
   For warp = 32,
   Block number = 800/16 * 600/16 = 50 * 38 = 1900
   warp number = 1900*16*16/32 = 15200
   ```
   
   since each warp would be responsible for 32/16 = 2 rows, considering the blocks with blockIdx.y = 600/16 = 37, half of these threads in the blocks would run the if line code, but since they're distributed along the y axis, 4 warps will run the if statement, 4 warps would run the else statement, no control divergence.

2. Now assume X=600 and Y=800 instead, how many warps will have control divergence? Please explain your answers.
   
   In this case, there would be 800/16 = 50 blocks with 50 * 16 * 16 / 32 = 400 warps have control divergence since they would be divided along x-axis.

3. Now assume X=600 and Y=799, how many warps will have control divergence? Please explain your answers.
   
   Ans:
   
   Similar to the previous one, still 400 warps.

# 
