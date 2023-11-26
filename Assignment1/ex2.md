## Task 2

1. The screenshot of the output from running deviceQuery test in /1_Utilities.
   
   ![](D:\Homework\KTH\DD2650\Assignment1\ex2.png)

2. What is the Compute Capability of your GPU device?
   
   7.5

3. The screenshot of the output from running bandwidthTest test in /1_Utilities.
   
   ![](D:\Homework\KTH\DD2650\Assignment1\ex2_4.png)

4. How will you calculate the GPU memory bandwidth (in GB/s) using the output from deviceQuery? (Hint: memory bandwidth is typically determined by clock rate and bus width, and check what double date rate (DDR) may impact the bandwidth). Are they consistent with your results from bandwidthTest?
   
   ```verilog
   With DDR : 2 * 256 * 5001 * 10^6 = 320GB/s
   Without DDR : 160GB/s
   Both figures have a about 100 GB/s difference in bandwidth with 
   the Device to Device Bandwidth from the test
   ```
