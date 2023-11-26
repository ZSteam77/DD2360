1. How do you launch the code on GPU on Dardel supercomputer?
   
   1. Connect to Dardel according to the tutorial
   
   2. Upload cpp and makefile through scp command
   
   3. Compile remotely on Dardel
   
   4. ```tcl
      salloc -A edu23.dd2360 -p gpu -N 1 -t 00:10:00
      srun -n 1 HelloWorld
      ```

2. Include a screenshot of your output from Dardel

![](C:\Users\Zhou\AppData\Roaming\marktext\images\2023-11-12-20-49-36-image.png)


