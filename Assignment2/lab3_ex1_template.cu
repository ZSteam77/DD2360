#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>


#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < len) out[index] = in1[index] + in2[index];
}

//@@ Insert code to implement timer start
#define START gettimeofday(&startTime, NULL)


//@@ Insert code to implement timer stop
#define END gettimeofday(&endTime, NULL)
#define DURATION (int)(endTime.tv_sec * 1000000 + endTime.tv_usec - (startTime.tv_sec * 1000000 + startTime.tv_usec))

int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  
  
  
  
  struct timeval startTime;
  struct timeval endTime;
  printf("The input length is %d\n", inputLength);
  //@@ Insert code below to allocate Host memory for input and output

  hostInput1 = (DataType *)malloc(inputLength * (int)sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputLength * (int)sizeof(DataType));
  hostOutput = (DataType *)malloc(inputLength * (int)sizeof(DataType));
  resultRef = (DataType *)malloc(inputLength * (int)sizeof(DataType));
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (unsigned i = 0; i < inputLength; ++i)
  {
      hostInput1[i] = (DataType) rand();
      hostInput2[i] = (DataType) rand();
      resultRef[i] = hostInput1[i] + hostInput2[i];
  }
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceInput1, inputLength * (int)sizeof(DataType));
  cudaMalloc((void **)&deviceInput2, inputLength * (int)sizeof(DataType));
  cudaMalloc((void **)&deviceOutput, inputLength * (int)sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  START;
  cudaMemcpy(deviceInput1, hostInput1, inputLength * (int)sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * (int)sizeof(DataType), cudaMemcpyHostToDevice);
  END;
  printf("Host copy to GPU time: %d\n", DURATION);

  //@@ Initialize the 1D grid and block dimensions here
  int blockSize = 256;
  int numBlocks = (inputLength + blockSize - 1) / blockSize;

  //@@ Launch the GPU Kernel here
  START;
  vecAdd<<<numBlocks, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  END;
  printf("GPU Kernel Launch time: %d\n", DURATION);
  //@@ Copy the GPU memory back to the CPU here
  START;
  cudaMemcpy(hostOutput, deviceOutput, inputLength * (int)sizeof(DataType), cudaMemcpyDeviceToHost);
  END;
  printf("GPU copy to Host time: %d\n", DURATION);

  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < inputLength; i++)
  {
    if (hostOutput[i] != resultRef[i]) printf("error at %dth element!!!!!!", i);
  }
  

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  
  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);
  return 0;
}
