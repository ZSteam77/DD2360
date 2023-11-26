#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <sys/time.h>

#define DataType float
void executeCommand(const char* command) {
    FILE* pipe = popen(command, "r");
    if (!pipe) {
        std::cerr << "popen failed!" << std::endl;
        return;
    }
    pclose(pipe);
}
// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < numARows && col < numBColumns) {
    DataType sum = 0.0;
    for (int i = 0; i < numAColumns; ++i) {
      sum += A[row * numAColumns + i] * B[i * numBColumns + col];
    }
    C[row * numBColumns + col] = sum;
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s numARows numAColumns numBColumns\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  int numARows = atoi(argv[1]);
  int numAColumns = atoi(argv[2]);
  int numBColumns = atoi(argv[3]);
  int numBRows = numAColumns;  // To make the matrix multiplication valid
  int numCRows = numARows;
  int numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  DataType *hostA = (DataType*)malloc(numARows * numAColumns * sizeof(DataType));
  DataType *hostB = (DataType*)malloc(numBRows * numBColumns * sizeof(DataType));
  DataType *hostC = (DataType*)malloc(numARows * numBColumns * sizeof(DataType));
  DataType *resultRef = (DataType*)malloc(numARows * numBColumns * sizeof(DataType));

  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows * numAColumns; ++i) {
    hostA[i] = rand() / (DataType)RAND_MAX;
  }

  for (int i = 0; i < numBRows * numBColumns; ++i) {
    hostB[i] = rand() / (DataType)RAND_MAX;
  }

  for (int i = 0; i < numARows * numBColumns; ++i) {
    resultRef[i] = 0.0;
  }

  for (int i = 0; i < numARows; ++i) {
    for (int j = 0; j < numAColumns; ++j) {
      for (int k = 0; k < numBColumns; ++k) {
        resultRef[i * numBColumns + k] += hostA[i * numAColumns + j] * hostB[j * numBColumns + k];
      }
    }
  }

  //@@ Insert code below to allocate GPU memory here
  DataType *deviceA, *deviceB, *deviceC;
  cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc((void**)&deviceC, numARows * numBColumns * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  struct timeval start, end;

  gettimeofday(&start, NULL);
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  gettimeofday(&end, NULL);

  double copyToDeviceTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1.0e6;

  //@@ Initialize the grid and block dimensions here
  dim3 blockSize(16, 16);
  dim3 gridSize((numBColumns + blockSize.x - 1) / blockSize.x, (numARows + blockSize.y - 1) / blockSize.y);

  //@@ Launch the GPU Kernel here
  gettimeofday(&start, NULL);
  gemm<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();  // Ensure that the kernel is complete before measuring time
  gettimeofday(&end, NULL);

  double kernelTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1.0e6;

  //@@ Copy the GPU memory back to the CPU here
  gettimeofday(&start, NULL);
  cudaMemcpy(hostC, deviceC, numARows * numBColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  gettimeofday(&end, NULL);

  double copyToHostTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1.0e6;

  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < numARows * numBColumns; ++i) {
    if (fabs(hostC[i] - resultRef[i]) > 1e-5) {
      fprintf(stderr, "Mismatch at index %d: %f != %f\n", i, hostC[i], resultRef[i]);
      break; // You may want to handle errors differently
    }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  printf("Copy to device time: %f seconds\n", copyToDeviceTime);
  printf("Kernel time: %f seconds\n", kernelTime);
  printf("Copy to host time: %f seconds\n", copyToHostTime);
    std::string plotCommand =
        "echo \"1 " + std::to_string(copyToDeviceTime) + " 'Copy to Device'\\n"
        "2 " + std::to_string(kernelTime) + " 'Kernel'\\n"
        "3 " + std::to_string(copyToHostTime) + " 'Copy to Host'\" | graph -C -m 0.2";

    // 执行生成图表的命令
    executeCommand(plotCommand.c_str());
  return 0;
}
