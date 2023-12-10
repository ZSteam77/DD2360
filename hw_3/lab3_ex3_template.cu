#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
    //@@ Insert code below to compute histogram of input using shared memory and atomics

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    extern __shared__ unsigned int sharedBins[];

    // Initialize shared memory bins
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        sharedBins[i] = 0;
        // printf("%d/n",i);
    }
    __syncthreads();

    // Compute local histogram using shared memory
    for (int i = tid; i < num_elements; i += stride) {
        atomicAdd(&sharedBins[input[i]], 1);
    }
    __syncthreads();

    // Update global histogram using atomics
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&bins[i], sharedBins[i]);
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
    //@@ Insert code below to clean up bins that saturate at 127

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Clean up bins that exceed the saturation limit
    for (int i = tid; i < num_bins; i += stride) {
        if (bins[i] > 127) {
            atomicExch(&bins[i], 127);
        }
    }
}

int main(int argc, char **argv) {
    int inputLength = 10000; // Set your desired input length
    unsigned int *hostInput, *hostBins, *resultRef;
    unsigned int *deviceInput, *deviceBins;

    //@@ Insert code below to read in inputLength from args
    // ...

    printf("The input length is %d\n", inputLength);

    //@@ Insert code below to allocate Host memory for input and output
    hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
    hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
    resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

    //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, NUM_BINS - 1) ;

    for (int i = 0; i < inputLength; ++i) {
        hostInput[i] = distribution(generator);
    }

    //@@ Insert code below to create reference result in CPU
    for (int i = 0; i < NUM_BINS; ++i) {
        resultRef[i] = 0;
    }

    for (int i = 0; i < inputLength; ++i) {
        resultRef[hostInput[i]]++;
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void **)&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));

    //@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    //@@ Insert code to initialize GPU results
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    //@@ Initialize the grid and block dimensions here
    dim3 blockDim(256); // You can adjust the block size
    dim3 gridDim((inputLength + blockDim.x - 1) / blockDim.x);

    //@@ Launch the GPU Kernel here
    histogram_kernel<<<gridDim, blockDim, NUM_BINS * sizeof(unsigned int)>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
    // for(int i = 0;i < NUM_BINS;++i){
    //     printf("%d",deviceBins[i]);
    // }

    //@@ Initialize the second grid and block dimensions here
    dim3 convertGridDim((NUM_BINS + blockDim.x - 1) / blockDim.x);

    //@@ Launch the second GPU Kernel here
    convert_kernel<<<convertGridDim, blockDim>>>(deviceBins, NUM_BINS);
    

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    for (int i = 0; i < NUM_BINS; ++i) {
        // printf("ith element is: %u" , resultRef[i]);

        if (hostBins[i] != resultRef[i]) {
            printf("Mismatch at bin %d: Expected %u, Got %u\n", i, resultRef[i], hostBins[i]);
            break;
        }
    }
    FILE *fp;
    fp = fopen("histogram_result.txt", "w");
    for (int i = 0; i < NUM_BINS; ++i) {
        fprintf(fp, "%d\n", hostBins[i]);
    }
    fclose(fp);

    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    //@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}
