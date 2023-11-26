#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <math.h>

// timing related
struct timeval startTime;
struct timeval endTime;
#define START_TIMER gettimeofday(&startTime, NULL)
#define END_TIMER gettimeofday(&endTime, NULL)
#define DURATION (int)(endTime.tv_sec * 1000000 + endTime.tv_usec - \
                       (startTime.tv_sec * 1000000 + startTime.tv_usec))

// Used to handle cuda error
#define CUDA_CALL(F)                                    \
    if ((F) != cudaSuccess)                             \
    {                                                   \
        printf("Error at line %d: %s\n", __LINE__,      \
               cudaGetErrorString(cudaGetLastError())); \
        exit(-1);                                       \
    };
#define PRINT_TIMER(F)                                   \
    printf("%s %d\n",F, DURATION);
