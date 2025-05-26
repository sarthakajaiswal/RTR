// header files 
// standard headers 
#include <stdio.h> 

// cuda headers 
#include <cuda.h> 
#include "helper_timer.h" 

// global variable  
// const int iNumberOfArrayElements = 5; 
const int iNumberOfArrayElements = 11444777; 

float* hostInput1 = NULL; 
float* hostInput2 = NULL; 
float* hostOutput = NULL; 
float* gold = NULL; 

float* deviceInput1 = NULL; 
float* deviceInput2 = NULL; 
float* deviceOutput = NULL; 

float timeOnCPU = 0.0f; 
float timeOnGPU = 0.0f; 

// CUDA Kernel 
__global__ void vecAddGPU(float* in1, float* in2, float* output, int len) 
{
    // code 
    int i = blockIdx.x + blockDim.x + threadIdx.x; 

    if(i < len) 
    {
        output[i] = in1[i] + in2[i]; 
    }
}

// entry-point function 
int main(void) 
{
    // function declarations 
    void fillFloatArrayWithRandomNumbers(float*, int); 
    void vecAddCPU(const float*, const float*, float*, int); 
    void cleanup(void); 

    // variable declarations 
    int size = iNumberOfArrayElements * sizeof(float); 
    cudaError_t result = cudaSuccess; 

    // code 
    // memory allocation on host 
    hostInput1 = (float*)malloc(size); 
    if(hostInput1 == NULL) 
    {
        printf("Host Memory allocation failed for hostInput1 array.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    hostInput2 = (float*)malloc(size); 
    if(hostInput2 == NULL) 
    {
        printf("Host memory allocation failed for hostInput2 array.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    hostOutput = (float*)malloc(size); 
    if(hostOutput == NULL) 
    {
        printf("Host memory allocation failed for hostOutput array.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    gold = (float*)malloc(size); 
    if(gold == NULL) 
    {
        printf("Host memory allocation falied for gold array.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    // filling values into host arrays 
    fillFloatArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements); 
    fillFloatArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements); 

    // device arrays memory allocation 
    result = cudaMalloc((void**)&deviceInput1, size); 
    if(result != cudaSuccess) 
    {
        printf("Device memory allocation failed for deviceInput1 array.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    result = cudaMalloc((void**)&deviceInput2, size); 
    if(result != cudaSuccess) 
    {
        printf("Device memory allocation failed for deviceInput2 array.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    result = cudaMalloc((void**)&deviceOutput, size); 
    if(result != cudaSuccess) 
    {
        printf("Device memory allocation failed for deviceOutput array.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    // copy data from host arrays into device arrays 
    result = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice); 
    if(result != cudaSuccess) 
    {
        printf("Host to device data copy is failed for deviceInput1 array.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    result = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice); 
    if(result != cudaSuccess) 
    {
        printf("Host to device data copy failed for deviceInput2 array.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    // CUDA kernel configuration 
    dim3 dimGrid = dim3((int)ceil((float)iNumberOfArrayElements / 256.0f), 1, 1); 
    dim3 dimBlock = dim3(256, 1, 1); 

    // CUDA kernel for vector addition 
    StopWatchInterface* timer = NULL; 
    sdkCreateTimer(&timer); 
    sdkStartTimer(&timer); 

    vecAddGPU<<<dimGrid, dimBlock>>> (deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements); 

    sdkStopTimer(&timer); 
    timeOnGPU = sdkGetTimerValue(&timer); 
    sdkDeleteTimer(&timer); 
    timer = NULL; 

    // copy data from device to host arrays into host arrays 
    result = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost); 
    if(result != cudaSuccess) 
    {
        printf("Device to host data copy is failed for hostOutput array.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    // vector addition on host 
    vecAddCPU(hostInput1, hostInput2, gold, iNumberOfArrayElements); 

    // comparison 
    const float epsilon = 0.000001f; 
    int breakValue = -1; 
    bool bAccuracy = true; 
    for(int i = 0; i < iNumberOfArrayElements; ++i) 
    {
        float val1 = gold[i]; 
        float val2 = hostOutput[i]; 
        if(fabs(val1 - val2) > epsilon) 
        {
            bAccuracy = false; 
            breakValue = i; 
            break; 
        }
    }

    char str[128]; 
    if(bAccuracy == false) 
        sprintf(str, "Comparison of CPU and GPU vector addition is not within accuracy of 0.000001 at array index %d", breakValue); 
    else 
        sprintf(str, "Comparison of CPU and GPU vector addition is within accuracy of 0.000001"); 

    // output 
    printf("Array1 begins from 0th index %.6f to %dth index %.6f\n", 
            hostInput1[0], iNumberOfArrayElements - 1, hostInput1[iNumberOfArrayElements - 1]); 
    printf("Array2 begins from 0th index %.6f to %dth index %.6f\n", 
            hostInput2[0], iNumberOfArrayElements - 1, hostInput2[iNumberOfArrayElements - 1]); 
    printf("CUDA Kernel Grid Dimention = %d,%d,%d and Block Dimention = %d,%d,%d\n", 
            dimGrid.x, dimGrid.y, dimGrid.z, 
            dimBlock.x, dimBlock.y, dimBlock.z); 
    printf("Output array begins from 0th index %.6f to %dth index %.6f\n", 
            hostOutput[0], iNumberOfArrayElements - 1, hostOutput[iNumberOfArrayElements - 1]); 
    printf("Time taken for vector addition on CPU = %.6f\n", timeOnCPU); 
    printf("Time taken for vector addition on GPU = %.6f\n", timeOnGPU); 
    printf("%s\n", str); 

    // cleanup 
    cleanup(); 

    return (0); 
}

void fillFloatArrayWithRandomNumbers(float* arr, int len) 
{
    // code 
    const float fscale = 1.0f / (float)RAND_MAX; 
    for(int i = 0; i < len; ++i) 
    {
        arr[i] = fscale * rand(); 
    }
}

void vecAddCPU(const float* arr1, const float* arr2, float* out, int len) 
{
    // code 
    StopWatchInterface* timer = NULL; 
    sdkCreateTimer(&timer); 
    sdkStartTimer(&timer); 

    for(int i = 0; i < len; ++i) 
    {
        out[i] = arr1[i] + arr2[i]; 
    }

    sdkStopTimer(&timer); 
    timeOnCPU = sdkGetTimerValue(&timer); 
    sdkDeleteTimer(&timer); 
    timer = NULL; 
}

void cleanup(void) 
{
    if(hostOutput)
    {
        free(hostOutput); 
        hostOutput = NULL; 
    }

    if(hostInput2)
    {
        free(hostInput2); 
        hostInput2 = NULL; 
    }

    if(hostInput1) 
    {
        free(hostInput1); 
        hostInput1 = NULL; 
    }

    if(gold) 
    {
        free(gold); 
        gold = NULL; 
    }

    if(deviceOutput)
    {
        cudaFree(deviceOutput); 
        deviceOutput = NULL; 
    }

    if(deviceInput2)
    {
        cudaFree(deviceInput2); 
        deviceInput2 = NULL; 
    }

    if(deviceInput1) 
    {
        cudaFree(deviceInput1); 
        deviceInput1 = NULL; 
    }
}
