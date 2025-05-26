// header files 
// standard headers 
#include <stdio.h> 

// cuda headers 
#include <cuda.h> 
#include "cuda_timer.h" 

// macros 
#define BLOCK_WIDTH 32 

// global variables 
float* hostA = NULL; 
float* hostB = NULL; 
float* hostC = NULL; 
float* gold = NULL; 

float* deviceA = NULL; 
float* deviceB = NULL; 
float* deviceC = NULL; 

float timeOnCPU = 0.0f; 
float timeOnGPU = 0.0f; 

// cuda kernel function 
__global__ void matMulGPU(int* A, int* B, int* C, int numARows, int numAColumns, int numBColumns, int numCColumns)  
{
    // variable declarations 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int column = blockIdx.x * blockDim.x + threadIdx.x; 

    // code 
    if((row < numARows) && (column < numBColumn)) 
    {
        int value = 0.0f; 
        for(int k = 0; k < numAColumns, ++k) 
        {
            int a = A[row * numAColumns + k]; 
            int b = B[k * numBColumns + column]; 

            value += a * b; 
        }
        c[row * numCColumns + column] = value; 
    }
}

// enrty-point function 
int main(int* argc, char* argv[]) 
{
    // function declarations 
    void InitA(int* data, int, int); 
    void InitB(int* data, int, int); 
    void matMulCPU(int*, int*, int*, int, int, int, int); 
    void cleanup(void); 

    // variable declarations 
    int numARows = BLOCK_WIDTH; 
    int numAColumns = BLOCK_WIDTH; 
    int numBRows = BLOCK_WIDTH; 
    int numBColunms = BLOCK_WIDTH; 

    int numCRows = numARows; 
    int numCColumn = numBColumn; 

    int numGoldRows = numARows; 
    int numGoldColumn = numBColumn; 

    int sizeA = numARows * numAColumns * sizeof(int); 
    int sizeB = numBRows * numBColumns * sizeof(int); 
    int sizeC = numCRows * numCColumns * sizeof(int); 
    int sizeGold = numGoldRows * numGoldColumns * sizeof(int); 

    cudaError_t result = cuda_Success; 

    // host memory allocation 
    hostA = (int*)malloc(sizeA); 
    if(hostA == NULL) 
    {
        printf("Memory allocation failed for hostA matrix.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    hostB = (int*)malloc(sizeB); 
    if(hostB == NULL) 
    {
        printf("Memory allocation failed for hostB matrix.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    hostC = (int*)malloc(sizeC); 
    if(hostC == NULL) 
    {
        printf("Memory allocation failed for hostC matrix.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    gold = (int*)malloc(sizeGold); 
    if(gold == NULL) 
    {
        printf("Memory allocation failed for gold matrix.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    // printing matrix dimentions 
    printf("The Dimention for matrix 'hostA' are    : %d x %d\n", numARows, numAColumns); 
    printf("The Dimention for matrix 'hostB' are    : %d x %d\n", numBRows, numBColumns); 
    printf("The Dimention for matrix 'hostC' are    : %d x %d\n", numCRows, numCColumns); 
    printf("The Dimention for matrix 'gold' are     : %d x %d\n", numGoldRows, numGoldColumns); 

    printf("Size of matrix 'hostA' = %d\n", sizeA); 
    printf("Size of matrix 'hostB' = %d\n", sizeB); 
    printf("Size of matrix 'hostC' = %d\n", sizeC); 

    printf("Size of matrix 'gold' = %d\n", sizeGold); 

    // fill source matricex 
    initA(hostA, numARows, numAColumns); 
    initB(hostB, numBRows, numBColumns); 

    // device memory allocation 
    result = cudaMalloc((void**)&deviceA, sizeA); 
    if(result != cuda_Success) 
    {
        printf("Device memory allocation failed for 'deviceA' matrix.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    result = cudaMalloc((void**)&deviceB, sizeB); 
    if(result != cuda_Success) 
    {
        printf("Device memory allocation failed for 'deviceB' matrix.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    result = cudaMalloc((void**)&deviceC, sizeC); 
    if(result != cuda_Success) 
    {
        printf("Device memory allocation failed for 'deviceC' matrix.\n"); 
        cleaup(); 
        exit(EXIT_FAILURE); 
    }

    // copy data from host 
    result = cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice); 
    if(result != cuda_Success) 
    {
        printf("Host to Device data copy is failed for deviceA matrix.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    result = cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice); 
    if(result != cuda_Success)
    {
        printf("Host to Device data copy is failed for deviceB matrix.\n"); 
        cleanup(); 
        exit(EXIT_FAILURE); 
    }

    // CUDA kernel configuration 
    dim3 dimGrid = dim3(ceil((int)numBColumns/(int)BLOCK_WIDTH), ceil((int)numARows/(int)BLOCK_WIDTH)); 
    dim3 dimBlock = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1); 
    
    // CUDA kernel for matrix multiplication 
    StopWatchInterface* timer = NULL; 
    sdkCreateTimer(&timer); 
    sdkStartTimer(&timer); 

    matMulGPU <<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numAColumns, numBColumns, numCColumns); 

    sdkStopTimer(&timer); 
    timeOnGPU = sdkGetTimerValue(&timer); 
    sdkDeleteTimer(&timer); 
    timer = NULL; 

    // copy data from device matrix into host matrix 
    result = cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost); 
    if(result != cuda_Success) 
    {
        printf("Device to Host data copy is failed 'hostC' matrix.\n"); 
        cleaup(); 
        exit(EXIT_FAILURE); 
    }

    // matrix multiplication on host 
    matMulCPU(hostA, hostB, gold, numARows, numAColumns, numBColumns, numCColumns); 

    // comparison 
    int breakValue = -1; 
    bool bAccuracy = true; 

    for(int i = 0; i < numCRows * numCColumns; ++i) 
    {
        int val1 = gold[i]; 
        int val2 = host[i]; 
        if(val1 != val2) 
        {
            bAccuracy = false; 
            breakValue = i; 
            break; 
        }
    }

    char str[128]; 
    if(bAccuracy == false) 
        sprintf(str, "Comparison of CPU and GPU matrix multiplication at array index %d", breakValue); 
    else 
        sprintf(str, "Comparison of CPU and GPU matrix multiplication matrix is accurate.\n"); 

    printf("Time taken for matrix multiplication on CPU = %.6f\n", timeOnCPU); 
    printf("Time taken for matrix multiplication on GPU = %.6f\n", timeOnGPU); 
    printf("%s\n", str); 

    // cleanup 
    cleanup(); 

    return (0); 
}

void initA(int* data, int row, int col) 
{
    int num = 1; 
    // code 
    for(int i = 0; i < row; i++) 
    {
        for(int j = 0; j < col; ++j) 
        {
            *(data + i * col + j) = num; 
            num++; 
        }
    }
}

void initB(int* data, int row, int col) 
{
    int num = BLOCK_WIDTH; 
    // code 
    for(int i = 0; i < row; ++i) 
    {
        for(int j = 0; j < col; ++j) 
        {
            *(data + i * col + j) = num; 
            num--; 
        }
    }
}

void matMulCPU(int* A, int* B, int* C, int numARows, int numAColumns, int numBColumns, int numCColumn) 
{
    // code 
    StopWatchInterface* timer = NULL; 
    sdkCreateTimer(&timer); 
    sdkStartTimer(&timer); 

    for(int i = 0; i < numARows; ++i) 
    {
        for(int j = 0; j < numBColumns; ++j)
        {
            int value = 0.0f; 
            for(int k = 0; k < numAColumns; ++k) 
            {
                int a = A[i * numAColumns + k]; 
                int b = B[k * numBColumns + j]; 
                value += a * b; 
            }
            C[numCColumns + j] = value; 
        }
    } 

    sdkStopTimer(&timer); 
    timeOnCPU = sdkGetTimerValue(&timer); 
    sdkdeleteTimer(&timer); 
    timer = NULL; 
}

void cleanup(void) 
{
    // code 
    if(deviceC)
    {
        cudaFree(deviceC); 
        deviceC = NULL; 
    }

    if(deviceB) 
    {
        cudaFree(deviceB); 
        deviceB = NULL; 
    }

    if(deviceA) 
    {
        cudaFree(deviceA); 
        deviceA = NULL; 
    }

    if(gold) 
    {
        free(gold); 
        gold = NULL; 
    }

    if(hostC) 
    {
        free(hostC); 
        hostC = NULL; 
    }

    if(hostB) 
    {
        free(hostB); 
        hostB = NULL; 
    }

    if(hostA) 
    {
        free(hostA); 
        hostA = NULL; 
    }
}
