#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <malloc.h>
//#include <stdlib.h>

#define nExperiments 5

static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int* CreateMatrix(int *MatrixSize, bool ZerosMatrix)
{
	int *A;
	A = (int*)malloc(*MatrixSize * *MatrixSize * sizeof(int));
	for (int i = 0; i < *MatrixSize * *MatrixSize; ++i)
	{
		A[i] = ZerosMatrix ? 0 : rand() % 200 - 99;
	}
	return A;
}

void MatrixMultiplicationCPU(int *a, int *b, int *c, int n)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			float temp = 0.0f;
			for (int k = 0; k < n; ++k)
			{
				temp += a[i * n + k] * b[k * n + j];
			}
			c[i * n + j] = temp;
		}
	}
}

__global__ void MatrixMultiplicationGPU(int *a, int *b, int *c, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n && col < n)
	{
		float temp = 0.0f;
		for (int i = 0; i < n; ++i)
		{
			temp += a[row * n + i] * b[i * n + col];
		}
		c[row * n + col] = temp;
	}
}

int main(void)
{
	char deviceToComputing;
	printf("Choose the device to computing: only GPU[g], only CPU[c], both[_] ");
	scanf("%c", &deviceToComputing);
//	printf("Choose the device to computing: only [%c] ", &deviceToComputing);
	int N;
	printf("Matrix size: ");
	scanf("%d", &N);

	int *A = NULL, *B = NULL, *C,
	//int *A, *B, *C,
		*A_device, *B_device, *C_device;
	clock_t startCPU, stopCPU;
	double timeCPU[nExperiments], resultCPU = 0;
	float timeGPU[nExperiments], resultGPU = 0;
	
	int threadsPerBlockDim = 32;
	dim3 blockDim(threadsPerBlockDim, threadsPerBlockDim, 1);
	int blocksPerGridDimX = ceilf(N / (float)threadsPerBlockDim);
	int blocksPerGridDimY = ceilf(N / (float)threadsPerBlockDim);
	dim3 gridDim(blocksPerGridDimX, blocksPerGridDimY, 1);
	
	for (int i = 0; i < nExperiments; ++i)
	{
		A = CreateMatrix(&N, false);
		B = CreateMatrix(&N, false);
		C = CreateMatrix(&N, true);

		if (deviceToComputing != 'c' && deviceToComputing != 'C')
		{
			//Allocate the memory on the GPU.
			HANDLE_ERROR(cudaMalloc((void**)&A_device, N * N * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)&B_device, N * N * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)&C_device, N * N * sizeof(int)));
			
			//Copy the arrays 'A' and 'B' to the GPU.
			HANDLE_ERROR(cudaMemcpy(A_device, A, N * N * sizeof(int), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(B_device, B, N * N * sizeof(int), cudaMemcpyHostToDevice));
			
			cudaEvent_t startGPU, stopGPU;
			timeGPU[i] = 0.0f;
			
			cudaEventCreate(&startGPU);
			cudaEventCreate(&stopGPU);
			cudaEventRecord(startGPU, 0);
			
			MatrixMultiplicationGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N);
			
			cudaEventRecord(stopGPU, 0);
			cudaEventSynchronize(stopGPU);
			cudaEventElapsedTime(&timeGPU[i], startGPU, stopGPU);

			cudaEventDestroy(startGPU);
			cudaEventDestroy(stopGPU);
			
			//Copy the array 'C' back from the GPU to the CPU.
			HANDLE_ERROR(cudaMemcpy(C, C_device, N * N * sizeof(int), cudaMemcpyDeviceToHost));
			
			printf("Time on GPU = %f seconds.\n", timeGPU[i]/1000);
			
			//Free the memory allocated on the GPU.
			HANDLE_ERROR(cudaFree(A_device));
			HANDLE_ERROR(cudaFree(B_device));
			HANDLE_ERROR(cudaFree(C_device));
		}
		
		if (deviceToComputing != 'g' && deviceToComputing != 'G')
		{
			startCPU = clock();
			MatrixMultiplicationCPU(A, B, C, N);
			stopCPU = clock();
			timeCPU[i] = (double)(stopCPU - startCPU) / CLOCKS_PER_SEC;
			printf("Time on CPU = %lf seconds.\n", timeCPU[i]);
		}

		free(A);
		free(B);
		free(C);
	}
	if (deviceToComputing != 'c' && deviceToComputing != 'C')
	{
		for (int i = 0; i < nExperiments; ++i)
		{
			resultGPU += timeGPU[i];
		}
		resultGPU /= nExperiments * 1000;
		printf("Average execution time on the GPU: %f.\n", resultGPU);
	}
	if (deviceToComputing != 'g' && deviceToComputing != 'G')
	{
		for (int i = 0; i < nExperiments; ++i)
		{
			resultCPU += timeCPU[i];
		}
		resultCPU /= nExperiments;
		printf("Average execution time on the CPU: %lf.\n", resultCPU);
	}
}
