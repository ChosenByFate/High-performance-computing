#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <limits>//

#define _Number_Of_Parameters (const char)5

static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

struct Polynomial
{
	double Coefficients[_Number_Of_Parameters];
	double Error;
	__host__ __device__ Polynomial()
	{
		for (char i = 0; i < _Number_Of_Parameters; ++i)
		{
			//Coefficients[i] = (rand() / double(RAND_MAX) - 0.5) * 0.000'000'000'000'000'1;
			Coefficients[i] = 0.;
		}
	}
	__host__ __device__ ~Polynomial()
	{
		//free(Coefficients);
		//std::cout << "I'm dying" << std::endl;
	}
	__host__ __device__ bool operator<(const Polynomial& other) const
	{
		return this->Error < other.Error;
	}
	//bool operator-(const Polynomial& other) const
	//{
	//	return this->Error - other.Error;
	//}
	static int compare(const void *x1, const void *x2)
	{
		//return ((*(Polynomial*)x1).Error - (*(Polynomial*)x2).Error);
		if (((*(Polynomial*)x1).Error > (*(Polynomial*)x2).Error))
			return 1;
		if (((*(Polynomial*)x1).Error < (*(Polynomial*)x2).Error))
			return -1;
		return 0;
	}
};

double *ReadFromFile(const char *FileName, int *count)
{
	FILE *Stream;
	if ((Stream = fopen(FileName, "r")) == NULL)
	{
		printf("Failed to open file.");
		return nullptr;
	}
	if (*count == NULL)
		fscanf(Stream, "%i", count);
	double *Destination = (double*)malloc(*count * sizeof(double));
	for (int i = 0; i < *count; ++i)
	{
		fscanf(Stream, "%lf", &Destination[i]);
	}
	fclose(Stream);
	return Destination;
}

void WriteToFile(const char *FileName, double *Source, int count = _Number_Of_Parameters, bool saveCount = false, const char *separator = "\t")
{
	FILE *Stream;
	if ((Stream = fopen(FileName, "w")) == NULL)
	{
		printf("Failed to open file.");
		return;
	}
	if (saveCount)
	{
		fprintf(Stream, "%i", count);
		fprintf(Stream, separator);
	}
	for (int i = 0; i < count-1; ++i)
	{
		fprintf(Stream, "%.20lf", Source[i]);
		fprintf(Stream, separator);
	}
	fprintf(Stream, "%.20lf", Source[count - 1]);
	fclose(Stream);
}

__global__ void ClearError(Polynomial *individuals, int numberOfIndividuals)
{
	int individual = blockIdx.x;
	if (individual < numberOfIndividuals)
		individuals[individual].Error = 0;
}

__global__ void Fitness(double *x, double *y, Polynomial *individuals, int numberOfPoints, int numberOfIndividuals)
{
	int individual = blockIdx.x * blockDim.x + threadIdx.x;
	if (individual < numberOfIndividuals)
	{
		double MSE = 0.;
		double approximatingFunction;
		for (int j = 0; j < numberOfPoints; ++j)
		{
			approximatingFunction = 0.;
			for (char k = 0; k < _Number_Of_Parameters; ++k)	// Мощность полинома.
			{
				approximatingFunction += individuals[individual].Coefficients[k] * pow(x[j], (double)k);
			}
			MSE += pow(approximatingFunction - y[j], 2);
		}
		individuals[individual].Error = MSE;
	}
}

//Потоков = numberOfIndividuals - threshold.
__global__ void Crossover(Polynomial *individuals, int numberOfIndividuals, int threshold)
{
	int individual = blockIdx.x + threshold;
	if (individual < numberOfIndividuals)
	{
		for (char i = 0; i < _Number_Of_Parameters; ++i)	//Худшие - умрут.
		{
			individuals[individual].Coefficients[i] = individuals[individual - threshold].Coefficients[i];
		}
	}
}

__global__ void CrossoverNext(Polynomial *individuals, int numberOfIndividuals, int threshold)
{
	int individual = blockIdx.x + threshold;
	if (individual < numberOfIndividuals && !(individual % 2))
	{
		curandState state;
		double exchange;
		for (char j = 0; j < _Number_Of_Parameters; ++j)	//Скрещивание.
		{
			curand_init((unsigned long long)clock() + individual, 0, 0, &state);
			if ((curand_normal(&state) - 0.5f) > 0)	// (2/5 и 3/5) 40% и 60% генов от 1 и 2 родителей.
			{
				exchange = individuals[individual].Coefficients[j];
				individuals[individual].Coefficients[j] = individuals[individual + 1].Coefficients[j];
				individuals[individual + 1].Coefficients[j] = exchange;
			}
		}
	}
	//В итоге: первая половина массива (кроме 1 лучшего индивида) - в будущем мутируют; вторая - потомство.
}

//Потоков = threshold - 1.
__global__ void Mutation(Polynomial *individuals, int numberOfIndividuals, int threshold, double mean, double variance)
{
	int individual = blockIdx.x + threshold + 1; 	//First individual is the best. That's why we don't touch it.
	if (individual < numberOfIndividuals)
	{
		curandState state;
		double change;
		for (int j = 0; j < _Number_Of_Parameters; ++j)
		{
			curand_init((unsigned long long)clock() + individual, 0, 0, &state);
			if ((curand_normal(&state) - 0.5f) > 0)	//Шанс мутации гена = 50%.
				continue;
			//curand_log_normal_double(...)
			curand_init((unsigned long long)clock() + individual, 0, 0, &state);
			change = curand_normal_double(&state) * variance + mean;
			individuals[individual].Coefficients[j] += change;
		}
	}
}

int main()
{
	srand((int)time(NULL));
	bool dataFromFiles;
	int numberOfPoints = NULL;
	int numberOfIndividuals;
	double mean, variance;
	int numberOfEpochs;
	int numberOfConstantEpochs;
	int currentConstEpoch = 0;
	int threshold;	// Порог разбивающий популяцию на две (равные) части.
	double *x = nullptr;
	double *y = nullptr;
	double minimalError = std::numeric_limits<double>::max();
	std::cout << "Points from files (1 - YES, 0 - NO): ";
	std::cin >> dataFromFiles;
	if (dataFromFiles)
	{
		x = ReadFromFile("InputX.txt", &numberOfPoints);
		y = ReadFromFile("InputY.txt", &numberOfPoints);
	}
	else
	{
		std::cout << "Number of points (500 - 1000): ";
		std::cin >> numberOfPoints;
		x = (double*)malloc(numberOfPoints * sizeof(double));
		y = (double*)malloc(numberOfPoints * sizeof(double));
		for (int i = 0; i < numberOfPoints; ++i)
		{
			x[i] = (double)i;
			y[i] = rand() % 41 - 20. + (1000. * i / (i + 500) - i / 5);
		}
		std::cout << "Save points in files? (1 - YES, 0 - NO): ";
		std::cin >> dataFromFiles;
		if (dataFromFiles)
		{
			WriteToFile("InputX.txt", x, numberOfPoints, true, "\n");
			WriteToFile("InputY.txt", y, numberOfPoints, false, "\n");
		}
	}
	double *xGPU;
	double *yGPU;
	HANDLE_ERROR(cudaMalloc((void**)&xGPU, numberOfPoints * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&yGPU, numberOfPoints * sizeof(double)));
	HANDLE_ERROR(cudaMemcpy(xGPU, x, numberOfPoints * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(yGPU, y, numberOfPoints * sizeof(double), cudaMemcpyHostToDevice));

	std::cout << "Number of individuals (1000 - 2000): ";
	std::cin >> numberOfIndividuals;
	std::cout << "Mean of mutation: ";
	std::cin >> mean;
	std::cout << "Variance of mutation: ";
	std::cin >> variance;
	std::cout << "Number of epochs: ";
	std::cin >> numberOfEpochs;
	std::cout << "Number of epochs with constant value of the best fitness: ";
	std::cin >> numberOfConstantEpochs;
	threshold = int(numberOfIndividuals / 2.f + 0.5f);
	
	int threadsPerBlockDim = 32;
	dim3 blockDim(threadsPerBlockDim, 1, 1);
	int blocksPerGridDimX = (int)ceilf(numberOfIndividuals / (float)threadsPerBlockDim);
	dim3 gridDim(blocksPerGridDimX, 1, 1);

	Polynomial *polynomials = (Polynomial*)malloc(numberOfIndividuals * sizeof(Polynomial));
	for (int i = 0; i < numberOfIndividuals; i++)
	{
		polynomials[i] = Polynomial();
	}
	Polynomial *polynomialsGPU;
	HANDLE_ERROR(cudaMalloc((void**)&polynomialsGPU, numberOfIndividuals * sizeof(Polynomial)));
	HANDLE_ERROR(cudaMemcpy(polynomialsGPU, polynomials, numberOfIndividuals * sizeof(Polynomial), cudaMemcpyHostToDevice));
	
	clock_t startTimer, stopTimer;
	startTimer = clock();
	for (int i = 0; i < numberOfEpochs; ++i)
	{
		Fitness<<<gridDim, blockDim>>>(xGPU, yGPU, polynomialsGPU, numberOfPoints, numberOfIndividuals);
		/// Поиск меньшей ошибки.
		thrust::sort(thrust::device, polynomialsGPU, polynomialsGPU + numberOfIndividuals);
		HANDLE_ERROR(cudaMemcpy(polynomials, polynomialsGPU, numberOfIndividuals * sizeof(Polynomial), cudaMemcpyDeviceToHost));
		printf("Epoch %i. Lowest error = %lf\n", i, polynomials[0].Error);
		if (minimalError > polynomials[0].Error)
		{
			minimalError = polynomials[0].Error;
			currentConstEpoch = 0;
		}
		else
		{
			++currentConstEpoch;
			if (currentConstEpoch >= numberOfConstantEpochs)
				break;
		}
		/// Репродукция и мутация.
		Crossover<<<numberOfIndividuals - threshold, 1>>>(polynomialsGPU, numberOfIndividuals, threshold);
		CrossoverNext<<<numberOfIndividuals - 1 - threshold, 1>>>(polynomialsGPU, numberOfIndividuals, threshold);
		Mutation<<<threshold - 1, 1>>>(polynomialsGPU, numberOfIndividuals, threshold, mean, variance);
		HANDLE_ERROR(cudaMemcpy(polynomials, polynomialsGPU, numberOfIndividuals * sizeof(Polynomial), cudaMemcpyDeviceToHost));
	}
	stopTimer = clock();
	printf("Time on GPU = %lf seconds.\n", (double)(stopTimer - startTimer) / CLOCKS_PER_SEC);
	HANDLE_ERROR(cudaMemcpy(polynomials, polynomialsGPU, numberOfIndividuals * sizeof(Polynomial), cudaMemcpyDeviceToHost));
	for (char i = 0; i < _Number_Of_Parameters - 1; ++i)
	{
		printf("%.20lf * x^%i + ", polynomials[0].Coefficients[i], i);
	}
	printf("%.20lf * x^%i\n", polynomials[0].Coefficients[_Number_Of_Parameters - 1], _Number_Of_Parameters - 1);
	printf("blocksPerGridDimX (points) %i.\n", blocksPerGridDimX);
	WriteToFile("Output.txt", polynomials[0].Coefficients);
	HANDLE_ERROR(cudaFree(polynomialsGPU));
	HANDLE_ERROR(cudaFree(xGPU));
	HANDLE_ERROR(cudaFree(yGPU));
	free(polynomials);
	free(x);
	free(y);
//	system("pause");
}