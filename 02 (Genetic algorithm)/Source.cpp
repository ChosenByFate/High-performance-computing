#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <limits>

#define _Number_Of_Parameters (const char)5

struct Polynomial
{
	double Coefficients[_Number_Of_Parameters];
	double Error;
	Polynomial()
	{
		for (char i = 0; i < _Number_Of_Parameters; ++i)
		{
			Coefficients[i] = (rand() / double(RAND_MAX) - 0.5) * 0.000'000'000'000'000'1;
			//Coefficients[i] = 0.;
		}
	}
	~Polynomial()
	{
		//free(Coefficients);
		std::cout << "I'm dying" << std::endl;
	}
	//bool operator<(const Polynomial& other) const
	//{
	//	return this->Error < other.Error;
	//}
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

void WriteToFile(const char *FileName, const double *Source, const int count = _Number_Of_Parameters, const bool saveCount = false, const char *separator = "\t")
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
		fprintf(Stream, (const char*)separator);
	}
	for (int i = 0; i < count-1; ++i)
	{
		fprintf(Stream, "%.20lf", Source[i]);
		fprintf(Stream, (const char*)separator);
	}
	fprintf(Stream, "%.20lf", Source[count - 1]);
	fclose(Stream);
}

void Fitness(const double *x, const double *y, Polynomial *individuals, const int *numberOfPoints, const int *numberOfIndividuals)
{
	double MSE;
	double approximatingFunction;
	for (int i = 0; i < *numberOfIndividuals; ++i)	// Индивиды.
	{
		MSE = 0.;
		for (int j = 0; j < *numberOfPoints; ++j)	// Все точки.
		{
			approximatingFunction = 0.;
			for (char k = 0; k < _Number_Of_Parameters; ++k)	// Мощность полинома.
			{
				approximatingFunction += individuals[i].Coefficients[k] * pow(x[j], k);
			}
			MSE += pow(approximatingFunction - y[j], 2);
		}
		individuals[i].Error = MSE;
	}
}

void Crossover(Polynomial *individuals, const int *numberOfIndividuals, const int *threshold)
{
	for (int i = *threshold; i < *numberOfIndividuals; i++)	//Сохранить лучших из популяции.
	{
		for (char j = 0; j < _Number_Of_Parameters; ++j)	//Худшие - умрут.
		{
			individuals[i].Coefficients[j] = individuals[i - *threshold].Coefficients[j];
		}
	}
	double exchange;
	for (int i = *threshold; i < *numberOfIndividuals - 1; i+=2)	//Crossing.
	{
		for (char j = 0; j < _Number_Of_Parameters; ++j)	//Скрещивание.
		{
			if (rand() % 2)	// (2/5 и 3/5) 40% и 60% генов от 1 и 2 родителей.
			{
				exchange = individuals[i].Coefficients[j];
				individuals[i].Coefficients[j] = individuals[i + 1].Coefficients[j];
				individuals[i + 1].Coefficients[j] = exchange;
			}
		}
	}
	//В итоге: первая половина массива (кроме 1 лучшего индивида) - в будущем мутируют; вторая - потомство.
}

void Mutation(Polynomial *individuals, const int *threshold, const double *mean, const double *variance)	//int *numberOfIndividuals, 
{
	double change;
	for (int i = 1; i < *threshold; ++i)	//First individual is the best.
	{
		//if (rand() % 2)	//Шанс мутации индивида = 50%.
		//	continue;
		for (int j = 0; j < _Number_Of_Parameters; ++j)
		{
			if (rand() % 2)	//Шанс мутации гена = 50%.
				continue;
			change = (std::rand() / (RAND_MAX / 2.) - 1.) * *variance + *mean;
			individuals[i].Coefficients[j] += change;
		}
	}
}

int main()
{
	srand((int)time(NULL));
	bool dataFromFiles;
	int numberOfPoints = NULL, numberOfIndividuals, numberOfEpochs, numberOfConstantEpochs, currentConstEpoch = 0;
	double mean, variance, *x = nullptr, *y = nullptr, minimalError = std::numeric_limits<double>::max();
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
//			y[i] = rand() % 11 - 5. + i;
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
	const int threshold = int(numberOfIndividuals / 2.f + 0.5f);
	Polynomial *polynomials = (Polynomial*)malloc(numberOfIndividuals * sizeof(Polynomial));
	for (int i = 0; i < numberOfIndividuals; i++)
	{
		polynomials[i] = Polynomial();
	}
	/*
	+ Fitness
	+ Crossover
	+ Mutation
	+/- Selection
	+ Show best fitness.
	*/
	clock_t startTimer, stopTimer;
	startTimer = clock();
	for (int i = 0; i < numberOfEpochs; ++i)
	{
		Fitness(x, y, polynomials, &numberOfPoints, &numberOfIndividuals);
		/// Поиск меньшей ошибки.
		std::qsort(polynomials, numberOfIndividuals, sizeof(Polynomial), Polynomial::compare);
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
		Crossover(polynomials, &numberOfIndividuals, &threshold);
		Mutation(polynomials, &threshold, &mean, &variance);
		
	}
	stopTimer = clock();
	printf("Time on CPU = %lf seconds.\n", (double)(stopTimer - startTimer) / CLOCKS_PER_SEC);
	for (char i = 0; i < _Number_Of_Parameters - 1; ++i)
	{
		printf("%.20lf * x^%i + ", polynomials[0].Coefficients[i], i);
	}
	printf("%.20lf * x^%i\n", polynomials[0].Coefficients[_Number_Of_Parameters - 1], _Number_Of_Parameters - 1);
	WriteToFile("Output.txt", polynomials[0].Coefficients);
	free(polynomials);
	free(x);
	free(y);
//	system("pause");
}
