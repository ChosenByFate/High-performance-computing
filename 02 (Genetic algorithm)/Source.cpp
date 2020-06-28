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

void Fitness(double *x, double *y, Polynomial *individuals, int *numberOfPoints, int *numberOfIndividuals)
{
	double MSE;
	double approximatingFunction;
	for (int i = 0; i < *numberOfIndividuals; ++i)	// Èíäèâèäû.
	{
		MSE = 0.;
		for (int j = 0; j < *numberOfPoints; ++j)	// Âñå òî÷êè.
		{
			approximatingFunction = 0.;
			for (char k = 0; k < _Number_Of_Parameters; ++k)	// Ìîùíîñòü ïîëèíîìà.
			{
				approximatingFunction += individuals[i].Coefficients[k] * pow(x[j], k);
			}
			MSE += pow(approximatingFunction - y[j], 2);
		}
		individuals[i].Error = MSE;
	}
}

void Crossover(Polynomial *individuals, int *numberOfIndividuals, int *threshold)
{
	for (int i = *threshold; i < *numberOfIndividuals; i++)	//Ñîõðàíèòü ëó÷øèõ èç ïîïóëÿöèè.
	{
		for (char j = 0; j < _Number_Of_Parameters; ++j)	//Õóäøèå - óìðóò.
		{
			individuals[i].Coefficients[j] = individuals[i - *threshold].Coefficients[j];
		}
	}
	double exchange;
	for (int i = *threshold; i < *numberOfIndividuals - 1; i+=2)	//Crossing.
	{
		for (char j = 0; j < _Number_Of_Parameters; ++j)	//Ñêðåùèâàíèå.
		{
			if (rand() % 2)	// (2/5 è 3/5) 40% è 60% ãåíîâ îò 1 è 2 ðîäèòåëåé.
			{
				exchange = individuals[i].Coefficients[j];
				individuals[i].Coefficients[j] = individuals[i + 1].Coefficients[j];
				individuals[i + 1].Coefficients[j] = exchange;
			}
		}
	}
	//Â èòîãå: ïåðâàÿ ïîëîâèíà ìàññèâà (êðîìå 1 ëó÷øåãî èíäèâèäà) - â áóäóùåì ìóòèðóþò; âòîðàÿ - ïîòîìñòâî.
}

void Mutation(Polynomial *individuals, int *threshold, double *mean, double *variance)	//int *numberOfIndividuals, 
{
	double change;
	for (int i = 1; i < *threshold; ++i)	//First individual is the best.
	{
		//if (rand() % 2)	//Øàíñ ìóòàöèè èíäèâèäà = 50%.
		//	continue;
		for (int j = 0; j < _Number_Of_Parameters; ++j)
		{
			if (rand() % 2)	//Øàíñ ìóòàöèè ãåíà = 50%.
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
	int numberOfPoints = NULL;
	int numberOfIndividuals;
	double mean, variance;
	int numberOfEpochs;
	int numberOfConstantEpochs;
	int currentConstEpoch = 0;
	int threshold;	// Ïîðîã ðàçáèâàþùèé ïîïóëÿöèþ íà äâå (ðàâíûå) ÷àñòè.
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
	threshold = int(numberOfIndividuals / 2.f + 0.5f);
	Polynomial *polynomials = (Polynomial*)malloc(numberOfIndividuals * sizeof(Polynomial));
	for (int i = 0; i < numberOfIndividuals; i++)
	{
		polynomials[i] = Polynomial();
	}
	/*
	+ Fitness
	+ Crossover
	+ Mutation
	+ Selection
	+ Show best fitness.
	*/
	clock_t startTimer, stopTimer;
	startTimer = clock();
	for (int i = 0; i < numberOfEpochs; ++i)
	{
		Fitness(x, y, polynomials, &numberOfPoints, &numberOfIndividuals);
		/// Ïîèñê ìåíüøåé îøèáêè.
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
		/// Ðåïðîäóêöèÿ è ìóòàöèÿ.
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
