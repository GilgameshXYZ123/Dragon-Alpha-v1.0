#pragma once

#define TEST_H
#ifndef TEST_H
#define TEST_H


#ifndef UTIL
#define UTIL

float *newFloatVec(int length)
{
	float *p = new float[length];
	memset(p, 0, sizeof(float)*length);
	return p;
}

float* newRandomFloatVec(int height, int width, int stride)//0-256
{
	int lengthv = height * stride;
	float *p = new float[lengthv], *tp = p;
	memset(p, 0, sizeof(float)*lengthv);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++) tp[j] = (float)(rand() % 1000) / 1000;
		tp += stride;
	}

	return p;
}
float *newDevFloatVec(int length)
{
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemset(dp, 0, size); printError(error);
	return dp;
}
float* newDevFloatVec(float *p, int length)
{
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemcpy(dp, p, size, cudaMemcpyHostToDevice); printError(error);
	return dp;
}
void println(float *p, int length)
{
	for (int i = 0; i < length; i++) cout << p[i] << ' ';
	cout << endl;
}
float samePercent(float *a, float *b, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		//if (b[i] == 0) cout << "zero: " << i << endl;
		if (a[i] == b[i])
		{
			sum++; continue;
		}
		float dif = fabs(a[i] - b[i]);
		//float dif = fabs((a[i] - b[i]) / (a[i] + b[i]));
		if (dif < 1e-3) sum++;
		//if (a[i] < b[i]) cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
		//else cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
	}
	return 1.0f*sum / length;
}

float zeroPercent(float *a, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
		if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}

float sum(float* a, int length)
{
	float r = 0;
	for (int i = 0; i < length; i++) r += a[i];
	return r;
}

#endif

void testCorrect(int height, int width)
{
	int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
	int length = height * width;

	printf("test correct:\n");
	printf("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
	printf("(lengthv, length) = (%d, %d)\n", lengthv, length);

	float *X = new float[lengthv];
	memset(X, 0, sizeof(int)*lengthv);

	//GPU--------------------------------------------------
	cudaError_t error;
	float *dX = newDevFloatVec(X, lengthv);

	__uniform2D(0, dX, 10, -1, 1, lengthv, width, stride);
	error = cudaDeviceSynchronize(); printError(error);

	float *Y = new float[lengthv];
	error = cudaMemcpy(Y, dX, sizeof(float)*lengthv, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU:\n"; 
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < stride; j++) cout << Y[j] << '\t';
		cout << endl; Y += stride;
	}

	//clear------------------------------------------------
	error = cudaFree(dX); printError(error);

	delete X;
}


void test()
{
	int height = 10, width = 11;
	testCorrect(height, width);
}

main()
{
	test();
}

#endif