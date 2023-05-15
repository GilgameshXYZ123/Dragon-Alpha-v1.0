#pragma once

#define TEST_H
#ifndef TEST_H
#define TEST_H


#ifndef UTIL
#define UTIL

float* newRandomFloatVec(int length) {//0-256
	float *p = new float[length];
	for (int i = 0; i < length; i++) p[i] = (float)(rand() % 1000) / 1000 + 1;
	return p;
}

char* newRandomCharVec(int length) {//0-256
	char *p = new char[length];
	for (int i = 0; i < length; i++) p[i] = (rand() & 127);
	return p;
}

float* newRandomFloatVec2D(int height, int width, int stride)
{
	int length = height * stride;
	float* p = new float[length];
	for (int i = 0; i < height; i++) 
	{
		for (int j = 0; j < width; j++) *p++ = (float)(rand() % 1000) / 1000 + 1;
		for (int j = width; j < stride; j++) *p++ = 0;
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

void println(float *p, int length) {
	for (int i = 0; i < length; i++) cout << p[i] << ' ';
	cout << endl;
}

void println(char *p, int length) {
	for (int i = 0; i < length; i++) cout << (int)p[i] << ' ';
	cout << endl;
}

float samePercent(float *a, float *b, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		//if (b[i] == 0) cout << "zero: " << i << endl;
		if (a[i] == b[i]) { sum++; continue; }
		float dif = fabs(a[i] - b[i]);
		//float dif = fabs((a[i] - b[i]) / (a[i] + b[i]));
		if (dif < 1e-2) sum++;
		else {
			//if (a[i] < b[i]) cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
			//cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
		}

	}
	return 1.0f*sum / length;
}

float samePercent(char *a, char *b, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++) {
		if (a[i] == b[i]) { sum++; continue; }
	}
	return 1.0f*sum / length;
}

float zeroPercent(float *a, int length) {
	int sum = 0;
	for (int i = 0; i < length; i++) if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}

float zeroPercent(char *a, int length) {
	int sum = 0;
	for (int i = 0; i < length; i++) if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}

float sum(float* a, int length)
{
	float r = 0;
	for (int i = 0; i < length; i++) r += a[i];
	return r;
}

#endif


#ifndef CPU2GPU_MEMCPY_TEST
#define CPU2GPU_MEMCPY_TEST

void memcpy2(float* X, float *dX, int length,
	cudaStream_t stream1, cudaStream_t stream2)
{
	int length1 = length >> 1, length2 = length - length1;
	cudaError_t error;
	error = cudaMemcpyAsync(dX, X, length1 * sizeof(float),
		cudaMemcpyHostToDevice, stream1); printError(error);
	error = cudaMemcpyAsync(dX + length1, X + length1, length1 * sizeof(float),
		cudaMemcpyHostToDevice, stream2); printError(error);
	error = cudaStreamSynchronize(stream1);
	error = cudaStreamSynchronize(stream2);
}


void testCorrect_v1(int length)
{
	cudaError_t error;

	float* X = NULL;
	error = cudaMallocHost((void**)&X, sizeof(float)*length); printError(error);

	cudaStream_t stream1, stream2;
	error = cudaStreamCreate(&stream1); printError(error);
	error = cudaStreamCreate(&stream2); printError(error);

	//GPU-----------------------------------------------------
	float* dX = newDevFloatVec(length);

	//error = cudaMemcpy(dX, X, length*sizeof(float), cudaMemcpyHostToDevice); printError(error);
	memcpy2(X, dX, length, stream1, stream2);


	//compare--------------------------------------------------
	float* X2 = new float[length];
	cudaMemcpy(X2, dX, sizeof(float)*length, cudaMemcpyDeviceToHost); printError(error);
	error = cudaFree(dX); printError(error);

	float sp = samePercent(X, X2, length);
	cout << "X1 = "; println(X, 10);
	cout << "X2 = "; println(X2, 10);
	cout << "sp = " << sp << endl;

	error = cudaFreeHost(dX);
	delete X2;
}


//CPU: 8GB/s, 
//GPU: 4GB/s, 4.29553GB/s
void testSpeed_v1(int nIter, int length)
{
	cudaError_t error;
	float* X = NULL;
	error = cudaMallocHost((void**)&X, sizeof(float)*length); printError(error);

	float* X2 = new float[length];
	float* dX = newDevFloatVec(length);

	clock_t start = clock();

	cudaStream_t stream1, stream2;
	error = cudaStreamCreate(&stream1); printError(error);
	error = cudaStreamCreate(&stream2); printError(error);

	for (int i = 0; i < nIter; i++)
	{
		//memcpy(X2, X, sizeof(float)*length);
		//error = cudaMemcpy(dX, X, length * sizeof(float), cudaMemcpyHostToDevice); printError(error);
		//memcpy2(X, dX, length, stream1, stream2);

	}
	clock_t end = clock();

	int div = (end - start);
	float time = 1.0f * div / nIter;
	int data_size = length * sizeof(float);
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	cout << "Time = " << time << " mesc, "
		<< "Speed = " << speed << "GB/s" << endl;

	error = cudaFree(dX); printError(error);
	error = cudaFreeHost(dX);
	delete X2;
}

#endif


#ifndef FLOAT_COPY
#define FLOAT_COPY

void copy_naive_2D_v1(float* a, float *b, int height, int width, int stride) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) *(b++) = *(a++);
		for (int j = width; j < stride; j++) *(b++) = 0;
	}
}

void copy_naive_2D_v2(float* a, float *b, int height, int width, int stride) {
	int length = height * width; a += length;
	int lengthv = height * stride; b += lengthv;

	for (int i = 0; i < height; i++) {
		for (int j = width; j < stride; j++) *(--b) = 0;
		for (int j = 0; j < width; j++) *(--b) = *(--a);
	}
}

void testCorrect_v2(int height, int width)
{
	cout << endl << "testCorrect_v2 = " << height << ", " << width << endl;

	int stride = (width + 3) >> 2 << 2;
	int length = height * width;
	int lengthv = height * stride;

	float* A = newRandomFloatVec(length);
	float* B = new float[lengthv];

	//copy_naive_2D(A, B, height, width, stride);
	copy_naive_2D_v2(A, B, height, width, stride);

	println(A, length  < 10 ? length  : 10);
	println(B, lengthv < 10 ? lengthv : 10);

	float zp = zeroPercent(B, lengthv);
	float sp = samePercent(A, B, length);
	cout << "zp = " << zp << endl;
	cout << "sp = " << sp << endl;
}

void testSpeed_v2(int nIter, int height, int width) 
{
	cout << endl << "testSpeed_v2 = " << height << ", " << width << endl;

	int stride = (width + 3) >> 2 << 2;
	int length = height * width;
	int lengthv = height * stride;

	float* A = newRandomFloatVec(length);
	float* B = new float[lengthv];

	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		//copy_naive_2D(A, B, height, width, stride);//8.69343GB/s
		copy_naive_2D_v2(A, B, height, width, stride);
	}
	clock_t end = clock();

	int div = (end - start);
	float time = 1.0f * div / nIter;
	int data_size = length * sizeof(float);
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	cout << "Time = " << time << " mesc, "
		<< "Speed = " << speed << "GB/s" << endl;
}

#endif


#ifndef CHAR_COPY
#define CHAR_COPY

void copy_naive_2D_v1(char* a, char *b,
	int height, int width, int stride) 
{
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) *(b++) = *(a++);
		for (int j = width; j < stride; j++) *(b++) = 0;
	}
}

//stride - width <= 3
void copy_naive_2D_v2(char* a, char *b, 
	int height, int width, int stride)
{
	int width4 = width >> 2 << 2;
	for (int i = 0; i < height; i++)
	{
		int *a64 = (int*)a;
		int *b64 = (int*)b;

		int j = 0;
		for (; j < width4; j += 4) *(b64++) = *(a64++);

		a = (char*)a64;
		b = (char*)b64;

		for (; j < width; j++) *(b++) = *(a++);
		for (j = width; j < stride; j++) *(b++) = 0;
	}
}

void test_v3(int nIter, int height, int width)
{
	cout << endl << "testCorrect_v3 = " << height << ", " << width << endl;

	int stride = (width + 3) >> 2 << 2;
	int length = height * width;
	int lengthv = height * stride;

	char* A = newRandomCharVec(length);
	char* B1 = new char[lengthv];
	char* B2 = new char[lengthv];

	//method1----------------------------------------
	copy_naive_2D_v1(A, B1, height, width, stride);


	//method2----------------------------------------
	__zeroPad_W3S4(A, B2, height);

	//compare----------------------------------------
	println(A,  length  < 20 ? length  : 20);
	println(B1, lengthv < 20 ? lengthv : 20);
	println(B2, lengthv < 20 ? lengthv : 20);

	float zp0 = zeroPercent(B1, lengthv);
	float zp1 = zeroPercent(B2, lengthv);
	float sp = samePercent(B1, B2, length);
	cout << "zp = " << zp0 << endl;
	cout << "zp = " << zp1 << endl;
	cout << "sp = " << sp << endl;
	
	if (sp != 1) exit(-2);
	if (nIter < 0) return;

	//testSpeed------------------------------------------------------------
	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		__zeroPad_W3S4(A, B2, height);
	}
	clock_t end = clock();

	int div = (end - start);
	float time = 1.0f * div / nIter;
	int data_size = length * sizeof(float);
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	cout << "Time = " << time << " mesc, "
		<< "Speed = " << speed << " GB/s" << endl;
}

#endif 


void test()
{
	//int length = 64 * 16 * 128 * 128;
	//testCorrect_v1(length);
	//testSpeed_v1(20, length);

	//testCorrect_v2(2, 5);
	//testSpeed_v2(500, 512, 32 * 32 * 3);

	test_v3(100, 512 * 32 * 32, 3);

	for (int h = 1; h < 512; h++)
		test_v3(-1, h, 3);
}

main() { test(); }

#endif

