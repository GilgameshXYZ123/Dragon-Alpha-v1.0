#pragma once

#define TEST_H
#ifndef TEST_H
#define TEST_H


float samePercent(float *a, float *b, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		if (a[i] == b[i]) { sum++; continue; }
		float dif = fabs((a[i] - b[i]) / (a[i] + b[i]));
		if (dif < 1e-3) sum++;
		else cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
	}
	return 1.0f*sum / length;
}

#ifndef UTIL
#define UTIL

float* newRandomFloatVec(int length)
{
	float *p = new float[length];
	for (int i = 0; i < length; i++) p[i] = 1.0f*(rand() % 1000) / 1000 + 1;
	return p;
}

float* newRandomFloatVec(int lengthv, int width, int stride)//0-256
{
	int height = lengthv / stride;
	float *p = new float[lengthv], *tp = p;
	memset(p, 0, sizeof(float)*lengthv);
	for (int i = 0; i < height; i++)
{
		for (int j = 0; j < width; j++) 
			tp[j] = (float)(rand() % 1000) / 1000;
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

float zeroPercent(float *a, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
		if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}


void SumEachRow(float *A, float *V, int N, int M, int SA)
{
	memset(V, 0, sizeof(float)*N);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++) V[i] += get(A, i, j, SA);
}
void MsumOfEachRow(float **A, float *V, int N, int M)
{
	for (int i = 0; i < N; i++)
	{
		V[i] = 0;
		for (int j = 0; j < M; j++) V[i] += A[i][j];
	}
}

float sum(float* a, int length)
{
	float r = 0;
	for (int i = 0; i < length; i++) r += a[i];
	return r;
}

void SumEachField(float *A, float *V, int N, int M)
{
	memset(V, 0, sizeof(float)*M);
	for (int j = 0; j < M; j++)
		for (int i = 0; i < N; i++) V[j] += get(A, i, j, M);
}

#endif

void MaxEachField(float *A, int N, int M, float *V, int width, int stride)
{
	for (int j = 0; j < M; j++)
	{
		if (j%stride >= width) continue;
		float max = FLOAT_MIN;
		for (int i = 0; i < N; i++) {
			int index = i * M + j;
			if((index % stride) < width)
				max = (max >= A[index] ? max : A[index]);
		}
		if (j % stride < width) V[j] = max;
	}
}

void BinomialEachField(float *A, float alpha, float beta, float gamma,
	float *V, int height, int width, int stride)
{
	memset(V, 0, sizeof(float)*width);
	for (int j = 0; j < stride; j++)
	{
		if ((j%stride) >= width) continue;
		for (int i = 0; i < height; i++)
		{
			float a = get(A, i, j, stride);
			V[j] += alpha * (a*a) + beta * a + gamma;
		}
	}
}

void BinomialEachRow(float *A, float alpha, float beta, float gamma,
	float *V, int N, int M, int width, int stride)
{
	memset(V, 0, sizeof(float)*N);
	for (int i = 0; i < N; i++) 
		for (int j = 0; j < M; j++)
		{
			float a = get(A, i, j, M);
			a = alpha * a*a + beta * a + gamma;
			V[i] += a * ((j % stride) < width);
		}
		
}

template<int LBY, int LBX, int LTY, int LTX>
void testCorrect(int N, int M, int width)
{
	int lengthv = N * M;
	int stride = (width + 3) >> 2 << 2;
	printf("width, stride = (%d, %d)\n", width, stride);
	printf("N, M, lengthv = (%d, %d, %d)\n", N, M, lengthv);

	float *A = newRandomFloatVec(lengthv, width, stride);
	float *dA = newDevFloatVec(A, lengthv);

	//CPU----------------------------------------------
	float *V1 = new float[N];

	//BinomialEachField(A, 1.0f, 2.0f, 3.0f, V1, N, M);
	BinomialEachRow(A, 1, 2, 3, V1, N, M, width, stride);

	cout << "CPU: "; println(V1, 10);

	//GPU-----------------------------------------------
	cudaError_t error;
	
	//int HV = N << 1 >> LBY >> LTY ;
	int HV = nextRowGridY(N);
	cout << "HV = " << HV << endl;

	float *dV = newDevFloatVec(HV * N);

	//max_field4(0, LBY, LBX, LTY, LTX, dA, N, M, dV, width, stride);
	//field_binomial4(0, LBY, LBX, LTY, LTX, dA, 1.0f, 2.0f, 3.0f, N, M, dV);
	__row_binomial_stage(0, dA, 1, 2, 3, N, M, dV, width, stride);

	error = cudaDeviceSynchronize(); printError(error);

	float *Vr = new float[HV * N];
	error = cudaMemcpy(Vr, dV, sizeof(float)* HV * N, cudaMemcpyDeviceToHost); printError(error);

	float* V2 = new float[N];
	SumEachField(Vr, V2, HV, N);
	cout << "GPU: "; println(V2, 10);

	error = cudaGetLastError(); printError(error);
	//compare------------------------------------------------
	float sp = samePercent(V1, V2, N); cout << "sp = " << sp << endl;
	float zp0 = zeroPercent(V1, N); cout << "zp0 = " << zp0 << endl;
	float zp1 = zeroPercent(V2, N); cout << "zp1 = " << zp1 << endl;
	
	//error = cudaFree(dA); printError(error);
	//error = cudaFree(dV); printError(error);

	/*delete A;
	delete V1;
	delete V2;*/

	if (sp != 1) {
		cout << "Error: N = " << N << ", M = " << M << endl;
		exit(2);
	}
}

template<int LBY, int LBX, int LTY, int LTX>
void testSpeed(int N, int M, int width)
{
	int lengthv = N * M;
	int stride = (width + 3) >> 2 << 2;
	printf("N, M, lengthv = (%d, %d, %d)\n", N, M, lengthv);

	float *A = newRandomFloatVec(lengthv, width, stride);
	float *dA = newDevFloatVec(A, lengthv);

	//GPU-----------------------------------------------
	cudaError_t error;

	int HV = N << 1 >> LBY >> LTY;
	float *dV = newDevFloatVec(HV * M);

	clock_t start = clock();
	int nIter = 500;
	for (int i = 0; i < nIter; i++)
	{
		//field_binomial4(0, LBY, LBX, LTY, LTX, dA, 1.0f, 2.0f, 3.0f, N, M, dV);
		__row_binomial_stage(0, dA, 1, 2, 3, N, M, dV, width, stride);
	}
	error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = (end - start);
	float time = 1.0f * div / nIter;
	int data_size = (N*M + HV * M) * sizeof(float);
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	cout << "Time = " << time << " mesc, "
		<< "Speed = " << speed << "GB/s" << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dV); printError(error);
	delete A;
}


template<int LBY, int LBX, int LTY, int LTX>
void testCorrectField(int height, int width)
{
	int stride = ((width + 3) >> 2) << 2;
	int lengthv = height * stride;
	printf("N, M, stride = (%d, %d, %d)\n", height, width, stride);

	float *A = newRandomFloatVec(lengthv, width, stride);
	float *dA = newDevFloatVec(A, lengthv);

	//CPU----------------------------------------------
	float *V1 = new float[stride];
	BinomialEachField(A, 1.0f, 2.0f, 3.0f, V1, height, width, stride);

	cout << "CPU: "; println(V1, 10);

	//GPU-----------------------------------------------
	cudaError_t error;

	//int HV = N << 1 >> LBY >> LTY ;
	int HV = nextFieldGridY(height, stride);
	cout << "HV = " << HV << endl;

	float *dV = newDevFloatVec(HV * stride);
	__field_binomial(0, dA, 1, 2, 3, height, stride, dV, width, stride, 1);

	error = cudaDeviceSynchronize(); printError(error);

	float* V2 = new float[stride];
	error = cudaMemcpy(V2, dV, sizeof(float)*stride, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: "; println(V2, 10);

	error = cudaGetLastError(); printError(error);
	//compare------------------------------------------------
	float sp = samePercent(V1, V2, stride); cout << "sp = " << sp << endl;
	float zp0 = zeroPercent(V1, stride); cout << "zp0 = " << zp0 << endl;
	float zp1 = zeroPercent(V2, stride); cout << "zp1 = " << zp1 << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dV); printError(error);

	delete A;
	delete V1;
	delete V2;

	if (sp != 1) {
		cout << "Error: N = " << height << ", M = " << width << endl;
		exit(2);
	}
}

void test()
{
	
	/*int N = 16, M = 16, width = 15;
	for (int n = 1; n <= 33; n++)
		for (int m = 4; m <= 128; m += 4)*/
			//testCorrect<5, 0, 0, 2>(n, m, m-1);

	//testCorrect<5, 0, 0, 2>(18, 12, 11);

	//17, 12, 204)
	//testCorrect<5, 0, 0, 2>(N, M, width);
	//testSpeed<5, 0, 0, 2>(1024, 1024, width);
	
	//int N = 64, M = 31;
	//testCorrectField<5, 0, 0, 2>(N, M);

	for (int n = 1; n <= 64; n++)
		for (int m = 4; m <= 32; m += 4) testCorrectField<5, 0, 0, 2>(n, m);
}
	


main() { test(); }
#endif