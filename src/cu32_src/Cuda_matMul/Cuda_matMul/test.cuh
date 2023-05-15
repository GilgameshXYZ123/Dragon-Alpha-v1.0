#pragma once

#define TEST_H
#ifndef TEST_H
#define TEST_H


#ifndef UTIL
#define UTIL

float *newPtr(int length)
{
	float *p = new float[length];
	memset(p, 0, sizeof(float)*length);
	return p;
}

float *newRandomPtr(int length)
{
	float *p = new float[length];
	for (int i = 0; i < length; i++) p[i] = 1.0f*(rand() % 1000) / 1000;
	return p;
}

float *newDevPtr(float *p, int length)
{
	float *dp = NULL;
	cudaError_t error = cudaMalloc((void**)&dp, length * sizeof(float)); printError(error);
	cudaMemcpy(dp, p, length * sizeof(float), cudaMemcpyHostToDevice); printError(error);
	return dp;
}

float *newDevPtr(int length)
{
	float *dp = NULL;
	cudaError_t error = cudaMalloc((void**)&dp, length * sizeof(float)); printError(error);
	error = cudaMemset(dp, 0, sizeof(float)*length); printError(error);
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
		float dif = fabs(a[i] - b[i]) / fabs(a[i] + b[i]);
		if (dif < 1e-5) sum++;
		//else cout << dif<<" " << a[i] << ", " << b[i] << endl;
	}
	return 1.0f*sum / length;
}

void multiply(float *A, float *B, float *C, int N, int M, int K)
{
	for (int k = 0; k < K; k++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++)
				get(C, i, j, M) += get(A, i, k, K) * get(B, k, j, M);
}

void multiplyT1(float *A, float *B, float *C, int N, int M, int K)
{
	for (int k = 0; k < K; k++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++) 
				get(C, i, j, M) += get(A, k, i, N) * get(B, k, j, M);
}

void multiplyT2(float *A, float *B, float *C, int N, int M, int K)
{
	for (int k = 0; k < K; k++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++)
				get(C, i, j, M) += get(A, i, k, K) * get(B, j, k, K);
}
#endif 


template<int LB>
void testCorrect(int N, int M, int K)
{
	printf("(N, M, K) = (%d, %d, %d)\n", N, M, K);
	
	float *A = newRandomPtr(N*K);
	float *B = newRandomPtr(K*M);
	float *C1 = newPtr(N*M);
	float *C2 = newPtr(N*M);

	float *dA = newDevPtr(A, N*K);
	float *dB = newDevPtr(B, K*M);
	float *dC = newDevPtr(N*M);
	cudaError_t error;

	//CPU------------------------------
	//multiply(A, B, C1, N, M, K);
	//multiplyT1(A, B, C1, N, M, K);
	multiplyT2(A, B, C1, N, M, K);

	cout << "CPU: "; println(C1, 10);

	int GZ = K >> 10; 
	int part = GZ - 1; 
	int size_Cbuf = part * N*M; 
	int K_slice = 0;
	float *dCbuf = NULL; 
	if (part > 0) {
		dCbuf = newDevPtr(size_Cbuf);
		K_slice = SK_K_slice(K, GZ);
	}

	cout << "GZ = " << GZ << endl;
	cout << "K_slice = " << K_slice << endl;
	cout << "sizeCbuf: " << size_Cbuf << endl;

	//GPU------------------------------
	//matMul===========================================
	{
		//u88_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//u84_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//u48_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//u44_mgk(LB, NULL, dA, dB, dC, N, M, K, M);

		//k88(LB, NULL, dA, dB, dC, N, M, K, M);
		//k84(LB, NULL, dA, dB, dC, N, M, K, M);
		//k48(LB, NULL, dA, dB, dC, N, M, K, M);
		//k44(LB, NULL, dA, dB, dC, N, M, K, M);
		//k82(LB, NULL, dA, dB, dC, N, M, K, M);
		//k28(LB, NULL, dA, dB, dC, N, M, K, M);
		//k42(LB, NULL, dA, dB, dC, N, M, K, M);
		//k24(LB, NULL, dA, dB, dC, N, M, K, M);

		//k22(LB, NULL, dA, dB, dC, N, M, K, M);

		//k81(LB, NULL, dA, dB, dC, N, M, K, M);
		//k41(LB, NULL, dA, dB, dC, N, M, K, M);
		//k21(LB, NULL, dA, dB, dC, N, M, K, M);
		//s8x2_1(LB, NULL, dA, dB, dC, N, M, K, M);
		//s4x2_1(LB, NULL, dA, dB, dC, N, M, K, M);
		//s2x2_1(LB, NULL, dA, dB, dC, N, M, K, M);

		//k18(LB, NULL, dA, dB, dC, N, M, K, M);
		//k14(LB, NULL, dA, dB, dC, N, M, K, M);
		//k12(LB, NULL, dA, dB, dC, N, M, K, M);
	}
	
	//matMulT1=========================================
	{
		//matMulT1=========================================
		//k88T1(LB, NULL, dA, dB, dC, N, M, K, N, M);
		//k84T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k48T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k82T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k28T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k81T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k18T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k44T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k42T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k24T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k22T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
	}

	//matMulT1 SK======================================
	{
		//k88T1SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k88T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k84T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k48T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k44T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k82T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k28T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k42T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k24T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k22T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k21T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k12T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k11T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);

		//if (GZ > 1) SKbuf_summary(NULL, dCbuf, dC, part, N * M);
	}

	//matMul T2========================================
	{
		//matMulT2=========================================
		//u88T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//u84T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//u48T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);

		//k88T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k84T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//sk48T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k82T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k28T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k44T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k42T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k24T2(LB, NULL, dA, dB, dC, N, M, K, M);

		//k22T2(LB, NULL, dA, dB, dC, N, M, K, M);
		
		//k81T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k18T2(LB, NULL, dA, dB, dC, N, M, K, M);
	}

	//compare--------------------------
	error = cudaStreamSynchronize(NULL);
	error = cudaMemcpy(C2, dC, sizeof(float)*N*M, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: ";  println(C2, 10);

	float sp = samePercent(C1, C2, M*N);
	cout << "Same Percent: " << sp << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dB); printError(error);
	error = cudaFree(dC); printError(error);
	error = cudaGetLastError();
	cout << error << ":" << cudaGetErrorName(error) << endl
		<< cudaGetErrorString(error) << endl;

	delete A;
	delete B;
	delete C1;
	delete C2;

	if (sp < 0.999) exit(-2);

	if (error != cudaSuccess) {
		cout << endl << N << " " << M << " " << K << endl;
		exit(0);
	}
}

template<int LB>
void testSpeed(int nIter, int N, int M, int K)
{
	printf("(N, M, K) = (%d, %d, %d)\n", N, M, K);

	float *A = newRandomPtr(N*K);
	float *B = newRandomPtr(K*M);

	float *dA = newDevPtr(A, N*K);
	float *dB = newDevPtr(B, K*M);
	float *dC = newDevPtr(N*M);

	int GZ = K >> 10;
	int part = GZ - 1;
	int size_Cbuf = part * N*M;
	int K_slice = 0;
	float *dCbuf = NULL;
	if (part > 0) {
		K_slice = SK_K_slice(K, GZ);
		dCbuf = newDevPtr(size_Cbuf);
	}

	cout << "GZ = " << GZ << endl;
	cout << "K_slice = " << K_slice << endl;
	cout << "sizeCbuf: " << size_Cbuf << endl;

	cudaError_t error;
	
	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		//matMul==================================
		{
			//u88_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//u84_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//u48_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//u44_mgk(LB, NULL, dA, dB, dC, N, M, K, M);

			//k88(LB, NULL, dA, dB, dC, N, M, K, M);
			//k84(LB, NULL, dA, dB, dC, N, M, K, M);
			//k48(LB, NULL, dA, dB, dC, N, M, K, M);
			//k44(LB, NULL, dA, dB, dC, N, M, K, M);
			//k82(LB, NULL, dA, dB, dC, N, M, K, M);
			//k28(LB, NULL, dA, dB, dC, N, M, K, M);
			//k42(LB, NULL, dA, dB, dC, N, M, K, M);
			//k24(LB, NULL, dA, dB, dC, N, M, K, M);

			//k22(LB, NULL, dA, dB, dC, N, M, K, M);
			
			//k81(LB, NULL, dA, dB, dC, N, M, K, M);
			//k41(LB, NULL, dA, dB, dC, N, M, K, M);
			//k21(LB, NULL, dA, dB, dC, N, M, K, M);
			//s8x2_1(LB, NULL, dA, dB, dC, N, M, K, M);
			//s4x2_1(LB, NULL, dA, dB, dC, N, M, K, M);
			//s2x2_1(LB, NULL, dA, dB, dC, N, M, K, M);

			//k18(LB, NULL, dA, dB, dC, N, M, K, M);
			//k14(LB, NULL, dA, dB, dC, N, M, K, M);
			//k12(LB, NULL, dA, dB, dC, N, M, K, M);
		}
		
		//matMulT1================================
		{
			//matMulT1================================
			//k88T1(LB, NULL, dA, dB, dC, N, M, K, N, M);
			//k84T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k48T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k82T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k28T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k81T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k18T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k44T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k42T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k24T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k22T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k88T1_mgk(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k84T1_mgk(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k48T1_mgk(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k82T1_mgk(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k28T1_mgk(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k81T1_mgk(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k18T1_mgk(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k44T1_mgk(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k42T1_mgk(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k24T1_mgk(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k22T1_mgk(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		}

		//matMulT1 SK=============================
		{
			//k88T1SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k88T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k84T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k48T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k44T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k82T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k28T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k42T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k24T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k22T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k21T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k12T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k11T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);

			//if (GZ > 1) SKbuf_summary(NULL, dCbuf, dC, part, N * M);
		}
		
		//matMulT2================================
		{
			//matMulT2================================
			//u88T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//u84T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//u48T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);

			//k88T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k84T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k48T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k82T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k28T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k44T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k42T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k24T2(LB, NULL, dA, dB, dC, N, M, K, M);

			//k22T2(LB, NULL, dA, dB, dC, N, M, K, M);

			//k81T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k18T2(LB, NULL, dA, dB, dC, N, M, K, M);
		}
	}
	error = cudaDeviceSynchronize(); printError(error);

	clock_t end = clock();
	int div = end - start;
	float time = 1.0f*div / nIter;

	float size = 1.0f * N / 1024 * M / 1024 * K / 1024;
	float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
	cout << "Size = " << size << ", Time = " << time << " msec, Performace = " << performance << " GFlop/s" << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dB); printError(error);
	error = cudaFree(dC); printError(error);
	error = cudaGetLastError(); printError(error);
}

main() 
{ 
	//int N = 1024, M = 1024, K = 1024;
	//int N = 512, M = 2048, K = 1024;
	//int N = 512, M = 128, K = 512;
	//int N = 256, M = 256, K = 1024 * 16;

	//int N = 1024 * 16, M = 16, K = 1024;
	//int N = 1024 * 16, M = 8, K = 2048;
	//int N = 1024 * 16, M = 4, K = 4096;

	//int N = 16, M = 1024 * 16, K = 1024;
	//int N = 8, M = 1024 * 16, K = 2048;
	//int N = 4, M = 1024 * 16, K = 2048;

	//testCorrect<4>(N, M, K);
	//testCorrect<4>(N, M, K - 16);
	//testCorrect<3>(N, M, K - 8);
	//testCorrect<4>(N, M, K - 1);
	//testCorrect<4>(N, M, K + 1);
	//testSpeed<4>(1000, N, M, K);

	float* a = new float[10000];
	int offset = 32768;

	float* a1 = a + offset;
	intptr_t address1 = (intptr_t)a1;
	intptr_t address2 = (intptr_t)a + offset * 4;
	long long address3 = (long long)a + offset * 4;
	
	cout << "address1 = " << address1 << endl;
	cout << "address2 = " << address2 << endl;
	cout << "address3 = " << address3 << endl;
 }

#endif