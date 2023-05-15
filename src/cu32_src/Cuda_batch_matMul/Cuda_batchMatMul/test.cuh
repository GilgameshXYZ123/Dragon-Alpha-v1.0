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

float *newRandomPtr(int height, int width, int stride)
{
	int lengthv = height * stride;
	float *p = new float[lengthv];
	int index = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			p[index++] = (float)(rand() % 1000) / 1000 + 1;
		for (int j = width; j < stride; j++)
			p[index++] = 0;
	}
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

float zeroPercent(float *a, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
		if (a[i] == 0) sum++;
	return 1.0f*sum / length;
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
		if (dif < 1e-2) sum++;
		else {
			//if (a[i] < b[i]) cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
			//cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
		}

	}
	return 1.0f*sum / length;
}

#endif 


#ifndef BMM_CPU
#define BMM_CPU

void bmm(float *A, float *B, float *C,
	int Batch, int N, int CM, int BK, int AK)
{
	//C[Batch, N, M]
	//A[Batch, N, K]
	//B[Batch, K, M]
	for (int b = 0; b < Batch; b++) {
		for (int k = 0; k < BK; k++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < CM; j++)
					get3d(C, b, i, j, N, CM) +=
					get3d(A, b, i, k, N, AK) * get3d(B, b, k, j, BK, CM);
	}
}

void bmmT1(float *A, float *B, float *C,
	int Batch, int CN, int AN, int CM, int K)
{
	//C[Batch, N, M]
	//A[Batch, K, N] logically-> A^T[Batch, N, K]
	//B[Batch, K, M]
	for (int b = 0; b < Batch; b++) {
		for (int k = 0; k < K; k++)
			for (int i = 0; i < CN; i++)
				for (int j = 0; j < CM; j++)
					get3d(C, b, i, j, CN, CM) +=
					get3d(A, b, k, i, K, AN) * get3d(B, b, k, j, K, CM);
	}
}

void bmmT2(float *A, float *B, float *C,
	int Batch, int N, int CM, int BM, int K)
{
	//C[Batch, N, M]
	//A[Batch, N, K]
	//B[Batch, M, K]
	for (int b = 0; b < Batch; b++) {
		for (int k = 0; k < K; k++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < BM; j++)
					get3d(C, b, i, j, N, CM) +=
						get3d(A, b, i, k, N, K) * get3d(B, b, j, k, BM, K);
	}
}

#endif


#ifndef TEST_MATMUL
#define TEST_MATMUL

template<int LB, int MOVE_A, int MOVE_B>
void testCorrect(int Batch, int N, int M, int BK)
{
	int AK = (BK + 3) >> 2 << 2;
	int sizeA = Batch * N * AK;
	int sizeB = Batch * BK * M;
	int sizeC = Batch * N * M;

	cout << "testCorrect: " << endl
		<< "(Batch, N, M, BK, AK) = "
		<< Batch << ", " << M << ", " << BK << ", " << AK << endl;

	float *A = newRandomPtr(sizeA);
	float *B = newRandomPtr(sizeB);
	float *C1 = newPtr(sizeC);
	float *C2 = newPtr(sizeC);

	float *dA = newDevPtr(A, sizeA);
	float *dB = newDevPtr(B, sizeB);
	float *dC = newDevPtr(sizeC);

	cudaTextureObject_t texA = floatTexture(dA, sizeA);
	cudaError_t error;

	//CPU------------------------------
	bmm(A, B, C1, Batch, N, M, BK, AK);

	cout << "CPU: "; println(C1, 10);

	//GPU------------------------------
	//bmm_k88_ptex(NULL, LB, 0, 0, texA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k88_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k44_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k22_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BK, AK, N, CM);

	//bmm_u88_mk(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_u44_mk(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	bmm_u44_mk_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);

	//bmm_k88_mk(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k88(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k44(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k82(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k28(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k42(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k24(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k22(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k41(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	//bmm_k14(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);

	error = cudaStreamSynchronize(NULL);
	error = cudaMemcpy(C2, dC, sizeof(float)*sizeC, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: ";  println(C2, 10);

	//compare--------------------------

	float zp0 = zeroPercent(C1, sizeC); cout << "zp0 = " << zp0 << endl;
	float zp1 = zeroPercent(C2, sizeC); cout << "zp1 = " << zp1 << endl;
	float sp = samePercent(C1, C2, sizeC);
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

	if (sp != 1) {
		cout << "adadadasad" << endl;
		exit(2);
	}

	if (error != cudaSuccess) {
		cout << endl << N << " " << M << " " << BK << endl;
		exit(0);
	}
}

template<int LB, int MOVE_A, int MOVE_B>
void testSpeed(int nIter, int Batch, int N, int M, int BK)
{
	int AK = (BK + 3) >> 2 << 2;
	int sizeA = Batch * N * AK;
	int sizeB = Batch * BK * M;
	int sizeC = Batch * N * M;

	cout << "testSpeed: " << endl
		<< "(Batch, N, M, BK, AK) = "
		<< Batch << ", " << M << ", " << BK << ", " << AK << endl;

	float *A = newRandomPtr(sizeA);
	float *B = newRandomPtr(sizeB);

	float *dA = newDevPtr(A, sizeA);
	float *dB = newDevPtr(B, sizeB);
	float *dC = newDevPtr(sizeC);

	cudaTextureObject_t texA = floatTexture(dA, sizeA);
	cudaError_t error;

	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		//bmm_k88_ptex(NULL, LB, 0, 0, texA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_k88_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_k44_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_k22_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BK, AK, N, CM);

		//bmm_u88_mk(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_u44_mk(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		bmm_u44_mk_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);

		//bmm_k88_mk(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_k88(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_k44(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_k82(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_k28(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_k42(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_k24(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_k22(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		//bmm_k41(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
		///bmm_k14(NULL, LB, 0, 0, dA, dB, dC, Batch, N, M, BK, AK, N, M);
	}
	error = cudaDeviceSynchronize(); printError(error);

	clock_t end = clock();
	int div = end - start;
	float time = 1.0f*div / nIter;

	float size = 1.0f * Batch * N / 1024 * M / 1024 * BK / 1024;
	float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
	cout << "Size = " << size << ", Time = " << time << " msec, Performace = " << performance << " GFlop/s" << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dB); printError(error);
	error = cudaFree(dC); printError(error);
	error = cudaGetLastError(); printError(error);
}

#endif


#ifndef TEST_MATMUL_T1
#define TEST_MATMUL_T1

template<int LB, int MOVE_A, int MOVE_B>
void testCorrectT1(int Batch, int CN, int M, int K)
{
	int AN = (CN + 3) >> 2 << 2;
	int sizeA = Batch * K * AN;
	int sizeB = Batch * K * M;
	int sizeC = Batch * CN * M;

	cout << "testCorrect: " << endl
		<< "(Batch, CN, AN, M, K) = "
		<< Batch << ", " << CN << ", " << AN << ", " << M << ", " << K << endl;

	int stride = AN, width = CN;
	float *A = newRandomPtr(Batch*K, CN, AN);
	float *B = newRandomPtr(sizeB);
	float *C1 = newPtr(sizeC);
	float *C2 = newPtr(sizeC);

	float *dA = newDevPtr(A, sizeA);
	float *dB = newDevPtr(B, sizeB);
	float *dC = newDevPtr(sizeC);

	cudaTextureObject_t texA = floatTexture(dA, sizeA);
	cudaError_t error;

	//CPU------------------------------
	bmmT1(A, B, C1, Batch, CN, AN, M, K);

	cout << "CPU: "; println(C1, 10);

	//GPU------------------------------
	//bmmT1_k88_p(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	//bmmT1_k44_p(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	//bmmT1_k22_p(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);

	bmmT1_k88_mk(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	//bmmT1_k88(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	//bmmT1_k44(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	//bmmT1_k82(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	//bmmT1_k28(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	//bmmT1_k42(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	//bmmT1_k24(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	//bmmT1_k22(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	//bmmT1_k41(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	//bmmT1_k14(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);

	error = cudaStreamSynchronize(NULL);
	error = cudaMemcpy(C2, dC, sizeof(float)*sizeC, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: ";  println(C2, 10);

	//compare--------------------------

	float zp0 = zeroPercent(C1, sizeC); cout << "zp0 = " << zp0 << endl;
	float zp1 = zeroPercent(C2, sizeC); cout << "zp1 = " << zp1 << endl;
	float sp = samePercent(C1, C2, sizeC);
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

	if (sp != 1)
	//if ((sp + zp1)!= 1) 
	{
		cout << "adadadasad" << endl;
		exit(2);
	}

	if (error != cudaSuccess) {
		cout << endl << CN << " " << M << " " << K << endl;
		exit(0);
	}
}

template<int LB, int MOVE_A, int MOVE_B>
void testSpeedT1(int nIter, int Batch, int CN, int M, int K)
{
	int AN = (CN + 3) >> 2 << 2;
	int sizeA = Batch * K * AN;
	int sizeB = Batch * K * M;
	int sizeC = Batch * CN * M;

	cout << "testSpeed: " << endl
		<< "(Batch, CN, AN, M, K) = "
		<< Batch << ", " << CN << ", " << AN << ", " << M << ", " << K << endl;

	float *A = newRandomPtr(Batch*K, CN, AN);
	float *B = newRandomPtr(sizeB);

	float *dA = newDevPtr(A, sizeA);
	float *dB = newDevPtr(B, sizeB);
	float *dC = newDevPtr(sizeC);

	cudaTextureObject_t texA = floatTexture(dA, sizeA);
	cudaError_t error;

	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		//bmmT1_k88_p(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
		//bmmT1_k44_p(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
		//bmmT1_k22_p(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);

		bmmT1_k88_mk(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
		//bmmT1_k88(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
		//bmmT1_k44(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
		//bmmT1_k82(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
		//bmmT1_k28(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
		//bmmT1_k42(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
		//bmmT1_k24(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
		//bmmT1_k22(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
		//bmmT1_k41(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
		//bmmT1_k14(NULL, LB, 0, 0, dA, dB, dC, Batch, CN, AN, M, K, CN, M);
	}
	error = cudaDeviceSynchronize(); printError(error);

	clock_t end = clock();
	int div = end - start;
	float time = 1.0f*div / nIter;

	float size = 1.0f * Batch * CN / 1024 * M / 1024 * K / 1024;
	float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
	cout << "Size = " << size << ", Time = " << time << " msec, Performace = " << performance << " GFlop/s" << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dB); printError(error);
	error = cudaFree(dC); printError(error);
	error = cudaGetLastError(); printError(error);
}

#endif


#ifndef TEST_MATMUL_T2
#define TEST_MATMUL_T2

template<int LB, int MOVE_A, int MOVE_B>
void testCorrectT2(int Batch, int N, int BM, int K)
{
	int CM = (BM + 3) >> 2 << 2;
	int sizeA = Batch * N * K;
	int sizeB = Batch * BM * K;
	int sizeC = Batch * N * CM;

	cout << "testCorrect: " << endl
		<< "(Batch, N, CM, BM, K) = "
		<< Batch << ", " << N <<", " << CM << ", " << BM << ", " << K << endl;

	float *A = newRandomPtr(sizeA);
	float *B = newRandomPtr(sizeB);
	float *C1 = newPtr(sizeC);
	float *C2 = newPtr(sizeC);

	float *dA = newDevPtr(A, sizeA);
	float *dB = newDevPtr(B, sizeB);
	float *dC = newDevPtr(sizeC);

	cudaTextureObject_t texA = floatTexture(dA, sizeA);
	cudaTextureObject_t texB = floatTexture(dB, sizeB);
	cudaError_t error;

	//CPU------------------------------
	bmmT2(A, B, C1, Batch, N, CM, BM, K);

	cout << "CPU: "; println(C1, 10);

	//GPU------------------------------
	//bmmT2_k88_ptex(NULL, LB, 0, 0, texA, texB, dC, Batch, N, CM, BM, K, N, CM);
	bmmT2_k88_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	//bmmT2_k44_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	//bmmT2_k22_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);

	//bmmT2_u88(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	//bmmT2_u44(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	//bmmT2_u44_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);

	//bmmT2_k88(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	//bmmT2_k44(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	//bmmT2_k82(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	//bmmT2_k28(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	//bmmT2_k42(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	//bmmT2_k24(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	//bmmT2_k22(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);

	//bmmT2_k41(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	//bmmT2_k14(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);

	error = cudaStreamSynchronize(NULL);
	error = cudaMemcpy(C2, dC, sizeof(float)*sizeC, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: ";  println(C2, 10);

	//compare--------------------------

	float zp0 = zeroPercent(C1, sizeC); cout << "zp0 = " << zp0 << endl;
	float zp1 = zeroPercent(C2, sizeC); cout << "zp1 = " << zp1 << endl;
	float sp = samePercent(C1, C2, sizeC);
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

	if (sp != 1) {
		cout << "adadadasad" << endl;
		exit(2);
	}

	if (error != cudaSuccess) {
		cout << endl << N << " " << BM << " " << K << endl;
		exit(0);
	}
}

template<int LB, int MOVE_A, int MOVE_B>
void testSpeedT2(int nIter, int Batch, int N, int BM, int K)
{
	int CM = (BM + 3) >> 2 << 2;
	int sizeA = Batch * N * K;
	int sizeB = Batch * BM * K;
	int sizeC = Batch * N * CM;

	cout << "testCorrect: " << endl
		<< "(Batch, N, CM, BM, K) = "
		<< Batch << ", " << N << ", " << CM << ", " << BM << ", " << K << endl;

	float *A = newRandomPtr(sizeA);
	float *B = newRandomPtr(sizeB);

	float *dA = newDevPtr(A, sizeA);
	float *dB = newDevPtr(B, sizeB);
	float *dC = newDevPtr(sizeC);

	cudaTextureObject_t texA = floatTexture(dA, sizeA);
	cudaTextureObject_t texB = floatTexture(dB, sizeB);
	cudaError_t error;

	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		//bmmT2_k88_ptex(NULL, LB, 0, 0, texA, texB, dC, Batch, N, CM, BM, K, N, CM);
		bmmT2_k88_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
		//bmmT2_k44_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
		//bmmT2_k22_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);

		//bmmT2_u88(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
		//bmmT2_u44(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
		//bmmT2_u44_p(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);

		//bmmT2_k88(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
		//bmmT2_k44(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
		//bmmT2_k82(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
		//bmmT2_k28(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
		//bmmT2_k42(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
		//bmmT2_k24(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);

		//bmmT2_k22(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
		//bmmT2_k41(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
		//bmmT2_k14(NULL, LB, 0, 0, dA, dB, dC, Batch, N, CM, BM, K, N, CM);
	}
	error = cudaDeviceSynchronize(); printError(error);

	clock_t end = clock();
	int div = end - start;
	float time = 1.0f*div / nIter;

	float size = 1.0f * Batch * N / 1024 * CM / 1024 * K / 1024;
	float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
	cout << "Size = " << size << ", Time = " << time << " msec, Performace = " << performance << " GFlop/s" << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dB); printError(error);
	error = cudaFree(dC); printError(error);
	error = cudaGetLastError(); printError(error);
}

#endif

main()
{
	int Batch = 64, N = 256, M = 256, K = 256;
	//int Batch = 64, N = 252, M = 256, K = 256;//for bmmT, K = 255
	//int Batch = 64, N = 252, M = 255, K = 256;//for bmmT2, K = 255

	//int Batch = 16, N = 24, M = 24, K = 8;

	//testCorrect<3, 1, 1>(Batch, N, M, K);
	//testCorrect<4, 1, 1>(Batch, N, M, K - 16);
	//testCorrect<3, 1, 1>(Batch, N, M, K + 8);
	//testCorrect<4, 1, 1>(Batch, N, M, K);
	//testSpeed<3, 1, 1>(500, Batch, N, M, K);
	
	//testCorrectT1<4, 1, 1>(Batch, N, M, K);
	//testCorrectT1<4, 1, 1>(Batch, N, M, 3);
	//testSpeedT1<4, 1, 1>(500, Batch, N, M, K);

	testCorrectT2<4, 1, 1>(Batch, N, M, K);
	//testCorrectT2<4, 1, 1>(Batch, N, M, K - 16);
	testCorrectT2<3, 1, 1>(Batch, N, M, K - 4);
	testSpeedT2<4, 1, 1>(500, Batch, N, M, K);

	//for (int N = 4; N <= 64; N += 4)
		//for (int M = 4; M <= 96; M += 4)
			//testCorrectT1<3, 1, 1>(16, N, M, 16);
}

#endif