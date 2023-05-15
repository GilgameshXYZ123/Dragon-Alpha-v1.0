#pragma once

#define TEST_H
#ifndef TEST_H
#define TEST_H


#ifndef UTIL
#define UTIL

float* newRandomFloatVec(int length)//0-256
{
	float *p = new float[length];
	for (int i = 0; i < length; i++) p[i] = (float)(rand() % 1000) / 1000;
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


#ifndef POOL_2D_CPU
#define POOL_2D_CPU


void Pool2D_max_img2col(
	float* X, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
	int FH, int FW, //W[OC, IC, KH, KW] => B[GK, GM]
	float* Y, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
	int N, int IC,
	int sh, int sw, int ph, int pw)
{
	int GN = IC;
	int GM = N * OH * OW;
	int GK = FH * FW;
	for (int i = 0; i < GN; i++)
	{
		int ic = i;
		for (int j = 0; j < GM; j++)
		{
			int n = j / (OH*OW);
			int j_res = j % (OH*OW);
			int oh = j_res / OW, ow = j_res % OW;

			float v = FLOAT_MIN;
			for (int k = 0; k < GK; k++)
			{
				int fh = k / FW, fw = k % FW;
				int ih = oh * sh - ph + fh;
				int iw = ow * sw - pw + fw;

				if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

				//float x = X[n][ih][iw][ic];
				float x = get4d(X, n, ih, iw, ic, IH, IW, IC);
				if (v < x) v = x;
			}
			//Y[n][oh][ow][ic] = v;
			get4d(Y, n, oh, ow, ic, OH, OW, IC) = v;
		}
	}
}

void Pool2D_avg_img2col(
	float* X, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
	int FH, int FW, //W[OC, IC, KH, KW] => B[GK, GM]
	float* Y, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
	int N, int IC,
	int sh, int sw, int ph, int pw)
{
	int GN = IC;
	int GM = N * OH * OW;
	int GK = FH * FW;
	for (int i = 0; i < GN; i++)
	{
		int ic = i;
		for (int j = 0; j < GM; j++)
		{
			int n = j / (OH*OW);
			int j_res = j % (OH*OW);
			int oh = j_res / OW, ow = j_res % OW;

			float v = 0;
			int count = 0;
			for (int k = 0; k < GK; k++)
			{
				int fh = k / FW, fw = k % FW;
				int ih = oh * sh - ph + fh;
				int iw = ow * sw - pw + fw;

				if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

				//float x = X[n][ih][iw][ic];
				float x = get4d(X, n, ih, iw, ic, IH, IW, IC);
				v += x;
				count++;

			}
			//Y[n][oh][ow][ic] = v;
			get4d(Y, n, oh, ow, ic, OH, OW, IC) = v / count;
		}
	}
}


#endif

template<int LBY, int LBX>
void testCorrect(
	int IH, int IW,
	int FH, int FW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
{
	int OH = (IH + ph * 2 - FH) / sh + 1;
	int OW = (IW + pw * 2 - FW) / sw + 1;
	int GN = GET_GN(IC);
	int GM = GET_GM(N, OH, OW);
	int GK = GET_GK(FH, FW);

	printf("Test Correct:\n");
	printf("\t(IH, IW, OH, OW) = (%d, %d, %d, %d)\n", IH, IW, OH, OW);
	printf("\t(N, IC, sh, sw, ph, pw) = (%d, %d, %d, %d, %d, %d)\n", N, IC, sh, sw, ph, pw);
	printf("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);

	int sizeX = N * IC*IH*IW;
	int sizeY = N * IC*OH*OW;

	float *X = newRandomFloatVec(sizeX);
	float *Y1 = new float[sizeY];
	float *Y2 = new float[sizeY];

	//CPU------------------------------------------------
	//Pool2D_max_img2col(X, IH, IW, FH, FW, Y1, OH, OW, N, IC, sh, sw, ph, pw);
	Pool2D_avg_img2col(X, IH, IW, FH, FW, Y1, OH, OW, N, IC, sh, sw, ph, pw);
	cout << "CPU: "; println(Y1, 10);
	//float zp0 = zeroPercent(Y1, sizeY); cout << "zp0: " << zp0 << endl;

	//GPU------------------------------------------------
	float *dX = newDevFloatVec(X, sizeX);
	float *dY = newDevFloatVec(sizeY);
	cudaError_t error;
	jlong streams[8];
	for (int i = 0; i < 8; i++) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		streams[i] = (intptr_t) stream;
	}

	//----------------------------------------------------------------
	//kmax1(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
	//kmax2(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
	//kmax4(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
	//int index = 0; __pool2D_max(streams, index, 8, dX, IH, IW, FH, FW, dY, OH, OW, N, IC, sh, sw, ph, pw);

	//kavg1(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
	//kavg2(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
	//kavg4(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
	int index = 0; __pool2D_avg(streams, index, 8, dX, IH, IW, FH, FW, dY, OH, OW, N, IC, sh, sw, ph, pw);

	//----------------------------------------------------------------
	error = cudaMemcpy(Y2, dY, sizeof(float)*sizeY, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: "; println(Y2, 10);

	float sp = samePercent(Y1, Y2, sizeY); cout << "sp: " << sp << endl;
	float zp1 = zeroPercent(Y2, sizeY); cout << "zp1: " << zp1 << endl;

	error = cudaGetLastError(); printError(error);
	error = cudaFree(dX); printError(error);
	error = cudaFree(dY); printError(error);

	delete X;
	delete Y1;
	delete Y2;

	if (sp + zp1 < 0.999f) exit(2);
}

template<int LBY, int LBX>
void testSpeed(
	int IH, int IW,
	int FH, int FW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
{
	int OH = (IH + ph * 2 - FH) / sh + 1;
	int OW = (IW + pw * 2 - FW) / sw + 1;
	int GN = GET_GN(IC);
	int GM = GET_GM(N, OH, OW);
	int GK = GET_GK(FH, FW);

	printf("Test Correct:\n");
	printf("\t(IH, IW, OH, OW) = (%d, %d, %d, %d)\n", IH, IW, OH, OW);
	printf("\t(N, IC, sh, sw, ph, pw) = (%d, %d, %d, %d, %d, %d)\n", N, IC, sh, sw, ph, pw);
	printf("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);

	int sizeX = N * IC*IH*IW;
	int sizeY = N * IC*OH*OW;

	float *X = newRandomFloatVec(sizeX);
	float *dX = newDevFloatVec(X, sizeX);
	float *dY = newDevFloatVec(sizeY);
	cudaError_t error;
	jlong streams[8];
	for (int i = 0; i < 8; i++) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		streams[i] = (intptr_t)stream;
	}

	clock_t start = clock();
	int nIter = 500;
	for (int i = 0; i < nIter; i++)
	{
		//---------------------------------------------
		//kmax1(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
		//kmax2(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
		//kmax4(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
		//int index = 0; __pool2D_max(streams, index, 8, dX, IH, IW, FH, FW, dY, OH, OW, N, IC, sh, sw, ph, pw);

		//kavg1(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
		//kavg2(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
		//kavg4(NULL, LBY, LBX, dX, IH, IW, FH, FW, dY, OH, OW, sh, sw, ph, pw, GN, GM, 0, 0);
		int index = 0; __pool2D_avg(streams, index, 8, dX, IH, IW, FH, FW, dY, OH, OW, N, IC, sh, sw, ph, pw);
		//---------------------------------------------
	}
	error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = end - start;
	float time = 1.0f*div / nIter;
	float size = 1.0f* GN / 1024 * GM / 1024 * GK / 1024;
	float performance = (1024 * 1024 * size*1.0e-9f * 1024) / (time*1e-3f);

	printf("Size = %f, Time = %f msec, Performance = %f GFlop/s\n",
		size, time, performance);

	error = cudaGetLastError(); printError(error);
	error = cudaFree(dX); printError(error);
	error = cudaFree(dY); printError(error);
	delete X;
}


void test()
{
	//int IH = 64, IW = 64;
	int IH = 62, IW = 62;
	int N = 4, IC = 252;
	int FH = 4, FW = 4;
	int sh = 2, sw = 2, ph = 1, pw = 1;
	testCorrect<4, 1>(IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
	testSpeed<4, 1>(IH, IW, FH, FW, N, IC, sh, sw, ph, pw);

}

main() { 
	test(); 
}

#endif