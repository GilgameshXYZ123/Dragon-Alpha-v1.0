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

int *newDevIntVec(int length)
{
	int *dp = NULL;
	size_t size = sizeof(int)*length;
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


#ifndef UNPOOL_2D_AVERAGE_CPU
#define UNPOOL_2D_AVERAGE_CPU

void unpooling_avg_img2col_plus(
	float* deltaY, int OH, int OW,
	int FH, int FW,
	float* deltaX, int IH, int IW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
{
	int GN = IC;
	int GM = N * IH*IW;
	int OH_p = OH + (OH - 1)*(sh - 1), OW_p = OW + (OW - 1)*(sw - 1);
	int oph = FH - ph - 1, opw = FW - pw - 1;

	for (int i = 0; i < GN; i++)
	{
		int ic = i;
		for (int j = 0; j < GM; j++)
		{
			int n = j / (IH*IW);
			int j_res = j % (IH*IW);
			int ih = j_res / IW, iw = j_res % IW;

			int fw_s = 0, fh_s = 0;
			for (; fh_s < FH; fh_s++)
			{
				int oh = ih - oph + fh_s;
				if (oh < 0 || oh >= OH_p || oh % sh != 0) continue;
				for (fw_s = 0; fw_s < FW; fw_s++)
				{
					int ow = iw - opw + fw_s;
					if (ow >= 0 && ow < OW_p && ow % sw == 0) goto loop_end;
				}
			}
		loop_end:

			int FH_r = (FH - fh_s + sh - 1) / sh;
			int FW_r = (FW - fw_s + sw - 1) / sw;
			int GK_r = FH_r * FW_r;

			float v = 0;
			for (int k = 0; k < GK_r; k++)
			{
				int fh_r = k / FW_r, fw_r = k % FW_r;
				int fh = fh_r * sh + fh_s;
				int fw = fw_r * sw + fw_s;

				int oh = ih - oph + fh;
				int ow = iw - opw + fw;

				if (oh >= OH_p || ow >= OW_p) continue;
				oh /= sh; ow /= sw;

				int ih_min = oh * sh - ph, ih_max = ih_min + FH;
				if (ih_min < 0) ih_min = 0;
				if (ih_max >= IH) ih_max = IH;

				int iw_min = ow * sw - pw, iw_max = iw_min + FW;
				if (iw_min < 0) iw_min = 0;
				if (iw_max >= IW) iw_max = IW;

				int div = (ih_max - ih_min)*(iw_max - iw_min);

				//v += deltaY[n][oh][ow][ic] / div;
				v += get4d(deltaY, n, oh, ow, ic, OH, OW, IC) / div;
			}

			//deltaX[n][ic][ih][iw] = v;
			get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = v;
		}
	}
}

#endif


#ifndef UNPOOL_2D_MAX_CPU
#define UNPOOL_2D_MAX_CPU

void unpool2D_max_img2col_plus(
	float* deltaY, float* Y, int OH, int OW,
	int FH, int FW,
	float* deltaX, float* X, int IH, int IW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
{
	int GN = IC;
	int GM = N * IH*IW;
	int OH_p = OH + (OH - 1)*(sh - 1), OW_p = OW + (OW - 1)*(sw - 1);
	int oph = FH - ph - 1, opw = FW - pw - 1;

	for (int i = 0; i < GN; i++)
	{
		int ic = i;
		for (int j = 0; j < GM; j++)
		{
			int n = j / (IH*IW);
			int j_res = j % (IH*IW);
			int ih = j_res / IW, iw = j_res % IW;
			
			float x = get4d(X, n, ih, iw, ic, IH, IW, IC);//float x = X[n][ih][iw][ic];

			//--------------------------------------------------------------
			int fw_s = 0, fh_s = 0;
			for (; fh_s < FH; fh_s++)
			{
				int oh = ih - oph + fh_s;
				if (oh < 0 || oh >= OH_p || oh % sh != 0) continue;
				for (fw_s = 0; fw_s < FW; fw_s++)
				{
					int ow = iw - opw + fw_s;
					if (ow >= 0 && ow < OW_p && ow % sw == 0) goto LOOPX_end;
				}
			}
		LOOPX_end:

			int FH_r = (FH - fh_s + sh - 1) / sh;
			int FW_r = (FW - fw_s + sw - 1) / sw;
			int GK_r = FH_r * FW_r;
			//--------------------------------------------------------------

			float v = 0;
			for (int k = 0; k < GK_r; k++)
			{
				int fh_r = k / FW_r, fw_r = k % FW_r;
				int fh = fh_r * sh + fh_s, fw = fw_r * sw + fw_s;

				int oh = ih - oph + fh, ow = iw - opw + fw;
				if (oh >= OH_p || ow >= OW_p) continue;

				oh /= sh; ow /= sw;
				//float y = Y[n][ic][oh][ow];
				float y = get4d(Y, n, oh, ow, ic, OH, OW, IC);

				if (y > x) continue;
				//v += deltaY[n][oh][ow][ic];
				v += get4d(deltaY, n, oh, ow, ic, OH, OW, IC);
			}
			//deltaX[n][ih][iw][ic] = v;
			get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = v;
		}
	}
}

#endif


template<int LBY, int LBX>
void testCorrect(
	int OH, int OW,
	int FH, int FW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
{
	int IH = (OH - 1)*sh + FH - 2 * ph;
	int IW = (OW - 1)*sw + FW - 2 * pw;
	int GN = GET_GN(IC);
	int GM = GET_GM(N, IH, IW);
	int GK = GET_GK(FH, FW);

	cout << "Test Correct:" << endl;
	printf("(IH, IW, FH, FW, OH, OW)=(%d, %d, %d, %d, %d, %d)\n", IH, IW, FH, FW, OH, OW);
	printf("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
	printf("(N, IC) = (%d, %d)\n", N, IC);
	printf("(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);

	int sizeX = N * IC*IH*IW;
	int sizeY = N * IC*OH*OW;

	float *deltaY = newRandomFloatVec(sizeY);
	float *Y = newRandomFloatVec(sizeY);
	float *X = newRandomFloatVec(sizeX);

	float *deltaX1 = new float[sizeX];
	float *deltaX2 = new float[sizeX];

	//CPU--------------------------------------------------
	//unpooling_avg_img2col_plus(deltaY, OH, OW, FH, FW, deltaX1, IH, IW, N, IC, sh, sw, ph, pw);
	unpool2D_max_img2col_plus(deltaY, Y, OH, OW, FH, FW, deltaX1, X, IH, IW, N, IC, sh, sw, ph, pw);
	cout << "CPU: "; println(deltaX1, 10);

	//GPU--------------------------------------------------
	float *d_deltaY = newDevFloatVec(deltaY, sizeY);
	float *d_deltaX = newDevFloatVec(sizeX);
	float *dX = newDevFloatVec(X, sizeX);
	float *dY = newDevFloatVec(Y, sizeY);
	
	cudaError_t error;

	jlong streams[4];
	for (int i = 0; i < 4; i++) {
		cudaStream_t stream; cudaStreamCreate(&stream);
		streams[i] = (intptr_t)stream;
	}


	//----------------------------------------------------

	//kavg81(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);//A
	//kavg41(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);//A
	//kavg21(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);//A
	//kavg11(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);//A
	//int index = 0; __upool2D_avg(streams, index, 4, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, N, IC, sh, sw, ph, pw);
	//kavg_tiled4(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));
	//kavg_tiled2(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));
	//kavg_tiled1(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));
	//int index = 0; __unpool2D_avg_tiled(streams, index, 4, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, N, IC, sh, sw, ph, pw);

	kmax81(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
	//kmax41(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
	//kmax41(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
	//kmax21(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
	//kmax11(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
	//int index = 0; __upool2D_max(streams, index, 4, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, N, IC, sh, sw, ph, pw);
	//kmax_tiled4(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));
	//kmax_tiled2(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));
	//kmax_tiled1(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));

	//----------------------------------------------------
	error = cudaMemcpy(deltaX2, d_deltaX, sizeof(float)*sizeX, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: "; println(deltaX2, 10);
	

	//compare----------------------------------------------
	float sp = samePercent(deltaX1, deltaX2, sizeX);
	cout << "sp: " << sp << endl;
	float zp0 = zeroPercent(deltaX1, sizeX); cout << "zp0: " << zp0 << endl;
	float zp1 = zeroPercent(deltaX2, sizeX); cout << "zp1: " << zp1 << endl;

	error = cudaGetLastError(); printError(error);
	error = cudaFree(d_deltaY); printError(error);
	error = cudaFree(d_deltaX); printError(error);

	delete deltaY;
	delete Y;
	delete X;
	delete deltaX1;
	delete deltaX2;

	if (sp < 0.999f) { cout << "asdasdasdasdxx" << endl; exit(2); }
}

template<int LBY, int LBX>
void testSpeed(
	int OH, int OW,
	int FH, int FW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
{
	int IH = (OH - 1)*sh + FH - 2 * ph;
	int IW = (OW - 1)*sw + FW - 2 * pw;
	int GN = GET_GN(IC);
	int GM = GET_GM(N, IH, IW);
	int GK = GET_GK(FH, FW);

	cout << "Test Speed:" << endl;
	printf("(IH, IW, FH, FW, OH, OW)=(%d, %d, %d, %d, %d, %d)\n", IH, IW, FH, FW, OH, OW);
	printf("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
	printf("(N, IC) = (%d, %d)\n", N, IC);
	printf("(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);

	int sizeX = N * IC*IH*IW;
	int sizeY = N * IC*OH*OW;

	float *deltaY = newRandomFloatVec(sizeY);
	float *X = newRandomFloatVec(sizeX);
	float *Y = newRandomFloatVec(sizeY);

	float *d_deltaY = newDevFloatVec(deltaY, sizeY);
	float *dY = newDevFloatVec(Y, sizeY);
	int* dIndex = newDevIntVec(sizeY);

	float *d_deltaX = newDevFloatVec(sizeX);
	float *dX = newDevFloatVec(X, sizeX);
	cudaError_t error;

	jlong streams[4];
	for (int i = 0; i < 4; i++) {
		cudaStream_t stream; cudaStreamCreate(&stream);
		streams[i] = (intptr_t)stream;
	}

	clock_t start = clock();
	int nIter = 500;
	for (int i = 0; i < nIter; i++)
	{
		//kavg81(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);//A
		//kavg41(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);//A
		//kavg21(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);//A
		//kavg11(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);//A
		//int index = 0; __upool2D_avg(streams, index, 4, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, N, IC, sh, sw, ph, pw);
		//kavg_tiled4(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));
		//kavg_tiled2(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));
		//kavg_tiled1(NULL, LBY, LBX, 0, 0, d_deltaY, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));

		//kmax81(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		//kmax41(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		//kmax41(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		//kmax21(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		//kmax11(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		//int index = 0; __upool2D_max(streams, index, 4, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, N, IC, sh, sw, ph, pw);
		//kmax_tiled4(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));
		//kmax_tiled2(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));
		//kmax_tiled2(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));
		//kmax_tiled1(NULL, LBY, LBX, 0, 0, d_deltaY, dY, OH, OW, FH, FW, d_deltaX, dX, IH, IW, IC, sh, sw, ph, pw, GN, (N*OH*OW));

		//kmaxIdx81(NULL, LBY, LBX, 0, 0, d_deltaY, dIndex, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		//kmaxIdx41(NULL, LBY, LBX, 0, 0, d_deltaY, dIndex, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		//kmaxIdx21(NULL, LBY, LBX, 0, 0, d_deltaY, dIndex, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		kmaxIdx11(NULL, LBY, LBX, 0, 0, d_deltaY, dIndex, OH, OW, FH, FW, d_deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
	}
	error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = end - start;
	float time = 1.0f*div / nIter;
	float size = 1.0f * GN / 1024 * GM / 1024 * GK / 1024;
	float total = 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
	printf("Size = %f, Time = %f msec, Peformance = %f GFlop/s\n", size, time, performance);

	error = cudaGetLastError(); printError(error);
	error = cudaFree(d_deltaY); printError(error);
	error = cudaFree(dY); printError(error);
	error = cudaFree(d_deltaX); printError(error);
	error = cudaFree(dX); printError(error);

	delete X;
	delete Y;
	delete deltaY;
}

void test()
{
	int OH = 32, OW = 32;
	int N = 8, IC = 128;
	int FH = 4, FW = 4;
	int sh = 4, sw = 4, ph = 1, pw = 1;

	//[5, 2] 32
	//testCorrect<3, 2>(OH, OW, FH, FW, N, IC, sh, sw, ph, pw);
	//testSpeed<3, 2>(OH, OW, FH, FW, N, IC, sh, sw, ph, pw);

	//GM = N * IH * IW
	//int IH = (OH - 1)*sh + FH - 2 * ph;
	//int IW = (OW - 1)*sw + FW - 2 * pw;

	/*int OH = 31, OW = 31;
	int N = 5, IC = 64;
	int FH = 8, FW = 8;
	int sh = 3, sw = 3, ph = 2, pw = 2;*/

	//for (int ic = 4; ic <= 256; ic += 4) testCorrect<0, 0>(OH, OW, FH, FW, N, ic, sh, sw, ph, pw);

	//testCorrect<3, 2>(OH, OW, FH, FW, N, IC, sh, sw, ph, pw);
	testSpeed<3, 2>(OH, OW, FH, FW, N, IC, sh, sw, ph, pw);

}

main() 
{
	test();
}

#endif