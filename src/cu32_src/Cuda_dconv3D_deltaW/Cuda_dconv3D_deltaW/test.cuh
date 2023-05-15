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
		//float dif = fabs(a[i] - b[i]);
		float dif = fabs((a[i] - b[i]) / (a[i] + b[i]));
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


#ifndef DECONV_3D_DELTAW_CPU
#define DECONV_3D_DELTAW_CPU

void deconv3D_deltaW_img2col2(
	float* X, int IH, int IW,
	float* deltaY, int OH, int OW,
	float* deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int GN = OC;
	int GM = IC * FH * FW;
	int GK = N * OH * OW;
	int oph = ph, opw = pw;

	for (int i = 0; i < GN; i++)
	{
		int oc = i;
		for (int j = 0; j < GM; j++)
		{
			int ic = j / (FH*FW);
			int j_res = j % (FH*FW);
			int fh = j_res / FW, fw = j_res % FW;

			float v = 0;
			for (int k = 0; k < GK; k++)
			{
				/*int oh = k / (OW * N);
				int k_res = k % (OW*N);
				int ow = k_res / N, n = k_res % N;*///equivalent
				int n = k / (OH*OW);
				int k_res = k % (OH*OW);
				int oh = k_res / OW, ow = k_res % OW;

				int ih = fh - oph + (oh*sh);
				int iw = fw - opw + (ow*sw);

				if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
				v += get4d(deltaY, n, oh, ow, oc, OH, OW, OC) *
					get4d(X, n, ih, iw, ic, IH, IW, IC);
				//v += X[n][ih][iw][ic] * deltaY[n][oh][ow][oc];
			}
			get4d(deltaW, oc, fh, fw, ic, FH, FW, IC) = v;
		}
	}
}

#endif 


template<int LB>
void testCorrect(
	int IH, int IW,
	int OH, int OW,
	int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	if (IH == -1) IH = (OH - 1)*sh + FH - 2 * ph;
	if (IW == -1) IW = (OW - 1)*sw + FW - 2 * pw;

	int OH_p = OH + (OH - 1)*(sh - 1), OW_p = OW + (OW - 1)*(sw - 1);
	int GN = OC;
	int GM = IC * FH*FW;
	int GK0 = N * OH_p*OW_p;
	int GK = N * OH*OW;

	printf("Test correct:\n");
	printf("\t      X(N , IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
	printf("\t deltaW(OC, IC, OH, OW) = (%d, %d, %d, %d)\n", OC, IC, OH, OW);
	printf("\t deltaY(N , OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
	printf("\t (OH_p, OW_p) = (%d, %d)\n", OH_p, OW_p);
	printf("\t (GN, GM, GK0, GK) = (%d, %d, %d, %d)\n", GN, GM, GK0, GK);

	int sizeX = N *IH*IW*IC;
	int sizeW = OC *FH*FW*IC;
	int sizeY = N *OH*OW*OC;

	float *X = newRandomFloatVec(sizeX);
	float *deltaY = newRandomFloatVec(sizeY);
	float *deltaW1 = new float[sizeW];
	float *deltaW2 = new float[sizeW];

	//CPU---------------------------------------------------
	deconv3D_deltaW_img2col2(X, IH, IW, deltaY, OH, OW, deltaW1, FH, FW, N, IC, OC, sh, sw, ph, pw);
	cout << "CPU: "; println(deltaW1, 10);

	//GPU---------------------------------------------------
	float *dX = newDevFloatVec(X, sizeX);
	float *d_deltaY = newDevFloatVec(deltaY, sizeY);
	float *d_deltaW = newDevFloatVec(sizeW);
	cudaError_t error;

	cudaTextureObject_t texX = float4Texture(dX, sizeX);

	//======[pre process for GEMMSK]=================================
	int GZ = (GK) >> 8; if (GZ > 32) GZ = 32;//MAX_GZ = 32
	int part = GZ - 1;//part = GZ - 1
	int GK_slice = 0;
	float *d_deltaW_buf = NULL;
	if (part > 0) {
		GK_slice = GEMMSK_GK_slice(GK, GZ);
		d_deltaW_buf = newDevFloatVec(sizeW * part);
	}

	cout << "GZ = " << GZ << endl;
	cout << "GK_slice = " << GK_slice << endl;
	cout << "sizeW_buf: " << sizeW << ":" << part << ":" << sizeW * part << endl;
	//======[pre process for GEMMSK]=================================

	float Q = PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW, sh, sw);
	cout << "Q = " << Q << endl;

	//------------------------------------------------------
	//Gemm SK Area==========================================
	{
		//GemmV2SK_EX2
		{
			//kGemmV2SK88O2P1_LB4(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, IC);
			//kGemmV2SK88O2P1_LB3(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, IC);

			//kGemmV2SK88O4P1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, IC);
			kGemmV2SK88O4P1_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, IC);

			//kGemmV2SK88O7P1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, IC);
			//kGemmV2SK88O7P1_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, IC);

			//kGemmV2SK88O8P1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, IC);
			//kGemmV2SK88O8P1_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, IC);
		}

		//GemmV2SK EX
		{
			//kGemmV2SK88_n2pow_LB4(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK88_n2pow_LB3(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);

			//kGemmV2SK84_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK48_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK44_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK82_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK28_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK42_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK24_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
		}

		//GemmV2SK
		{
			//kGemmV2SK88(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK84(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK48(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK44(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK82(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK28(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK42(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK24(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);

			//kGemmV2SK22(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);

			//kGemmV2SK81(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK41(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK21(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//sGemmV2SK_8x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//sGemmV2SK_4x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//sGemmV2SK_2x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);

			//kGemmV2SK14(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK12(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//kGemmV2SK11(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//sGemmV2SK_1_4x2(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			//sGemmV2SK_1_2x2(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
		}
	
		//GemmSK_EX
		{
			//fGemmSK88(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, 8, 8, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//fGemmSK88(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, 7, 7, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);

			//uGemmSK88_ohw2pow_LB4(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//uGemmSK88_ohw2pow_LB3(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//uGemmSK88_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GM);
			
			//kGemmSK88_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK88_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK84_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK48_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK44_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK82_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK28_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK42_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK24_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		}

		//GemmSK
		{
			//kGemmSK88(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK84(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK48(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK82(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK28(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK44(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK42(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK24(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);

			//kGemmSK22(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);

			//kGemmSK81(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK41(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK21(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//sGemmSK_8x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//sGemmSK_4x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//sGemmSK_2x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);

			//kGemmSK14(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK12(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemmSK11(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//sGemmSK_1_4x2(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			//sGemmSK_1_2x2(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			
		}
	
		//GemmSK W1
		{
			//kGemmSK88W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//kGemmSK84W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//kGemmSK48W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//kGemmSK44W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//kGemmSK82W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//kGemmSK28W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//kGemmSK42W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//kGemmSK24W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);

			//kGemmSK22W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);

			//kGemmSK81W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//kGemmSK41W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//kGemmSK21W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//sGemmSK_8x2_1_W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//sGemmSK_4x2_1_W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//sGemmSK_2x2_1_W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);

			//kGemmSK14W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//kGemmSK12W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//kGemmSK11W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//sGemmSK_1_4x2_W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			//sGemmSK_1_2x2_W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
		}

		if (GZ > 1) {
			//general_buf_summary(NULL, d_deltaW_buf, d_deltaW, part, sizeW);
			buf_summary(NULL, d_deltaW_buf, d_deltaW, part, sizeW);
		}
	}

	//Gemm Area=============================================
	{
		//Gemm EX
		{
			//fGemm88(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, 2, 2, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//fGemm88(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, 3, 3, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//fGemm88(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, 5, 5, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//fGemm88(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, 7, 7, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//kGemm88_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//kGemm84_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, log2(OH), log2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm48_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//kGemm44_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm82_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, log2(OH), log2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm28_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, log2(OH), log2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}

		//Gemm
		{
			//kGemm88(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW,  N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//kGemm84(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW,  N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//kGemm48(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//kGemm44(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm82(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm28(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm42(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm24(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm22(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm41(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm14(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm21(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm12(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//kGemm11(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}
	
		//Gemm W1
		{
			//k44W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
			//k42W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
			//k24W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
			//k22W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
			//k21W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
			//k12W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
			//k11W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
		}
	}
	//------------------------------------------------------

	error = cudaMemcpy(deltaW2, d_deltaW, sizeof(float)*sizeW, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: "; println(deltaW2, 10);

	//compare-----------------------------------------------s
	float sp = samePercent(deltaW1, deltaW2, sizeW); cout << "sp: " << sp << endl;
	float zp0 = zeroPercent(deltaW1, sizeW); cout << "zp0: " << zp0 << endl;
	float zp1 = zeroPercent(deltaW2, sizeW); cout << "zp1: " << zp1 << endl;

	error = cudaGetLastError(); printError(error);
	error = cudaFree(dX); printError(error);
	error = cudaFree(d_deltaY); printError(error);
	error = cudaFree(d_deltaW); printError(error);

	delete deltaW1;
	delete deltaW2;
	delete deltaY;
	delete X;

	if (sp != 1) { cout << "asdasdasdasdasda"; exit(2); }
}


template<int LB>
void testSpeed(int nIter, 
	int IH, int IW,
	int OH, int OW,
	int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	if (IH == -1) IH = (OH - 1)*sh + FH - 2 * ph;
	if (IW == -1) IW = (OW - 1)*sw + FW - 2 * pw;

	int OH_p = OH + (OH - 1)*(sh - 1), OW_p = OW + (OW - 1)*(sw - 1);
	int GN = OC;
	int GM = IC * FH*FW;
	int GK0 = N * OH_p*OW_p;
	int GK = N * OH*OW;

	printf("Test Speed:\n");
	printf("\t      X(N , IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
	printf("\t deltaW(OC, IC, OH, OW) = (%d, %d, %d, %d)\n", OC, IC, OH, OW);
	printf("\t deltaY(N , OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
	printf("\t (OH_p, OW_p) = (%d, %d)\n", OH_p, OW_p);
	printf("\t (GN, GM, GK0, GK) = (%d, %d, %d, %d)\n", GN, GM, GK0, GK);

	int sizeX = N * IC*IH*IW;
	int sizeW = OC * IC*FH*FW;
	int sizeY = N * OC*OH*OW;

	float *X = newRandomFloatVec(sizeX);
	float *deltaY = newRandomFloatVec(sizeY);

	float *dX = newDevFloatVec(X, sizeX);
	float *d_deltaY = newDevFloatVec(deltaY, sizeY);
	float *d_deltaW = newDevFloatVec(sizeW);
	cudaError_t error;

	cudaTextureObject_t texX = float4Texture(dX, sizeX);

	int GZ = (GK) >> 10; if (GZ > 32) GZ = 32;
	int part = GZ - 1;//part = GZ - 1
	int GK_slice = 0;
	float *d_deltaW_buf = NULL;
	if (part > 0) {
		GK_slice = GEMMSK_GK_slice(GK, GZ);
		d_deltaW_buf = newDevFloatVec(sizeW * part);
	}

	cout << "GZ = " << GZ << endl;
	cout << "GK_slice = " << GK_slice << endl;
	cout << "sizeW_buf: " << sizeW << ":" << part << ":" << sizeW * part << endl;

	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{

		//GemmSK Area=======================================
		{
			//GemmSK V2 EX2
			{
				//kGemmV2SK88O2P1_LB4(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, IC);
				//kGemmV2SK88O2P1_LB3(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, IC);

				//kGemmV2SK88O4P1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, IC);
				kGemmV2SK88O4P1_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, IC);

				//kGemmV2SK88O7P1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, IC);
				//kGemmV2SK88O7P1_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, IC);

				//kGemmV2SK88O8P1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, IC);
				//kGemmV2SK88O8P1_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, IC);
			}

			//GemmSK V2 EX
			{
				//kGemmV2SK88_n2pow_LB4(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK88_n2pow_LB3(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);

				//kGemmV2SK84_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK48_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK44_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK82_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK28_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK42_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK24_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, IC);
			}

			//GemmV2 SK
			{
				//kGemmV2SK88(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK84(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK48(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK44(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK82(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK28(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK42(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK24(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);

				//kGemmV2SK22(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);

				//kGemmV2SK81(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK41(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK21(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//sGemmV2SK_8x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//sGemmV2SK_4x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//sGemmV2SK_2x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);

				//kGemmV2SK14(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK12(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//kGemmV2SK11(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//sGemmV2SK_1_4x2(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
				//sGemmV2SK_1_2x2(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, IC);
			}

			//GemmSK EX
			{
				//fGemmSK88(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, 8, 8, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//fGemmSK88(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, 7, 7, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);

				//uGemmSK88_ohw2pow_LB4(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//uGemmSK88_ohw2pow_LB3(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//uGemmSK88_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GM);
				
				//kGemmSK88_n2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK88_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK84_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK48_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK44_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK42_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK82_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK28_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK24_ohw2pow(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
		
			//GemmSK
			{
				//usk88_v1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//usk88_v2(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//usk88_v3(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//usk88_v4(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//usk88_v5(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GM);
				//usk88_v6(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GM);
				//usk88_v7(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);

				//kGemmSK88(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK84(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK48(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK44(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK82(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK28(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK42(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK24(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);

				//kGemmSK22(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);

				//kGemmSK81(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK41(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK21(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//sGemmSK_8x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//sGemmSK_4x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//sGemmSK_2x2_1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);

				//kGemmSK14(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK12(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemmSK11(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//sGemmSK_1_4x2(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
				//sGemmSK_1_2x2(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, d_deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
		
			//GemmSK W1
			{
				//kGemmSK88W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//kGemmSK84W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//kGemmSK82W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//kGemmSK28W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//kGemmSK48W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//kGemmSK44W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//kGemmSK42W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//kGemmSK24W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);

				//kGemmSK22W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);

				//kGemmSK81W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//kGemmSK41W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//kGemmSK21W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//sGemmSK_8x2_1_W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//sGemmSK_4x2_1_W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//sGemmSK_2x2_1_W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);

				//kGemmSK14W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//kGemmSK12W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//kGemmSK11W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//sGemmSK_1_4x2_W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
				//sGemmSK_1_2x2_W1(NULL, LB, GZ, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, d_deltaW_buf, IC, OC, GN, GM);
			}

			//===================================================================================================
			if (GZ > 1) {
				//general_buf_summary(NULL, d_deltaW_buf, d_deltaW, part, sizeW);
				buf_summary(NULL, d_deltaW_buf, d_deltaW, part, sizeW);
			}
		}

		//Gemm Area=========================================
		{
			//Gemm EX
			{
				//fGemm88(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, 2, 2, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//fGemm88(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, 3, 3, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//fGemm88(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, 5, 5, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//fGemm88(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, 7, 7, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed

				//kGemm88_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//kGemm48_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, LOG2(OH), LOG2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//kGemm84_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, log2(OH), log2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm44_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, log2(OH), log2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm82_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, log2(OH), log2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm28_ohw2pow(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, log2(OH), log2(OW), d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}

			//Gemm
			{
				//kGemm88(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//A
				//kGemm84(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//kGemm48(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//kGemm44(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm82(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm28(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm42(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm24(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm22(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm41(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm14(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm21(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm12(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//kGemm11(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, OH, OW, d_deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}

			//Gemm W1
			{
				//k44W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
				//k42W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
				//k24W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
				//k22W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
				//k21W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
				//k12W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
				//k11W1(NULL, LB, 0, 0, dX, IH, IW, d_deltaY, d_deltaW, N, IC, OC, GN, GM);
			}
		}
		cudaDeviceSynchronize();
	}
	error = cudaDeviceSynchronize(); printError(error);
	
	clock_t end = clock();
	int div = end - start;
	float time = 1.0f*div / nIter;

	float size = 1.0f * GN / 1024 * GM / 1024 * GK / 1024;
	float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
	cout << "Size = " << size << ", Time = " << time << " msec, Performace = " << performance << " GFlop/s" << endl;

	float size0 = 1.0f * GN / 1024 * GM / 1024 * GK0 / 1024;
	float total0 = 2 * 1024 * size0 * 1e-9f * 1024 * 1024;
	float performance0 = total0 / (time*1e-3f);
	cout << "Size0 = " << size0 << ", Time = " << time << " msec, Performace0 = " << performance0 << " GFlop/s" << endl;

	long long Rsize = (GZ * sizeW);
	float Rspeed = (1.0f *(Rsize) / (1 << 28)) / (time*1e-3);
	cout << "Rspeed = " << Rspeed << " GB / s" << endl;

	error = cudaGetLastError(); printError(error);
	error = cudaFree(dX); printError(error);
	error = cudaFree(d_deltaW); printError(error);
	error = cudaFree(d_deltaY); printError(error);

	delete X;
	delete deltaY;
}

void test()
{
	//int IH = -1, IW = -1;
	//int OH = 32, OW = 32;
	//int FH = 4, FW = 4;
	//int sh = 2, sw = 2, ph = 1, pw = 1;
	//int N = 4;
	//int IC = 64, OC = 128;//9*4=36 


	//int IH = 16, IW = 16;
	//int OH = 8, OW = 8;
	//int FH = 3, FW = 3;
	//int sh = 2, sw = 2, ph = 1, pw = 1;//N * OH * OW
	//int N = 8;
	//int IC = 128, OC = 128;//9*4=36 128 -> 20

	//int IH = 6, IW = 6;
	//int OH = 3, OW = 3, N = 255;//256 / 8 = 32, GK_slice = 32*4*4=32*16 = 512
	//int IH = 8, IW = 8;
	//int OH = 4, OW = 4, N = 256;//256 / 8 = 32, GK_slice = 32*4*4=32*16 = 512
	//int IH = 16, IW = 16;
	//int OH = 8, OW = 8, N = 128;
	////int IH = 32, IW = 32;
	////int OH = 16, OW = 16, N = 16;
	//int FH = 4, FW = 4;
	//int sh = 2, sw = 2, ph = 1, pw = 1;//N * OH * OW
	//int IC = 32, OC = 128;//9*4=36 128 -> 20
	//int IC = 128, OC = 256;//9*4=36 128 -> 20

	//int IH = 16, IW = 16;
	//int OH = 15, OW = 15;
	//int FH = 4, FW = 4;
	//int sh = 1, sw = 1, ph = 1, pw = 1;//N * OH * OW
	//int N = 15;
	//int IC = 15, OC = 128;//9*4=36 128 -> 20

	//int IH = -1, IW = -1;
	//int OH = 32, OW = 32;//[16, 16]
	//int FH = 1, FW = 1;
	//int sh = 1, sw = 1, ph = 0, pw = 0;
	//int IC = 128, OC = 128, N = 8;

	//int IC = 16, OC = 128, N = 32;
	//int IC = 8, OC = 128, N = 64;
	//int IC = 4, OC = 128, N = 128;

	//int IC = 128, OC = 16, N = 32;
	//int IC = 128, OC = 8, N = 64;
	//int IC = 128, OC = 4, N = 128;

	//testCorrect<3>(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, OH - 1, OW - 1, FH, FW, N - 4, IC, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, OH + 1, OW + 1, FH, FW, N + 4, IC, OC, sh, sw, ph, pw);
	//testSpeed<3>(500, IH, IW, OH, OW, FH, FW, N*2, IC * 2, OC * 2, sh, sw, ph, pw);

	//int IH = 4, IW = 4, OH = 2, OW = 2, N = 512;
	int IH = 8, IW = 8, OH = 4, OW = 4, N = 128;
	//int IH = 16, IW = 16, OH = 8, OW = 8, N = 32;
	//int IH = 32, IW = 32, OH = 16, OW = 16, N = 8;

	//int IH = 6, IW = 6, OH = 3, OW = 3, N = 256;
	//int IH = 30, IW = 30, OH = 15, OW = 15, N = 9;
	//int IH = 14, IW = 14, OH = 7, OW = 7, N = 7;

	int FH = 3, FW = 3, ph = 1, pw = 1;//N * OH * OW
	int sh = 2, sw = 2;
	int IC = 128, OC = 128;//IC = 64 \ 128

	//int IC = 16, OC = 128;
	//int IC = 8, OC = 256;
	//int IC = 4, OC = 256; N *= 2;

	//int IC = 128, OC = 16;
	//int IC = 256, OC = 8;
	//int IC = 256, OC = 4; N *= 2;
	
	testCorrect<4>(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, OH, OW, FH, FW, N + 4, IC, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, OH, OW, FH, FW, N - 4, IC, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, OH - 1, OW - 1, FH, FW, N - 3, IC, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, OH + 1, OW + 1, FH, FW, N + 3, IC, OC, sh, sw, ph, pw);
	//testSpeed<3>(500, IH, IW, OH, OW, FH, FW, N, IC, OC*4, sh, sw, ph, pw);
	testSpeed<4>(500, IH, IW, OH, OW, FH, FW, N*2, IC*4, OC, sh, sw, ph, pw);


	//=================Cross Area====================================
	//int FH = 3, FW = 3, sh = 2, sw = 2, ph = 1, pw = 1;
	
	//int IH = 4, IW = 4;
	//int OH = 2, OW = 2, N = 128;
	//int IC = 256, OC = 512;
	//OC = 32, N = 512;

	//testCorrect<3>(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	//testSpeed<3>(500, IH, IW, OH, OW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);

	//==============Big Graph========================================
	//int IH = 16, IW = 16, OH = 8, OW = 8, N = 64;
	//int IH = 32, IW = 32, OH = 16, OW = 16, N = 16;
	//int IH = 64, IW = 64, OH = 32, OW = 32, N =  4;

	//int FH = 3, FW = 3, ph = 1, pw = 1;
	//int sh = 2, sw = 2;
	//int IC = 128, OC = 128;

	//testCorrect<4>(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, OH, OW, FH, FW, 8, IC, OC, sh, sw, ph, pw);
	//testSpeed<4>(500, IH, IW, OH, OW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
}

main()
{
	test();
}

#endif