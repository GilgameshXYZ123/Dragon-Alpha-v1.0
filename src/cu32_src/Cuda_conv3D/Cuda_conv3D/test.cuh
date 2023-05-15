#pragma once

#define TEST_H
#ifndef TEST_H
#define TEST_H

#ifndef UTIL
#define UTIL

float* newRandomFloatVec(int length)//0-256
{
	float *p = new float[length];
	for (int i = 0; i < length; i++) p[i] = (float)(rand() % 1000) / 1000 + 1;
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
		if (dif < 1e-2) sum++;
		else {
			//if (a[i] < b[i]) cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
			//cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
		}
		
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


#ifndef CONV_3D_CPU
#define CONV_3D_CPU

void CONV_3D_NAIVE(
	float *X, int IH, int IW,
	float *W, int FH, int FW,
	float *Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw,
	int ph, int pw)
{
	//Y[N, OH, OW, OC] = > C[GN, GM]
	for (int oc = 0; oc < OC; oc++)//OUT channel: use kernel[oc]
	for (int n = 0; n < N; n++)//for each sample
	{
		int ic_s, ih_s, iw_s, oh, ow;
		for (ih_s = -ph, oh = 0; ih_s <= (IH + ph - FH); ih_s += sh, oh++)//oh < OH
		for (iw_s = -pw, ow = 0; iw_s <= (IW + pw - FW); iw_s += sw, ow++)//ow < OW
		{
			float v = 0;
			for (int fh = 0; fh < FH; fh++)
			for (int fw = 0; fw < FW; fw++)
			for (int ic = 0; ic < IC; ic++)
			{
				int ih = ih_s + fh, iw = iw_s + fw;
				if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
						v += get4d(X, n, ih, iw, ic, IH, IW, IC)*
						get4d(W, oc, fh, fw, ic, FH, FW, IC);
			}
			get4d(Y, n, oh, ow, oc, OH, OW, OC) = v;
		}
	}
}


void CONV_3D_img2col(
	float *X, int IH, int IW, //X[N , IH, IW, IC] => A[GN, GK]
	float *W, int FH, int FW, //W[OC, KH, KW, IC] => B[GK, GM]
	float *Y, int OH, int OW, //Y[N , OH, OW, OC] => C[GN, GM]
	int N, int IC, int OC,
	int sh, int sw,
	int ph, int pw)
{
	int GN = OC;
	int GM = N * OH * OW;
	int GK = FH * FW * IC;

	for (int i = 0; i < GN; i++)
	{
		int oc = i;
		for (int j = 0; j < GM; j++)
		{
			int n = j / (OH*OW);
			int j_res = j % (OH*OW);
			int oh = j_res / OW;
			int ow = j_res % OW;

			float v = 0;
			for (int k = 0; k < GK; k++)
			{
				int fh = k / (FW*IC);
				int k_res = k % (FH*IC);
				int fw = k_res / IC;
				int ic = k_res % IC;
				int ih = oh * sh - ph + fh;
				int iw = ow * sw - pw + fw;

				if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
				v += get4d(X, n, ih, iw, ic, IH, IW, IC)*
					get4d(W, oc, fh, fw, ic, FH, FW, IC);
			}
			get4d(Y, n, oh, ow, oc, OH, OW, OC) = v;
		}
	}
}
#endif


#ifndef PROOF1
#define PROOF1

//int OH = (IH + 2 * ph - FH) / sh + 1;
//int OW = (IW + 2 * pw - FW) / sw + 1;
void proof1(
	int IH, int IW,
	int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw,
	int ph, int pw)
{
	float *X = newRandomFloatVec(N*IH*IW*IC);
	float *W = newRandomFloatVec(OC*FH*FW*IC);

	int OH = (IH + 2 * ph - FH) / sh + 1;
	int OW = (IW + 2 * pw - FW) / sw + 1;
	printf("(OH, OW) = (%d, %d)\n", OH, OW);

	int sizeY = N * OC*OH*OW;
	float *Y1 = new float[sizeY];
	memset(Y1, 0, sizeof(float)*sizeY);
	float *Y2 = new float[sizeY];
	memset(Y2, 0, sizeof(float)*sizeY);

	//use img2col---------------------------
	CONV_3D_img2col(X, IH, IW,
		W, FH, FW,
		Y1, OH, OW,
		N, IC, OC, sh, sw, ph, pw);
	cout << "use img2col method:"; println(Y1, 10);

	//use naive method----------------------
	CONV_3D_NAIVE(X, IH, IW,
		W, FH, FW,
		Y2, OH, OW,
		N, IC, OC, sh, sw, ph, pw);
	cout << "use naive method:  "; println(Y2, 10);

	float sp = samePercent(Y1, Y2, sizeY);
	cout << "SamePercent: " << sp << endl;
}

//(correct)
void proof()
{
	//int OH = (IH + 2 * ph - FH) / sh + 1;
	//int OW = (IW + 2 * pw - FW) / sw + 1;
	int IH = 64, IW = 64;
	int FH = 8, FW = 8;
	int N = 4;
	int IC = 3, OC = 2;
	int sh = 4, sw = 4, ph = 4, pw = 4;
	proof1(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
}

#endif


template<int LB>
void testCorrect(
	int IH, int IW,
	int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int OH = (IH + 2 * ph - FH) / sh + 1;
	int OW = (IW + 2 * pw - FW) / sw + 1;
	printf("Test Correct:\n");
	printf("\t(IH, IW) = (%d, %d)\n", IH, IW);
	printf("\t(FH, FW) = (%d, %d)\n", FH, FW);
	printf("\t(N, IC, OC) = (%d, %d, %d)\n", N, IC, OC);
	printf("\t(OH, OW) = (%d, %d)\n", OH, OW);
	printf("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
	int GN = OC;
	int GM = N * OH*OW;
	int GK = FH * FW*IC;
	printf("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);

	int sizeX = N * IH*IW*IC;
	int sizeY = N * OH*OW*OC;
	int sizeW = OC * FH*FW*IC;

	float *X = newRandomFloatVec(sizeX);
	float *W = newRandomFloatVec(sizeW);
	float *Y1 = new float[sizeY]; memset(Y1, 0, sizeof(float)*sizeY);
	float *Y2 = new float[sizeY]; memset(Y2, 0, sizeof(float)*sizeY);
	float *Y3 = new float[sizeY]; memset(Y3, 0, sizeof(float)*sizeY);

	//CPU----------------------------
	CONV_3D_NAIVE(X, IH, IW, W, FH, FW, Y1, OH, OW, N, IC, OC, sh, sw, ph, pw); printf("CPU1: "); println(Y1, 10);
	//CONV_3D_img2col(X, IH, IW, W, FH, FW, Y2, OH, OW, N, IC, OC, sh, sw, ph, pw);printf("CPU2: "); println(Y2, 10);

	//GPU-----------------------------
	float *dX = newDevFloatVec(X, sizeX);
	float *dW = newDevFloatVec(W, sizeW);
	float *dY = newDevFloatVec(sizeY);
	jlong *streams = new jlong[8];
	for (int i = 0; i < 8; i++) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		streams[i] = (intptr_t)stream;
	}

	cudaError_t error;
	
	float* dCW = newDevFloatVec(sizeW);
	cudaTextureObject_t texX = floatTexture(dX, sizeX);

	float Q = PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW, sh, sw);
	cout << "Q = " << Q << endl;

	//----------------------------------------------------------
	//Winograd
	{
		//__kernel_remode(NULL, dW, dCW, FH, FW, OC, IC);
		//wingrad_b1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_b2(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_b3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_b4(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_b5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);

		//wingrad_c1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_c2(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_c3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_c4(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_c5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);

		//wingrad_d1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_d2(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_d3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_d4(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_d5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_d6(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//wingrad_d7(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
	}

	//GemmR=====================================================
	{
		__kernel_remode(NULL, dW, dCW, FH, FW, OC, IC);

		//pxkernel14(NULL, LB, 0, 0, dX, IH, IW, dCW, 3, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//qxkernel1(NULL, LB, 0, 0, dX, IH, IW, dCW, 3, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//qxkernel2(NULL, LB, 0, 0, dX, IH, IW, dCW, 3, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		//qxkernel3(NULL, LB, 0, 0, dX, IH, IW, dCW, 3, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		qxkernel4(NULL, LB, 0, 0, texX, IH, IW, dCW, 3, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		
		//GemmRA kernel
		{
			//conv3dGemm_k88RA(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88RA_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88RA_W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_W3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88RA_W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88RA_W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_W5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
		}

		//GemmR V2
		{
			//conv3dGemmV2_k88RW3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed
			//conv3dGemmV2_k88RW4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed
			//conv3dGemmV2_k88RW5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed

			//conv3dGemmV2_k88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			//conv3dGemmV2_k84R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			//conv3dGemmV2_k48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			//conv3dGemmV2_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			//conv3dGemmV2_k42R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			//conv3dGemmV2_k24R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
		}
	
		//GemmR uernel V2
		{
			//conv3dGemmV2_u88RW3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			//conv3dGemmV2_u88RW4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			//conv3dGemmV2_u88RW5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			//conv3dGemmV2_u88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
		}

		//GemmR uernel
		{
			//conv3dGemm_u88R4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

			//conv3dGemm_u88R4(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);

			//conv3dGemm_u88RW3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88RW3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R4W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R4W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

			//conv3dGemm_u88RW5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88RW5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R4W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R4W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);	

			//conv3dPure_u48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_u44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}

		//FH = FW = 5
		{
			//conv3dGemm_k88RW5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4W5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88RW5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
		}
	
		//FH = FW = 3
		{
			//conv3dGemm_k88R4W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4W3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88R4W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		}

		//GemmR 8*8 kernel
		{
			//conv3dGemm_k88R4_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
		
			//conv3dGemm_k88R4_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			
			//conv3dGemm_k88R4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88R4(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		}
		
		//GemmR pure
		{
			//conv3dPure_k48R_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k28R_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);

			//conv3dPure_k84R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k82R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k42R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}

		//conv3dGemm_k84R_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

		//GemmRW1
		{
			//conv3d_u88R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);

			//conv3d_k88R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			//conv3d_k84R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			//conv3d_k48R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			//conv3d_k44R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			//conv3d_k82R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			//conv3d_k42R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
		}
	}

	//Gemm======================================================
	{
		//Gemm_np
		{
			//conv3dGemm_k88x4_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k88x4_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k88x4_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed

			//conv3dGemm_k88_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k88_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k88_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k84_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k84_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k84_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k48_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k48_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k44_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k44_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k22_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k41_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k14_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
		}

		//Gemm sV2
		{
			//conv3dGemmV2_k88(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
			//conv3dGemmV2_k88W3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			//conv3dGemmV2_k88W4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			//conv3dGemmV2_k88W5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
		}

		//FH = FW = 5
		{
			//conv3dGemm_k88W5(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5x4(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88W5_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
		}
		
		//FH = FW = 3
		{
			//conv3dGemm_k88W3x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W3x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W3_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88W3x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W3x4(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			///conv3dGemm_k88W3(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		}
		
		//Gemm 8*8 kernel
		{
			//conv3dGemm_k88x4_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//====================================================================================
			//conv3dGemm_k88x4_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88x4(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		}
		
		//Gemm pure
		{
			//conv3dPure_k48_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);

			//conv3dPure_k84(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k48(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k44(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k82(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k28(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k42(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k24(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
	
		//conv3dGemm_k44(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k42(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k24(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

		//conv3dGemm_k22(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

		//conv3dGemm_k41(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k21(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

		//conv3dGemm_k14(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k12(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k11(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_s1_4x2(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_s1_2x2(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

		//Gemm W1
		{
			//int index = 0; Conv3D(streams, index, 8, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw);
			//conv3d_k88_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k84_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k48_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k44_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k82_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
			//conv3d_k28_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
			//conv3d_k42_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k24_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

			//conv3d_k22_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k21_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

			//conv3d_k14_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k12_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_s1_4x2_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

			//conv3d_k11_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

			//int index = 0; Conv3D_W1(streams, index, 8, dX, IH, IW, dW, dY, N, IC, OC);
		}

	}
	//--------------------------------
	error = cudaGetLastError(); printError(error);
	error = cudaMemcpy(Y3, dY, sizeof(float)*sizeY, cudaMemcpyDeviceToHost); printError(error);
	printf("GPU : "); println(Y3, 10);

	////compare--------------------------
	//float sp0 = samePercent(Y1, Y2, sizeY); cout << "sp0: " << sp0 << endl;
	float sp1 = samePercent(Y1, Y3, sizeY); cout << "sp1: " << sp1 << endl;
	//float sp2 = samePercent(Y2, Y3, sizeY); cout << "sp2: " << sp2 << endl;
	//float zp1 = zeroPercent(Y1, sizeY); cout << "zpY1:" << zp1 << endl;
	float zp3 = zeroPercent(Y3, sizeY); cout << "zpY3:" << zp3 << endl;

	//clear mem------------------------
	error = cudaFree(dX); printError(error);
	error = cudaFree(dW); printError(error);
	error = cudaFree(dY); printError(error);

	if (sp1 < 0.999f) exit(-2);

	delete X;
	delete W;
	delete Y1; delete Y2; delete Y3;
	for (int i = 0; i < 8; i++) {
		cudaStream_t stream = (cudaStream_t)(intptr_t)streams[i];
		cudaStreamDestroy(stream);
	}
	delete[] streams;
}

template<int LB>
void testSpeed(
	int nIter,
	int IH, int IW,
	int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int OH = (IH + 2 * ph - FH) / sh + 1;
	int OW = (IW + 2 * pw - FW) / sw + 1;
	printf("Test Correct:\n");
	printf("\t(IH, IW) = (%d, %d)\n", IH, IW);
	printf("\t(FH, FW) = (%d, %d)\n", FH, FW);
	printf("\t(n, IC, OC) = (%d, %d, %d)\n", N, IC, OC);
	printf("\t(OH, OW) = (%d, %d)\n", OH, OW);
	printf("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
	int GN = OC;
	int GM = N * OH*OW;
	int GK = IC * FH*FW;
	printf("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);

	size_t sizeX = N * IC*IH*IW;
	size_t sizeW = OC * IC*FH*FW;
	size_t sizeY = N * OC*OH*OW;

	float *X = newRandomFloatVec(sizeX);
	float *W = newRandomFloatVec(sizeW);

	float *dX = newDevFloatVec(X, sizeX);
	float *dW = newDevFloatVec(W, sizeW);
	float *dY = newDevFloatVec(sizeY);
	cudaError_t error;

	float *dCW = newDevFloatVec(sizeW);
	
	cudaEvent_t start, end;
	cudaEventCreate(&start, cudaEventDefault);
	cudaEventCreate(&end, cudaEventDefault);

	//clock_t start = clock();
	cudaEventRecord(start, NULL);
	cudaTextureObject_t texX = floatTexture(dX, sizeX);

	for (int i = 0; i < nIter; i++) 
	{
		//Winograd
		{
			//__kernel_remode(NULL, dW, dCW, FH, FW, OC, IC);
			//wingrad_b1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_b2(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_b3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_b4(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_b5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);

			//wingrad_c1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_c2(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_c3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_c4(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_c5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);

			//wingrad_d1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_d2(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_d3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_d4(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_d5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_d6(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//wingrad_d7(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		}

		//GemmR==================================================
		{
			//cudaTextureObject_t texX = floatTexture(dX, sizeX);

			//__kernel_remode(NULL, dW, dCW, FH, FW, OC, IC);

			//pxkernel14(NULL, LB, 0, 0, dX, IH, IW, dCW, 3, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//qxkernel1(NULL, LB, 0, 0, dX, IH, IW, dCW, 3, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//qxkernel2(NULL, LB, 0, 0, dX, IH, IW, dCW, 3, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//qxkernel3(NULL, LB, 0, 0, dX, IH, IW, dCW, 3, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			qxkernel4(NULL, LB, 0, 0, texX, IH, IW, dCW, 3, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);

			//GemmRA
			{
				//conv3dGemm_k88RA(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RA_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RA_W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_W3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RA_W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RA_W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_W5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RA_W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			}

			//GemmR V2
			{
				//conv3dGemmV2_k88RW3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed
				//conv3dGemmV2_k88RW4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed
				//conv3dGemmV2_k88RW5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed

				//conv3dGemmV2_k88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
				//conv3dGemmV2_k84R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
				//conv3dGemmV2_k48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
				//conv3dGemmV2_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
				//conv3dGemmV2_k42R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
				//conv3dGemmV2_k24R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			}
		
			//GemmR uernel V2
			{
				//conv3dGemmV2_u88RW3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				//conv3dGemmV2_u88RW4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				//conv3dGemmV2_u88RW5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				//conv3dGemmV2_u88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			}

			//GemmR uernel
			{
				//conv3dGemm_u88R4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_u88R_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_u88R4(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);

				//conv3dGemm_u88RW3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88RW3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R4W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R4W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

				//conv3dGemm_u88RW5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88RW5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R4W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R4W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

				//conv3dPure_u48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_u44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}

			//FH = FW = 5
			{
				//conv3dGemm_k88RW5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RW5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4W5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RW5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RW5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			}

			//FH = FW = 3
			{
				//conv3dGemm_k88R4W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4W3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RW3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RW3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88R4W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RW3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RW3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			}

			//GemmR 8*8 kernel
			{
				//conv3dGemm_k88R4_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			
				//conv3dGemm_k88R4_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88R4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88R4(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			}
			
			//GemmR pure
			{
				//conv3dPure_k48R_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k28R_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);

				//conv3dPure_k84R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k82R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k42R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			
			//conv3dGemm_k84R_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//cudaDestroyTextureObject(texX);

			//GemmRW1
			{
				//conv3d_u88R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);

				//conv3d_k88R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
				//conv3d_k84R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
				//conv3d_k48R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
				//conv3d_k44R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
				//conv3d_k82R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
				//conv3d_k42R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			}
		}

		//Gemm==================================================
		{  
			//Gemmnp
			{
				//conv3dGemm_k88x4_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k88x4_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k88x4_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed

				//conv3dGemm_k88_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k88_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k88_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k84_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k84_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k84_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k48_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k48_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k44_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k44_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k22_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k41_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k14_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			}

			//GemmR V2
			{
				//conv3dGemmV2_k88(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
				//conv3dGemmV2_k88W3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				//conv3dGemmV2_k88W4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				//conv3dGemmV2_k88W5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			}
			
			//FH = FW = 5
			{
				//conv3dGemm_k88W5(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5x4(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88W5_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			}
		
			//FH = FW = 3
			{
				//conv3dGemm_k88W3x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
	
				//conv3dGemm_k88W3x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3x4(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			}
			
			//Gemm 8*8 kernel
			{
				//conv3dGemm_k88x4_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

				//============================================================
				//conv3dGemm_k88x4_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88x4(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			}

			//Gemm pure
			{
				//conv3dPure_k48_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);

				//conv3dPure_k84(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k48(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k44(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k82(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k28(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k42(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k24(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			
			//conv3dGemm_k44(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k22(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k41(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k21(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k14(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k12(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k11(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_s1_4x2(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_s1_2x2(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//GemmW1
			{
				//conv3d_k88_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
				//conv3d_k84_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
				//conv3d_k48_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
				//conv3d_k44_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
				//conv3d_k82_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
				//conv3d_k28_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
				//conv3d_k42_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY,IC, OC, GN, GM);//passed
				//conv3d_k24_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed

				//conv3d_k22_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
				//conv3d_k21_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

				//conv3d_k14_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
				//conv3d_k12_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
				//conv3d_s1_4x2_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

				//conv3d_k11_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

				//Conv3D_W1(pool, dX, IH, IW, dW, dY, IC, OC);
			}
		}
		//----------------------------
		//cudaDeviceSynchronize();
	}
	cudaDestroyTextureObject(texX);
	cudaEventRecord(end, NULL);
	
	error = cudaDeviceSynchronize(); printError(error);
	error = cudaGetLastError(); printError(error);
	//clock_t end = clock();
	
	//int div = end - start;
	float div; cudaEventElapsedTime(&div, start, end);
		
	float time = 1.0f*div / nIter;
	float size = 1.0f * GN / 1024 * GM / 1024 * GK / 1024;
	cout << "Size = " << size;
	float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
	cout << ", Time = " << time << " msec, Performace = " << performance << " GFlop/s";


	error = cudaFree(dX); printError(error);
	error = cudaFree(dY); printError(error);
	error = cudaFree(dW); printError(error);

	delete X;
}

void test()
{
	//proof();

	/*int GN = OC;
	int GM = N * OH*OW;
	int GK = IC * FH*FW;*/

	
	//for (int oc = 4; oc <= 192; oc+=4) testCorrect<3>(IH, IW, FH, FW, N, 8, oc, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	//testSpeed<3>(500, IH, IW, FH, FW, N * 4, IC * 2, OC, sh, sw, ph, pw);
	//for (int oc = 4; oc <= 256; oc += 4)
	//	testCorrect<2>(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);

	/*testCorrect<4>(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	testCorrect<3>(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	testCorrect<2>(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);*/
	//testSpeed<4>(IH, IW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);


	//int IH = 128, IW = 128;
	//int FH = 2, FW = 1;//(2. 4), (4, 2)可以//16/2=8
	//int sh = 1, sw = 1, ph = 1, pw = 1;
	//int N = 4, IC = 4, OC = 128;//9*4=36 

	//int IH = 64, IW = 64;
	//int FH = 8, FW = 8;//(2. 4), (4, 2)可以
	//int sh = 2, sw = 2, ph = 1, pw = 1;
	//int N = 2, IC = 16, OC = 128;//9*4=36 

	//GM = N * OH * OW
	//int IH = 127, IW = 127;
	//int FH = 3, FW = 3;//(2. 4), (4, 2)可以
	//int N = 4;
	////int IC = 4, OC = 128;//9*4=36 
	//int sh = 2, sw = 2, ph = 1, pw = 1;
	//for (int oc = 4; oc <= 192; oc+=4) testCorrect<3>(IH, IW, FH, FW, N, 4, oc, sh, sw, ph, pw);

	//int IH = 128, IW = 128;
	//int FH = 8, FW = 8;//(2. 4), (4, 2)可以
	//int N = 4;
	//int IC = 4, OC = 128;//128+64+32+16+8+4
	//int sh = 2, sw = 2, ph = 1, pw = 1;

	//int IH = 33, IW = 33;
	//int FH = 4, FW = 4;
	//int sh = 1, sw = 1, ph = 1, pw = 1;
	//int N = 8;
	//int IC = 32, OC = 128;//9*4=36 

	//16*4=64

	//int IH = 34, IW = 34;
	//int FH = 5, FW = 5;
	//int sh = 1, sw = 1, ph = 1, pw = 1;
	//int N = 4;
	//int IC = 4, OC = 128;//9*4=36 

	//int IH = 64, IW = 64;
	//int FH = 1, FW = 1;
	//int N = 4;
	//int IC = 64, OC = 128;//128+64+32+16+8+4
	//int sh = 1, sw = 1, ph = 0, pw = 0;

	//int IH = 35, IW = 35;
	//int FH = 4, FW = 4;
	//int sh = 1, sw = 1, ph = 0, pw = 0;
	//int N = 4;
	//int IC = 32, OC = 128;//9*4=36 

	//int IH = 32, IW = 32;
	//int FH = 4, FW = 4, ph = 1, pw = 1;
	//int FH = 3, FW = 3, ph = 1, pw = 1;
	//int sh = 2, sw = 2;
	//int N = 32, IC = 64;
	//int OC = 128;//9*4=36 

	//int OC = 64; IC *= 2;
	//int OC = 32; IC *= 2; N *= 2;
	//int OC = 16; N = 64;
	//int OC = 8; N = 64; IC *= 2;
	//int OC = 4; N = 128; IC *= 2;

	//testCorrect<3>(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, FH, FW, N, IC + 8, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, FH - 1, FW - 1, N, IC + 1, OC, sh, sw, ph, pw);
	//testSpeed<3>(500, IH, IW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);

	//int IH = 32, IW = 32;
	//int FH = 3, FW = 3;
	//int sh = 2, sw = 2, ph = 1, pw = 1;
	//int N = 64;
	//int IC = 64, OC = 128;//9*4=36 

	//OC /= 2; N *= 2;
	//OC /= 4; N *= 4;
	//OC /= 8; N *= 8;

	//int N = 256, IC = 128;//9*4=36 64 / 81

	//int IH = 32, IW = 32, OC = 128; N /= 2; IC /= 8;
	//int IH = 16, IW = 16, OC = 128; N /= 2; IC /= 2;
	//int IH = 8, IW = 8, OC = 128;
	//int IH = 4, IW = 4, OC = 512;
	//int IH = 2, IW = 2, OC = 512;
	//int FH = 3, FW = 3, ph = 1, pw = 1;
	//int FH = 4, FW = 4, ph = 1, pw = 1;
	//int FH = 5, FW = 5, ph = 2, pw = 2; N /= 2;
	//int sh = 2, sw = 2; 
	//V2: testSpeed: N*2, IC*2

	//int IH = 16, IW = 16;
	//int IH = 32, IW = 32;
	//int FH = 5, FW = 5;
	//int sh = 2, sw = 2, ph = 2, pw = 2;
	//int N = 64;
	//int IC = 32, OC = 128;//9*4=36 

	/*int IH = 16, IW = 16;
	int FH = 1, FW = 1, ph = 0, pw = 0;
	int sh = 1, sw = 1;
	int IC = 128, OC = 128, N = 128;*/
	//int IC = 128, OC = 16, N = 256;
	//int IC = 256, OC = 4, N = 512;

	//testCorrect<4>(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	//testCorrect<4>(IH, IW, FH, FW, N, IC - 16, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, FH, FW, N, IC + 8, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, FH, FW, N, IC - 1, OC, sh, sw, ph, pw);
	//testSpeed<3>(500, IH, IW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
	//system("pause");
	
	//compress_area==========================
	//int IH = 32, IW = 32;
	//int FH = 4, FW = 4;
	//int sh = 2, sw = 2, ph = 1, pw = 1;
	//int N = 128;
	//int IC = 64, OC = 8;//9*4=36 

	//testCorrect<4>(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	//testSpeed<4>(20, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);


	//Winograd area==========================================================
	//bool x = 31;
	//cout << sizeof(x) << endl;
	//char c = 63;
	//cout << (int)c << endl;
	//63: 32 + 16 + 8 + 4 + 2 + 1


	int IH = 32, IW = 32;
	int FH = 3, FW = 3;
	int sh = 1, sw = 1, ph = 1, pw = 1;
	int N = 16;
	int IC = 32, OC = 128;//9*4=36 

	testCorrect<4>(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, FH, FW, N, IC + 8, OC, sh, sw, ph, pw);
	//testCorrect<3>(IH, IW, FH - 1, FW - 1, N, IC + 1, OC, sh, sw, ph, pw);
	testSpeed<4>(500, IH, IW, FH, FW, N, IC*2, OC, sh, sw, ph, pw);
	//testSpeed<4>(500, IH*2, IW*2, FH, FW, N, IC * 2, OC, sh, sw, ph, pw);
}

main() { test(); }

#endif

