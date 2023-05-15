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
float* newDevFloatVec(int length)
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

char* newDevCharVec(int length)
{
	char *dp = NULL;
	size_t size = sizeof(char)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemset(dp, 0, size); printError(error);
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
		if (dif < 1e-4) sum++;
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


#ifndef DECONV_CPU
#define DECONV_CPU

void deconv3D_deltaX_img2col(
	float* deltaY, int OH, int OW,
	float* W, int FH, int FW,
	float* deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int GN = IC;
	int GM = N * IH * IW;
	int GK = OC * FH * FW;

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

			float v = 0;
			for (int k = 0; k < GK; k++)
			{
				int oc = k / (FH*FW);
				int k_res = k % (FH*FW);
				int fh = k_res / FW, fw = k_res % FW;

				int oh = ih - oph + fh;
				int ow = iw - opw + fw;

				if (oh < 0 || ow < 0 || oh >= OH_p || ow >= OW_p) continue;
				if (oh%sh != 0 || ow % sw != 0) continue;

				v += get4d(deltaY, n, (oh / sh), (ow / sw), oc, OH, OW, OC)*
					 get4d(W, oc, (FH - 1 - fh), (FW - 1 - fw), ic, FH, FW, IC);
			}

			get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = v;
		}
	}
}
#endif


template<int LB>
void testCorrect(int IH, int IW,
	int OH, int OW,
	int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	if (IH == -1) IH = (OH - 1)*sh + FH - 2 * ph;
	if (IW == -1) IW = (OW - 1)*sw + FW - 2 * pw;

	int GN = IC;
	int GM = N * IH*IW;
	int GK = OC * FH*FW;

	printf("Test Correct:\n");
	printf("\t(OH, OW) = (%d, %d)\n", OH, OW);
	printf("\t(FH, FW) = (%d, %d)\n", FH, FW);
	printf("\t(n, IC, OC) = (%d, %d, %d)\n", N, IC, OC);
	printf("\t(IH, IW) = (%d, %d)\n", IH, IW);
	printf("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
	printf("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);

	float Qs1 = s1_PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW);
	cout << "Qs1 = " << Qs1 << endl;

	float QIms2; Ims2_PADDING_SCALE_UP(QIms2, IH >> 1, IW >> 1, OH, OW, FH, FW);
	cout << "QIms2 = " << QIms2 << endl;

	int sizeX = N * IC*IH*IW;
	int sizeW = OC * IC*FH*FW;
	int sizeY = N * OC*OH*OW; cout << "sizeY = " << sizeY << endl;

	float *deltaY = newRandomFloatVec(sizeY);
	float *W = newRandomFloatVec(sizeW);

	float *deltaX1 = new float[sizeX];
	float *deltaX2 = new float[sizeX];

	//CPU--------------------------------------
	deconv3D_deltaX_img2col(deltaY, OH, OW, W, FH, FW, deltaX1, IH, IW, N, IC, OC, sh, sw, ph, pw);
	cout << "CPU: "; println(deltaX1, 10);
	//float zp0 = zeroPercent(deltaX1, 10); cout << "zp0: " << zp0 << endl;

	//GPU--------------------------------------
	float *d_deltaY = newDevFloatVec(deltaY, sizeY);
	float *dW = newDevFloatVec(W, sizeW);
	float *d_deltaX = newDevFloatVec(sizeX);

	cudaError_t error;

	jlong streams[8];
	for (int i = 0; i < 8; i++) {
		cudaStream_t stream; cudaStreamCreate(&stream);
		streams[i] = (intptr_t)stream;
	}

	//-----------------------------------
	cudaTextureObject_t texDy = floatTexture(d_deltaY, sizeY);

	//KernelSplit
	{
		int LOC = LOG2(OC);

		//ImsR
		{
		/*	Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
			cout << "GN, GM = " << GN << ", " << GM << endl;
			cout << "CFH, CFW = " << CFH << ", " << CFW << endl;

			int sizeWks = sh * sw * OC * CFH * CFW * IC;
			float* dCW = newDevFloatVec(sizeWks);

			__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);
			cudaStreamSynchronize(NULL);

			float* CW = new float[sizeWks];
			error = cudaMemcpy(CW, dCW, sizeof(float)*sizeWks, cudaMemcpyDeviceToHost); printError(error);
			float zp_ks = zeroPercent(CW, sizeWks);
			cout << "zp_ks = " << zp_ks << endl;*/

			//==============================================
			//ksIms_88R8_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_88R4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_88R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S

			//ksIms_88R8_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_88R4_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_88R_tex(NULL, LB, 0, 0,texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S

			//ksIms_84R_tex(NULL, LB, 0, 0,texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_84R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S

			//ksIms_48R_tex(NULL, LB, 0, 0,texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_48R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S

			//ksIms_44R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_44R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S

			//===============================================
			//ksIms_88R8_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_88R4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S

			//ksIms_88R8(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_88R4(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S

			//ksIms_84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S

			//ksIms_48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S

			//ksIms_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S

			//ksIms_42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_22R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_21R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_12R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
			//ksIms_11R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}

		//ImsR2
		{
			Ims2_init(N, IH, IW, FH, FW, OC, IC);
			cout << "GN, GM = " << GN << ", " << GM << endl;
			cout << "CFH, CFW = " << CFH << ", " << CFW << endl;

			int sizeWks = sh * sw * OC * CFH * CFW * IC;
			float* dCW = newDevFloatVec(sizeWks);

			__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);
			cudaStreamSynchronize(NULL);

			float* CW = new float[sizeWks];
			error = cudaMemcpy(CW, dCW, sizeof(float)*sizeWks, cudaMemcpyDeviceToHost); printError(error);
			float zp_ks = zeroPercent(CW, sizeWks);
			cout << "zp_ks = " << zp_ks << endl;
			
			//ksIms2V2
			{
				//ksV2_Ims2_u88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
				//ksV2_Ims2_u88R_CFW_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
				//ksV2_Ims2_u88R_W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
				
				//===================================================
				//ksV2_Ims2_88R_CFW_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
				//ksV2_Ims2_88R_W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);

				//ksV2_Ims2_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
				//ksV2_Ims2_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
				//ksV2_Ims2_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
				//ksV2_Ims2_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
			}
			
			//uernel
			{
				//ksIms2_u88R8_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_u88R4_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				ksIms2_u88R_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
		
			//CFW_OC_2pow
			{
				//ksIms2_88R8_oc_CFW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
				//ksIms2_88R4_oc_CFW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

				//ksIms2_88R8_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R4_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}

			//OC_2POW
			{
				//ksIms2_88R8_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC,  LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);

				//ksIms2_88R8_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
			
			//Common 8*8
			{
				//ksIms2_88R8_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R4_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);

				//ksIms2_88R8(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R4(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			}

			//CW 2pow
			{
				//ksIms2_88R8_CW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R4_CW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);

				//ksIms2_88R8_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R4_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			}

			//Kernel A
			{
				//ksIms2A_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
				//ksIms2A_88R_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
			}

			//ksIms2_84R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			//ksIms2_84R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
			//ksIms2_48R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			//ksIms2_48R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
			//ksIms2_44R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			//ksIms2_44R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

			//====================================================================
			//ksIms2_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			//ksIms2_84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

			//ksIms2_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			//ksIms2_48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

			//ksIms2_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			//ksIms2_44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

			//ksIms2_42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			//ksIms2_24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		}

		//Kernel Split
		{
			//KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
			//cout << "GN, GM = " << GN << ", " << GM << endl;
			//cout << "CFH, CFW = " << CFH << ", " << CFW << endl;

			//int sizeWks = sh * sw * OC * CFH * CFW * IC;
			//float* dCW = newDevFloatVec(sizeWks);

			//__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);
			//cudaStreamSynchronize(NULL);

			//float* CW = new float[sizeWks];
			//error = cudaMemcpy(CW, dCW, sizeof(float)*sizeWks, cudaMemcpyDeviceToHost); printError(error);
			//float zp_ks = zeroPercent(CW, sizeWks);
			//cout << "zp_ks = " << zp_ks << endl;

			//ks88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			//ks84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			//ks48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			//ks44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);

			//ks88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks22R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks21R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks12R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks11R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}
	}

	//Cross Add
	{
		//int index = 0; __dconv3D_deltaX_CrossAdd(streams, index, 8, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw);
		//crossAdd_k16_2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
		//crossAdd_k82(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
		//crossAdd_k42(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
		//crossAdd_k22(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
		//crossAdd_k11(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
	}
	
	//kernel s1
	{
		//kernel V2
		{
			//uV2_88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
			//uV2_88s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);

			//uV2_88s1W3P1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
			//uV2_88s1W3P1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);

			//uV2_88s1W5P2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
			//uV2_88s1W5P2_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);

			//===============================================================================================
			//kV2_88s1W5P2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
			//kV2_88s1W5P2_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);

			//kV2_88s1W3P1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
			//kV2_88s1W3P1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);

			//kV2_88s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
			//kV2_84s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
			//kV2_48s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
			//kV2_44s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);

			//kV2_88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
			//kV2_84s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
			//kV2_48s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
			//kV2_44s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
		}
		
		//kernel A
		{
			//k88As1W3(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//k88As1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			//k88As1W3_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			
			//k88As1W5(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//k88As1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			//k88As1W5_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);

			//k88As1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//k88As1_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);

			//k88As1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//k88As1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
		}

		//8*8 uernel
		{
			//u88As1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//u88As1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			
			//u88As1W3(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//u88As1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			//u88s1W3x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

			//u88As1W5(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//u88As1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			//u88s1W5x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
		}

		//8*8 kernel
		{
			//f88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1x4(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1x4(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			
			//f88s1x4_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1x4_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

			//==============================================================
			//k88s1W5x4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			//k88s1W5_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			
			//k88s1W5x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			//k88s1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

			//==============================================================
			//k88s1W3x4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			//k88s1W3_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			
			//k88s1W3x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			//k88s1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

			//==============================================================
			//k88s1_W2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, LOG2(FH), LOG2(FW), d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

			//k88s1_W2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, log2(FH), log2(FW), d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}

		//pure kernel
		{
			//k48s1_pure_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k44s1_pure_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

			//k84s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k48s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k44s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k82s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k28s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k42s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k24s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}

		//k44s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		//k22s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		//k21s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		//k12s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		//k11s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		//int index = 0; __dconv3D_deltaX_s1(streams, index, 8, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw);
	}

	//kernel W1
	{
		//k88W1_LB4(NULL, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k88W1_LB3(NULL, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k84W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k48W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k44W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k82W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k28W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k42W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k24W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k22W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k21W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k12W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k11W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//int index = 0; __dconv3D_deltaX_W1(streams, index, 8, d_deltaY, dW, d_deltaX, IH, IW, IC, OC);
	}

	error = cudaMemcpy(deltaX2, d_deltaX, sizeof(float)*sizeX, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: "; println(deltaX2, 10);
	float sp = samePercent(deltaX1, deltaX2, sizeX); cout << "sp: " << sp << endl;
	float zp1 = zeroPercent(deltaX2, sizeX); cout << "zp1: " << zp1 << endl;

	error = cudaGetLastError(); printError(error);
	
	//error = cudaFree(d_deltaY); printError(error);
	//error = cudaFree(dW); printError(error);
	//error = cudaFree(d_deltaX); printError(error);
	//error = cudaDestroyTextureObject(texDy); printError(error);

	delete deltaY;
	delete W;
	delete deltaX1;
	delete deltaX2;

	if (sp < 0.99f) {exit(2);}
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

	int GN = IC;
	int GM = N * IH*IW;
	int GK = OC * FH*FW;

	printf("Test Correct:\n");
	printf("\t(OH, OW) = (%d, %d)\n", OH, OW);
	printf("\t(FH, FW) = (%d, %d)\n", FH, FW);
	printf("\t(n, IC, OC) = (%d, %d, %d)\n", N, IC, OC);
	printf("\t(IH, IW) = (%d, %d)\n", IH, IW);
	printf("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
	printf("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);

	int sizeX = N  * IC * IH * IW;
	int sizeW = OC * IC * FH * FW;
	int sizeY = N  * OC * OH * OW;

	float* deltaY = newRandomFloatVec(sizeY);
	//float* W = newRandomFloatVec(sizeW);

	float* d_deltaY = newDevFloatVec(deltaY, sizeY);
	float* dW = newDevFloatVec(sizeW);
	float* d_deltaX = newDevFloatVec(sizeX);
	cudaError_t error;

	jlong streams[8];
	for (int i = 0; i < 8; i++) {
		cudaStream_t stream; cudaStreamCreate(&stream);
		streams[i] = (intptr_t)stream;
	}

	clock_t start = clock();
	cudaTextureObject_t texDy = floatTexture(d_deltaY, sizeY);

	float *dCW = 0L;
	int sizeWks = 0;
	{
		KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw);

		sizeWks = sh * sw * OC * CFH * CFW * IC;
		dCW = newDevFloatVec(sizeWks);
	}

	for (int i = 0; i < nIter; i++)
	{
		{
			//KernelSplit ImsR
			{
				//Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
				//__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);

				//===============================================================
				//ksIms_88R8_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
				//ksIms_88R4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
				//ksIms_88R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);

				//ksIms_88R8_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_88R4_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_88R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);

				//ksIms_84R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_84R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);

				//ksIms_48R_tex(NULL, LB, 0, 0,texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_48R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);

				//ksIms_44R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_44R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);

				//===============================================
				//ksIms_88R8_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
				//ksIms_88R4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
				//ksIms_88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);

				//ksIms_88R8(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_88R4(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);

				//ksIms_84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
				//ksIms_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);

				//ksIms_48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
				//ksIms_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);

				//ksIms_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);

				//ksIms_42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_22R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_21R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_12R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ksIms_11R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}

			//KernelSplit Ims2R
			{
				Ims2_init(N, IH, IW, FH, FW, OC, IC);
				__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);

				//ksIms2 V2
				{
					//ksV2_Ims2_u88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
					//ksV2_Ims2_u88R_CFW_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
					//ksV2_Ims2_u88R_W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);

					//========================================================
					//ksV2_Ims2_88R_CFW_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
					//ksV2_Ims2_88R_W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);

					//ksV2_Ims2_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
					//ksV2_Ims2_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
					//ksV2_Ims2_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
					//ksV2_Ims2_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
				}
				
				//============================================================================================
				
				//ksIms2_84R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_84R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

				//ksIms2_48R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_48R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

				//ksIms2_44R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_44R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

				//=======================================================================================
				
				//uernel
				{
					//ksIms2_u88R8_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_u88R4_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					ksIms2_u88R_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				}

				//cfw oc 2pow
				{
					//ksIms2_88R8_oc_CFW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R4_oc_CFW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);

					//ksIms2_88R8_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R4_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				}

				//oc 2pow
				{
					//ksIms2_88R8_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);

					//ksIms2_88R8_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				}
				
				//CW 2pow 
				{
					//ksIms2_88R8_CW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R4_CW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);

					//ksIms2_88R8_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R4_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				}

				//common 8*8
				{
					//ksIms2_88R8_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R4_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);

					//ksIms2_88R8(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R4(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				}

				//Kernel A
				{
					//ksIms2A_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
					//ksIms2A_88R_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
					//ksIms2A_88R_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOG2(OC), ph, pw, GN, GM);
				}
				
				//ksIms2_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
				//ksIms2_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
				//ksIms2_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
				//ksIms2_42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			}

			//Kernel Split
			{
				//KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
				//__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);

				//ks88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				//ks84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				//ks48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				//ks44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);

				//ks88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks22R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks21R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks12R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks11R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}
		}
		
		{
			//int index = 0; __dconv3D_deltaX_CrossAdd(streams, index, 8, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw);
			//crossAdd_k16_2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
			//crossAdd_k82(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
			//crossAdd_k42(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
			//crossAdd_k22(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
			//crossAdd_k11(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
		}

		//kernel s1
		{
			//kernel V2
			{
				//uV2_88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
				//uV2_88s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
				
				//uV2_88s1W3P1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
				//uV2_88s1W3P1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);

				//uV2_88s1W5P2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
				//uV2_88s1W5P2_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);
				
				//==================================================================================
				//kV2_88s1W5P2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
				//kV2_88s1W5P2_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);
				
				//kV2_88s1W3P1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
				//kV2_88s1W3P1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);

				//kV2_88s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
				//kV2_84s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
				//kV2_48s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
				//kV2_44s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);

				//kV2_88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
				//kV2_84s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
				//kV2_48s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
				//kV2_44s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
			}
			
			//kernel A	
			{
				//k88As1W3(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//k88As1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				//k88As1W3_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				
				//k88As1W5(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//k88As1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				//k88As1W5_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);

				//k88As1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//k88As1_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				
				//k88As1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//k88As1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			}

			//8*8 uernel
			{
				//u88As1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//u88As1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);

				//u88As1W3(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//u88As1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				//u88s1W3x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

				//u88As1W5(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//u88s1W5x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				//u88As1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			}

			//8*8 kernel
			{
				//f88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1x4(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1x4(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

				//f88s1x4_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1x4_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

				//==================================================================
				//k88s1W5x4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				//k88s1W5_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				
				//k88s1W5x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				//k88s1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
	
				//==================================================================
				//k88s1W3x4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				//k88s1W3_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				
				//k88s1W3x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				//k88s1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			
				//==================================================================
				//k88s1_W2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, LOG2(FH), LOG2(FW), d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

				//k88s1_W2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, log2(FH), log2(FW), d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			}

			//pure kernel
			{
				//k48s1_pure_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k44s1_pure_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				
				//k84s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k48s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k44s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k82s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k28s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k42s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k24s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			}
			
			//k44s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k22s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k21s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k12s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k11s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}

		//kernel W1
		{
			//k88W1_LB4(NULL, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k88W1_LB3(NULL, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k84W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k48W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k44W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k82W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k28W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k42W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k24W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k22W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k21W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k12W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k11W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//int index = 0; __dconv3D_deltaX_W1(streams, index, 8, d_deltaY, dW, d_deltaX, IH, IW, IC, OC);
		}
		//-------------------------------------
	}
	error = cudaDestroyTextureObject(texDy); printError(error);
	error = cudaGetLastError(); printError(error);
	error = cudaDeviceSynchronize(); printError(error);
	error = cudaGetLastError(); printError(error);
	clock_t end = clock();

	int div = end - start;
	float time = 1.0f*div / nIter;

	float size = 1.0f * GN / 1024 * GM / 1024 * GK / 1024;
	float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
	cout << "Size = " << size << ", Time = " << time << " msec, Performace = " << performance << " GFlop/s";

	error = cudaFree(d_deltaX); printError(error);
	error = cudaFree(d_deltaY); printError(error);

	delete deltaY;
	//delete W;
}


int maxInt(int a, int b)
{
	return b & ((a - b) >> 31) | a & (~(a - b) >> 31);
}

void test()
{
	//int OH = 30, OW = 30;
	//int FH = 3, FW = 3;
	//int N = 4;
	//int IC = 128, OC = 31;//9*4=36 
	//int sh = 1, sw = 1, ph = 1, pw = 1;

	//int IH = 32, IW = 32;
	//int OH = 16, OW = 16;
	//int FH = 4, FW = 4;//4*4*8 = 32*4 = 128
	//int N = 4;
	//int IC = 128, OC = 16;//9*4=36 3*3*3 = 9*3 = 27;
	//int sh = 2, sw = 2, ph = 1, pw = 1;

	//int IH = -1, IW = -1;
	//int OH = 15, OW = 15;
	//int FH = 6, FW = 6;
	//int N = 4, OC = 16, IC = 128;
	//int sh = 3, sw = 1;
	//int ph = 1, pw = 1;

	//int IH = 32, IW = 32;
	//int OH = 16, OW = 16;
	//int FH = 7, FW = 7;//4*4*8 = 32*4 = 128
	//int N = 4;
	//int IC = 128, OC = 16;//9*4=36 3*3*3 = 9*3 = 27;
	//int sh = 2, sw = 2, ph = 3, pw = 3;

	//(31 - 3 + 2) / 2 + 1 = 30 / 2 + 1 = 16
	int IH = 32, IW = 32;
	int OH = 16, OW = 16;
	int FH = 3, FW = 3;//4*4*8 = 32*4 = 128
	int N = 8;
	int IC = 128, OC = 64;//9*4=36 3*3*3 = 9*3 = 27;
	int sh = 2, sw = 2, ph = 1, pw = 1;

	//int N = 128, IC = 128;
	//int IH = 32, IW = 32, OH = 16, OW = 16, OC = 16;
	//int IH = 16, IW = 16, OH = 8, OW = 8, OC = 16;
	//int IH = 8, IW = 8, OH = 4, OW = 4, OC = 64;
	//int IH = 4, IW = 4, OH = 2, OW = 2, OC = 256;

	//int FH = 3, FW = 3, ph = 1, pw = 1;
	//int FH = 5, FW = 5, ph = 2, pw = 2; OC /= 2;
	//int sh = 2, sw = 2;

	//OC *= 4; N /= 4;

	//int IH = 31, IW = 31, OH = 16, OW = 16, N = 4;
	//int IC = 128, OC = 16;//9*4=36 3*3*3 = 9*3 = 27;
	//int sh = 2, sw = 2, ph = 2, pw = 2;

	//testCorrect<4>(IH, IW, OH, OW, 5, 5, N, IC, OC, sh, sw, 2, 2);//3*3*4 = 9*4=36
	//testCorrect<4>(IH, IW, OH, OW, 7, 7, N, IC, OC, sh, sw, 3, 3);//3*3*4 = 9*4=36
	//testCorrect<4>(IH, IW, OH, OW, 3, 3, N, IC, OC, sh, sw, 1, 1);//3*3*4 = 9*4=36
	
	testCorrect<4>(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);//3*3*4 = 9*4=36
	testSpeed<4>(1000, IH, IW, OH, OW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);//3*3*2 = 9*2

	//=============Area_for kernel S1=================================
	//int IH = -1, IW = -1;
	//int OH = 31, OW = 31;
	//int FH = 4, FW = 4;
	//int N = 8;
	//int IC = 128, OC = 32;//9*4=36 
	//int sh = 1, sw = 1, ph = 1, pw = 1;
	
	//IC /= 4;//IC -> 32
	//N *= 4;

	//int IH = -1, IW = -1;
	//int OH = 32, OW = 32;
	//int IC = 128, OC = 64;//9*4=36 
	////int FH = 3, FW = 3, ph = 1, pw = 1;
	//int FH = 5, FW = 5, ph = 2, pw = 2;
	//int N = 8;
	//int sh = 1, sw = 1;

	//testCorrect<4>(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);//3*3*4 = 9*4=36
	//testSpeed<4>(500, IH, IW, OH, OW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);//3*3*2 = 9*2
	
	//int FH = 5, FW = 5, OC = 128, ph = 2, pw = 2;//4*4*8 = 32*4 = 128
	//int FH = 3, FW = 3, OC = 256, ph = 1, pw = 1;//4*4*8 = 32*4 = 128
	//int FH = 1, FW = 1, OC = 256, ph = 0, pw = 0; OC *= 8;
	//int IC = 128;//9*4=36 3*3*3 = 9*3 = 27;
	//int sh = 1, sw = 1;

	//int IH = 2, IW = 2, OH = 2, OW = 2, N = 512;
	//int IH = 4, IW = 4, OH = 4, OW = 4, N = 128;
	//int IH = 8, IW = 8, OH = 8, OW = 8, N = 128; OC /= 4;
	//int IH = 16, IW = 16, OH = 16, OW = 16, N = 128; OC /= 16;
	//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128; OC /= 16;
	//int IH = 64, IW = 64, OH = 64, OW = 64, N = 128; OC /= 16;
	//int IH = 128, IW = 128, OH = 128, OW = 128, N = 128; OC /= 16;

	//IC /= 4;//IC -> 32
	//N *= 4;

	//testCorrect<3>(IH, IW, OH, OW, FH, FW, N, IC, OC + 8, sh, sw, ph, pw);//3*3*4 = 9*4=36
	//testCorrect<3>(IH, IW, OH, OW, FH, FW, N, IC, OC - 8, sh, sw, ph, pw);//3*3*4 = 9*4=36
	//testSpeed<3>(500, IH, IW, OH, OW, FH, FW, N, IC, OC*2, sh, sw, ph, pw);//3*3*2 = 9*2
}

main() {test();}

#endif