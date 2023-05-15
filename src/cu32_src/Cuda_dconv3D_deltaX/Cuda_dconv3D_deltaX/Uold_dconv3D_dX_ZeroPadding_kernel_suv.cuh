#pragma once

#ifndef MICRO
#define MICRO

//=================cancel the if condition============================
//#define load4d_suv(V, n, ohp, owp, oc) \
//	if (ohp < 0 || ohp >= OHp || owp < 0 || owp >= OWp) (V) = 0.0f;\
//	else {\
//		int oh = ohp / sh, ohr = ohp - sh*oh;\
//		int ow = owp / sw, owr = owp - sw*ow; \
//		bool lflag = (!ohr && !owr);\
//		(V) = lflag * get4d(deltaY, n, oh, ow, oc, OH, OW, OC);}
//=================cancel the if condition============================
#define load4d_suv(V, n, ohp, owp, oc) {\
	int oh = ohp / sh, ohr = ohp - sh*oh;\
	int ow = owp / sw, owr = owp - sw*ow;\
	bool lflag = (ohp>=0) && (ohp<OHp) && (owp>=0) &&(owp<OWp) && (!ohr) && (!owr);\
	intptr_t address1 = (intptr_t)(&get4d(deltaY, n, oh, ow, oc, OH, OW, OC));\
	intptr_t address2 = (intptr_t)(&_ZERO);\
	intptr_t address = (address1 - address2)*lflag + address2;\
	(V) = *(float*)(address); }

#define load4d_suv_tex(V, n, ohp, owp, oc) {\
	int oh = ohp / sh, ohr = ohp - sh*oh;\
	int ow = owp / sw, owr = owp - sw*ow;\
	bool lflag = (ohp>=0) && (ohp<OHp) && (owp>=0) &&(owp<OWp) && (!ohr) && (!owr);\
	int yoffset = ((n*OH + oh)*OW + ow)*OC + oc;\
	(V) = lflag * tex1Dfetch<float>(deltaY, yoffset);}

#endif


//Dense Kernel : for sh*sw < 4
#ifndef DECONV_3D_DELTAX_ZERO_PADDING_SUV
#define DECONV_3D_DELTAX_ZERO_PADDING_SUV

#define __dconv3D_deltaX_ZeroPadding_suv(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw) \
	dconv3d_deltaX_ZeroPadding_suv(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
		GET_GN_ZeroPadding(IC), GET_GM_ZeroPadding(N,IH,IW), GET_GK_ZeroPadding(OC,FH,FW), 0, 0)\

#define dconv3d_dX_suv_Zero_Padding_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_ZeroPadding_suv(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
			GNr, GM, GK, next_ic_index, j_index);\
		dconv3d_deltaX_ZeroPadding_suv(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
			GN, GMr, GK, ic_index, next_j_index);\
		dconv3d_deltaX_ZeroPadding_suv(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
			GNr, GMr, GK, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3d_deltaX_ZeroPadding_suv(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
			GNr, GM, GK, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_ZeroPadding_suv(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
			GN, GMr, GK, ic_index, next_j_index);}}

//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//performance=============================================================
//[OH, OW] = 31, [FH, FW] = 4, [N, OC] = 4, 64, [sh, sw] = 2, [ph, pw] = 1
//IC = 128: Size = 1.87695, Time = 4.68   msec, Performace = 861.266 GFlop/s
//IC = 192: Size = 1.87695, Time = 4.68   msec, Performace = 861.266 GFlop/s
//IC = 224: Size = 3.28467, Time = 10.04  msec, Performace = 702.567 GFlop/s
//IC = 240: Size = 3.51929, Time = 12.438 msec, Performace = 607.623 GFlop/s
//IC = 248: Size = 3.6366 , Time = 14.932 msec, Performace = 523.006 GFlop/s
//IC = 252: Size = 3.69525, Time = 19.622 msec, Performace = 404.418 GFlop/s
void dconv3d_deltaX_ZeroPadding_suv(jlong *streams, int &index, int length,
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM, int GK,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 127) && (GM > 127) && !(GK & 7)) {//[128, 128], GK % 8 == 0
		k88suv(stream, 4, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_suv_Zero_Padding_Branch(127, 127); return;
	}
	if ((GN > 63) && (GM > 63)) {//[64, 64]
		k88suv(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_suv_Zero_Padding_Branch(63, 63); return;
	}
	if ((GN > 31) && (GM > 31)) {//[32, 32]
		k44suv(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_suv_Zero_Padding_Branch(31, 31); return;
	}
	if ((GN > 31) && (GM > 15)) {//[32, 16]
		k42suv(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_suv_Zero_Padding_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		k24suv(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_suv_Zero_Padding_Branch(15, 31); return;
	}

	if (GK > 7)//GK >= 8
	{
		if ((GN > 15) && (GM > 15)) {//[16, 16] 
			k22suv(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_suv_Zero_Padding_Branch(15, 15); return;
		}
		if ((GN > 15) && (GM > 7)) {//[16, 8]
			k21suv(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_suv_Zero_Padding_Branch(15, 7); return;
		}
		if ((GN > 7) && (GM > 15)) {//[8, 15]
			k12suv(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_suv_Zero_Padding_Branch(7, 15); return;
		}
		if ((GN > 7) && (GM > 7)) {//[8, 8]
			k11suv(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_suv_Zero_Padding_Branch(7, 7); return;
		}
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8]
		k22suv(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_suv_Zero_Padding_Branch(7, 7); return;
	}
	if (GN > 7) {//[8 ,4]
		k21suv(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_suv_Zero_Padding_Branch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		k12suv(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_suv_Zero_Padding_Branch(3, 7); return;
	}
	k11suv(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif

#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_SUV_H
#define DECONV3D_DX_ZERO_PADDING_KERNEL_SUV_H

//Unsparse Matrix Method
//We have:
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) ph = 0, pw = 0

#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_SUV_CALL
#define DECONV3D_DX_ZERO_PADDING_KERNEL_SUV_CALL

#define k88suv(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	kernel_8_8_suv<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, ic_index,j_index)

#define k44suv(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	kernel_4_4_suv<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, ic_index,j_index)

#define k42suv(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	kernel_4_2_suv<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, ic_index,j_index)\

#define k24suv(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	kernel_2_4_suv<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, ic_index,j_index)

#define k22suv(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	kernel_2_2_suv<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, ic_index,j_index)

#define k21suv(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	kernel_2_1_suv<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, ic_index,j_index)

#define k12suv(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	kernel_1_2_suv<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, ic_index,j_index)

#define k11suv(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	kernel_1_1_suv<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, ic_index,j_index)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8) GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 ==0
#ifndef DECONV3D_DX_KERNEL_8_8_SUV
#define DECONV3D_DX_KERNEL_8_8_SUV

//So when sh * sw > 4: use sparese kernel, else use dense kernel
//(sh, sw) = (1, 2) = (2, 1)
//LB = 4: Size = 0.96875, Time = 2.67 msec, Performace = 779.166 GFlop/s
//k81<3>: Size = 0.96875, Time = 3.838 msec, Performace = 542.047 GFlop/s
//(sh, sw) = (2, 2) = (1, 4)
//LB = 4: Size = 0.938477, Time = 2.292 msec, Performace = 879.303 GFlop/s
//LB = 3: Size = 0.938477, Time = 3.69  msec, Performace = 546.169 GFlop/s
//LB = 4: Size = 1, Time = 2.518 msec, Performace = 852.853 GFlop/s
//LB = 3: Size = 1, Time = 3.142 msec, Performace = 683.477 GFlop/s
//k81<3>: Size = 1.87695, Time = 3.82 msec, Performace = 1055.16 GFlop/s 
//(sh, sw) = (1, 3) = (3, 1)
//LB = 4: Size = 1.4375, Time = 3.768 msec, Performace = 819.27 GFlop/s
//k81<3>: Size = 1.4375, Time = 5.992 msec, Performace = 515.188 GFlop/s
//(sh, sw) = (3, 3)
//LB = 4: Size = 4.13281, Time = 10.05 msec, Performace = 883.099 GFlop/s
//k81<3>: Size = 4.13281, Time = 5.542 msec, Performace = 1601.43 GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_8_suv(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 3) + ic_index;
	const int tic0 = ic0 + ((tx >= STEP) << 2);
	
	//prepared for GM = N * IH * IW
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = OC * FH_FW;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx - ((tx >= STEP) << LB >> 1), W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	Ws[buf][tx][ty] = *(float4*)(&get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC));

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty - ((ty >= STEP) << LB >> 1), dY_oc, dY_fh, dY_fw;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	int ohp0 = tih0 + dY_fh, owp0 = tiw0 + dY_fw;
	int ohp1 = tih1 + dY_fh, owp1 = tiw1 + dY_fw;
	int ohp2 = tih2 + dY_fh, owp2 = tiw2 + dY_fw;
	int ohp3 = tih3 + dY_fh, owp3 = tiw3 + dY_fw;
	load4d_suv(dYs[buf][ty][tx].x, tn0, ohp0, owp0, dY_oc);
	load4d_suv(dYs[buf][ty][tx].y, tn1, ohp1, owp1, dY_oc);
	load4d_suv(dYs[buf][ty][tx].z, tn2, ohp2, owp2, dY_oc);
	load4d_suv(dYs[buf][ty][tx].w, tn3, ohp3, owp3, dY_oc);
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];
			float4  w0 =  Ws[buf][ik][ty],  w1 =  Ws[buf][ik + STEP][ty];

			//transposed compute core: (W * dY)^T
			simdMM4( v0, dy0.x, w0); simdMM4( v1, dy0.x, w1);
			simdMM4( v2, dy0.y, w0); simdMM4( v3, dy0.y, w1);
			simdMM4( v4, dy0.z, w0); simdMM4( v5, dy0.z, w1);
			simdMM4( v6, dy0.w, w0); simdMM4( v7, dy0.w, w1);
			simdMM4( v8, dy1.x, w0); simdMM4( v9, dy1.x, w1);
			simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
			simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
			simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx; //Ws: with the same ty
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][tx][ty] = *(float4*)(&get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC));

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok - (ty >= STEP)) << LB >> 1) + ty; //dYs: with the same tx
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		int ohp0 = tih0 + dY_fh, owp0 = tiw0 + dY_fw;
		int ohp1 = tih1 + dY_fh, owp1 = tiw1 + dY_fw;
		int ohp2 = tih2 + dY_fh, owp2 = tiw2 + dY_fw;
		int ohp3 = tih3 + dY_fh, owp3 = tiw3 + dY_fw;
		load4d_suv(dYs[buf][ty][tx].x, tn0, ohp0, owp0, dY_oc);
		load4d_suv(dYs[buf][ty][tx].y, tn1, ohp1, owp1, dY_oc);
		load4d_suv(dYs[buf][ty][tx].z, tn2, ohp2, owp2, dY_oc);
		load4d_suv(dYs[buf][ty][tx].w, tn3, ohp3, owp3, dY_oc);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];
		float4  w0 =  Ws[buf][ik][ty],  w1 =  Ws[buf][ik + STEP][ty];

		simdMM4( v0, dy0.x, w0); simdMM4( v1, dy0.x, w1);
		simdMM4( v2, dy0.y, w0); simdMM4( v3, dy0.y, w1);
		simdMM4( v4, dy0.z, w0); simdMM4( v5, dy0.z, w1);
		simdMM4( v6, dy0.w, w0); simdMM4( v7, dy0.w, w1);
		simdMM4( v8, dy1.x, w0); simdMM4( v9, dy1.x, w1);
		simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
		simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
		simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
	}

	j0 = j0 * IC + ic0; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) =  v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) =  v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) =  v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) =  v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) =  v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4) GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 ==0
#ifndef DECONV3D_DX_KERNEL_4_4_SUV
#define DECONV3D_DX_KERNEL_4_4_SUV

//LB = 4: Size = 1.87695, Time = 6.336 msec, Performace = 636.163 GFlop/s
//LB = 3: Size = 1.87695, Time = 10.33 msec, Performace = 390.196 GFlop/s
//LB = 4: Size = 1, Time = 3.478 msec, Performace = 617.448 GFlop/s
//LB = 3: Size = 1, Time = 5.722 msec, Performace = 375.303 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_4_suv(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = IC
	const int ic0 = (((by << LB) + ty) << 2) + ic_index;
	const int tic0 = ((tx & 1) << 1) + ic0;

	//prepared for GM = N * IH * IW
	int j0 = (((bx << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = OC * FH_FW;

	//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx >> 1, W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC));

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty >> 1, dY_oc, dY_fh, dY_fw;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	int ohp0 = tih0 + dY_fh, owp0 = tiw0 + dY_fw;
	int ohp1 = tih1 + dY_fh, owp1 = tiw1 + dY_fw;
	const int dYs_y = (ty >> 1), dY_x = (tx << 1) + (ty & 1);
	load4d_suv(dYs[buf][dYs_y][dY_x].x, tn0, ohp0, owp0, dY_oc);
	load4d_suv(dYs[buf][dYs_y][dY_x].y, tn1, ohp1, owp1, dY_oc);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
			float4  w = *(float4*)(&Ws[buf][ik][ty << 1]);
			simdMM4(v0, dy.x, w); 
			simdMM4(v1, dy.y, w);
			simdMM4(v2, dy.z, w);
			simdMM4(v3, dy.w, w);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok << LB) + tx) >> 1; //Ws: with the same ty
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_x][Ws_y] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC));

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok << LB) + ty) >> 1; //dYs: with the same tx
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		int ohp0 = tih0 + dY_fh, owp0 = tiw0 + dY_fw;
		int ohp1 = tih1 + dY_fh, owp1 = tiw1 + dY_fw;
		load4d_suv(dYs[buf][dYs_y][dY_x].x, tn0, ohp0, owp0, dY_oc);
		load4d_suv(dYs[buf][dYs_y][dY_x].y, tn1, ohp1, owp1, dY_oc);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
		float4  w = *(float4*)(&Ws[buf][ik][ty << 1]);
		simdMM4(v0, dy.x, w);
		simdMM4(v1, dy.y, w);
		simdMM4(v2, dy.z, w);
		simdMM4(v3, dy.w, w);
	}

	j0 = j0*IC + ic0; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	*(float4*)(deltaX + j0) = v0; 
	*(float4*)(deltaX + j1) = v1;
	*(float4*)(deltaX + j2) = v2;
	*(float4*)(deltaX + j3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2) GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 ==0
#ifndef DECONV3D_DX_KERNEL_4_2_SUV
#define DECONV3D_DX_KERNEL_4_2_SUV

//LB = 4: Size = 0.938477, Time = 4.478 msec, Performace = 450.059 GFlop/s
//LB = 3: Size = 0.938477, Time = 7.312 msec, Performace = 275.624 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_2_suv(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB << LB];
	__shared__ float dYs[2][1 << LB << LB];

	//prepare for GN = IC
	const int ic0 = (((by << LB) + ty) << 2) + ic_index;
	const int tic0 = ((tx & 1) << 1) + ic0;

	//prepared for GM = N * IH * IW
	int j0 = (((bx << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int IH_IW = IH * IW;
	get_n_ih_iw(j0, n0, ih0, iw0);
	get_n_ih_iw(j1, n1, ih1, iw1);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	bool flagY = (ty & 1);
	int tn0 = (n1 - n0)*flagY + n0;
	int tih0 = ((ih1 - ih0)*flagY + ih0) - oph;
	int tiw0 = ((iw1 - iw0)*flagY + iw0) - opw;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = OC * FH_FW;

	//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx >> 1, W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	const int Ws_xy = ((tx >> 1) << 1 << LB) + (ty << 1) + (tx & 1);
	Ws[buf][Ws_xy] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC));

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty >> 1, dY_oc, dY_fh, dY_fw;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	int ohp0 = tih0 + dY_fh, owp0 = tiw0 + dY_fw;
	const int dYs_yx = ((ty >> 1) << 1 << LB) + (tx << 1) + (ty & 1);
	load4d_suv(dYs[buf][dYs_yx], tn0, ohp0, owp0, dY_oc);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 dy = *(float2*)(&dYs[buf][(((ik << LB) + tx) << 1)]);
			float4  w = *(float4*)(&Ws[buf][(((ik << LB) + ty) << 1)]);
			simdMM4(v0, dy.x, w);
			simdMM4(v1, dy.y, w);
		}

		buf ^= 1;
		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		W_k = ((ok << LB) + tx) >> 1; //Ws: with the same ty
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_xy] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC));

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		dY_k = ((ok << LB) + ty) >> 1; //dYs: with the same tx
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		ohp0 = tih0 + dY_fh, owp0 = tiw0 + dY_fw;
		load4d_suv(dYs[buf][dYs_yx], tn0, ohp0, owp0, dY_oc);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 dy = *(float2*)(&dYs[buf][(((ik << LB) + tx) << 1)]);
		float4  w = *(float4*)(&Ws[buf][(((ik << LB) + ty) << 1)]);
		simdMM4(v0, dy.x, w);
		simdMM4(v1, dy.y, w);
	}

	j0 *= IC; j1 = j0 + IC; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	*(float4*)(&deltaX[j0 + ic0]) = v0;
	*(float4*)(&deltaX[j1 + ic0]) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4) GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 ==0
#ifndef DECONV3D_DX_KERNEL_2_4_SUV
#define DECONV3D_DX_KERNEL_2_4_SUV

//LB = 4: Size = 0.938477, Time = 5.488 msec, Performace = 367.231 GFlop/s
//LB = 3: Size = 0.938477, Time = 9.452 msec, Performace = 213.221 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_4_suv(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float   Ws[2][1 << LB << LB];
	__shared__ float2 dYs[2][1 << LB << LB];

	//prepare for GN = IC
	const int ic0 = (((by << LB) + ty) << 1) + ic_index;
	const int tic0 = (tx & 1) + ic0;

	//prepared for GM = N * IH * IW
	int j0 = (((bx << LB) + tx) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(j0, n0, ih0, iw0);
	get_n_ih_iw(j1, n1, ih1, iw1);
	get_n_ih_iw(j2, n2, ih2, iw2);
	get_n_ih_iw(j3, n3, ih3, iw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	bool flagY = (ty & 1);
	int tn0 = (n2 - n0)*flagY + n0;
	int tn1 = (n3 - n1)*flagY + n1;
	int tih0 = ((ih2 - ih0)*flagY + ih0) - oph;
	int tih1 = ((ih3 - ih1)*flagY + ih1) - oph;
	int tiw0 = ((iw2 - iw0)*flagY + iw0) - opw;
	int tiw1 = ((iw3 - iw1)*flagY + iw1) - opw;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = OC * FH_FW;

	//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx >> 1, W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	const int Ws_xy = ((tx >> 1) << 1 << LB) + (ty << 1) + (tx & 1);
	Ws[buf][Ws_xy] = get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC);

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty >> 1, dY_oc, dY_fh, dY_fw;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	int ohp0 = tih0 + dY_fh, owp0 = tiw0 + dY_fw;
	int ohp1 = tih1 + dY_fh, owp1 = tiw1 + dY_fw;
	const int dYs_yx = ((ty >> 1) << 1 << LB) + (tx << 1) + (ty & 1);
	load4d_suv(dYs[buf][dYs_yx].x, tn0, ohp0, owp0, dY_oc);
	load4d_suv(dYs[buf][dYs_yx].y, tn1, ohp1, owp1, dY_oc);
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 dy = *(float4*)(&dYs[buf][(((ik << LB) + tx) << 1)]);
			float2  w = *(float2*)(&Ws[buf][(((ik << LB) + ty) << 1)]);

			simdMM2(v0, dy.x, w);
			simdMM2(v1, dy.y, w);
			simdMM2(v2, dy.z, w);
			simdMM2(v3, dy.w, w);
		}

		buf ^= 1;
		//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		W_k = ((ok << LB) + tx) >> 1; //Ws: with the same ty
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_xy] = get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		dY_k = ((ok << LB) + ty) >> 1; //dYs: with the same tx
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		ohp0 = tih0 + dY_fh, owp0 = tiw0 + dY_fw;
		ohp1 = tih1 + dY_fh, owp1 = tiw1 + dY_fw;
		load4d_suv(dYs[buf][dYs_yx].x, tn0, ohp0, owp0, dY_oc);
		load4d_suv(dYs[buf][dYs_yx].y, tn1, ohp1, owp1, dY_oc);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 dy = *(float4*)(&dYs[buf][(((ik << LB) + tx) << 1)]);
		float2  w = *(float2*)(&Ws[buf][(((ik << LB) + ty) << 1)]);

		simdMM2(v0, dy.x, w);
		simdMM2(v1, dy.y, w);
		simdMM2(v2, dy.z, w);
		simdMM2(v3, dy.w, w);
	}

	j0 *= IC; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	j1 = j0 + IC; j2 = j1 + IC; j3 = j2 + IC;
	*(float2*)(&deltaX[j0 + ic0]) = v0;
	*(float2*)(&deltaX[j1 + ic0]) = v1;
	*(float2*)(&deltaX[j2 + ic0]) = v2;
	*(float2*)(&deltaX[j3 + ic0]) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2) GK >= BLOCK_SIZE
//LB = 4: GK > 16
#ifndef DECONV3D_DX_KERNEL_2_2_SUV
#define DECONV3D_DX_KERNEL_2_2_SUV

//LB = 4: Size = 0.757027, Time = 6.59  msec, Performace = 246.692 GFlop/s
//LB = 3: Size = 0.757027, Time = 9.658 msec, Performace = 168.327 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_2_suv(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB << LB];
	__shared__ float2 dYs[2][1 << LB << LB];

	//prepare for GN = IC
	const int ic0 = (((by << LB) + ty) << 1) + ic_index;

	//prepared for GM = N * IH * IW
	const int IH_IW = IH * IW;
	int j0 = (((bx << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_ih_iw(j0, n0, ih0, iw0);
	get_n_ih_iw(j1, n1, ih1, iw1);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	ih0 -= oph; iw0 -= opw;
	ih1 -= oph; iw1 -= opw;
	
	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = OC * FH_FW;

	//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx, W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	const int Ws_xy = (tx << LB) + ty;
	Ws[buf][Ws_xy] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC));

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty, dY_oc, dY_fh, dY_fw;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	int ohp0 = ih0 + dY_fh, owp0 = iw0 + dY_fw;
	int ohp1 = ih1 + dY_fh, owp1 = iw1 + dY_fw;
	const int dYs_yx = (ty << LB) + tx;
	load4d_suv(dYs[buf][dYs_yx].x, n0, ohp0, owp0, dY_oc);
	load4d_suv(dYs[buf][dYs_yx].y, n1, ohp1, owp1, dY_oc);
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = GK >> LB; ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 dy = dYs[buf][(ik << LB) + tx];
			float2  w = Ws[buf][(ik << LB) + ty];
			simdMM2(v0, dy.x, w);
			simdMM2(v1, dy.y, w);
		}

		buf ^= 1;
		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		W_k = (ok << LB) + tx;
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_xy] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC));

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		dY_k = (ok << LB) + ty;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		ohp0 = ih0 + dY_fh, owp0 = iw0 + dY_fw;
		ohp1 = ih1 + dY_fh, owp1 = iw1 + dY_fw;
		load4d_suv(dYs[buf][dYs_yx].x, n0, ohp0, owp0, dY_oc);
		load4d_suv(dYs[buf][dYs_yx].y, n1, ohp1, owp1, dY_oc);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 dy = dYs[buf][(ik << LB) + tx];
		float2  w = Ws[buf][(ik << LB) + ty];
		simdMM2(v0, dy.x, w);
		simdMM2(v1, dy.y, w);
	}

	//when GK % STEP != 0------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		dY_k = k;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		Wr_fh_fw = FH_FW - 1 - dY_k;
		
		//load 2 elements from W
		float2 w = *(float2*)(&get3d(W, dY_oc, Wr_fh_fw, ic0, FH_FW, IC));

		//load 2 elements from deltaY
		int oh0 = ih0 + dY_fh, ow0 = iw0 + dY_fw;
		int oh1 = ih1 + dY_fh, ow1 = iw1 + dY_fw;
		float2 dy;
		load4d_suv(dy.x, n0, oh0, ow0, dY_oc);
		load4d_suv(dy.y, n1, oh1, ow1, dY_oc);
		
		simdMM2(v0, dy.x, w);
		simdMM2(v1, dy.y, w);
	}
	//when GK % STEP != 0------------------------------------

	j0 *= IC; j1 = j0 + IC;//j -> [n, oh, ow]
	*(float2*)(&deltaX[j0 + ic0]) = v0;
	*(float2*)(&deltaX[j1 + ic0]) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1) GK >= BLOCK_SIZE
//LB = 4: GK > 16
#ifndef DECONV3D_DX_KERNEL_2_1_SUV
#define DECONV3D_DX_KERNEL_2_1_SUV

//LB = 4: Size = 0.757027, Time =  9.664 msec, Performace = 168.223 GFlop/s
//LB = 3: Size = 0.757027, Time = 13.938 msec, Performace = 116.638 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_1_suv(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB << LB];
	__shared__ float  dYs[2][1 << LB << LB];

	//prepare for GN = IC
	const int ic0 = (((by << LB) + ty) << 1) + ic_index;

	//prepared for GM = N * IH * IW
	const int IH_IW = IH * IW;
	int j0 = ((bx << LB) + tx) + j_index;
	get_n_ih_iw(j0, n0, ih0, iw0);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	ih0 -= oph; iw0 -= opw;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = OC * FH_FW;

	//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx, W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	const int Ws_xy = (tx << LB) + ty;
	Ws[buf][Ws_xy] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC));

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty, dY_oc, dY_fh, dY_fw;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	int ohp0 = ih0 + dY_fh, owp0 = iw0 + dY_fw;
	const int dYs_yx = (ty << LB) + tx;
	load4d_suv(dYs[buf][dYs_yx], n0, ohp0, owp0, dY_oc);
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = GK >> LB; ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float dy = dYs[buf][(ik << LB) + tx];
			float2 w = Ws[buf][(ik << LB) + ty];
			simdMM2(v, dy, w);
		}

		buf ^= 1;
		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		W_k = (ok << LB) + tx;
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_xy] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC));

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		dY_k = (ok << LB) + ty;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		ohp0 = ih0 + dY_fh, owp0 = iw0 + dY_fw;
		load4d_suv(dYs[buf][dYs_yx], n0, ohp0, owp0, dY_oc);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float dy = dYs[buf][(ik << LB) + tx];
		float2 w = Ws[buf][(ik << LB) + ty];
		simdMM2(v, dy, w);
	}

	//when GK % STEP != 0------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		dY_k = k;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		Wr_fh_fw = FH_FW - 1 - dY_k;

		//load 2 elements from W
		float2 w = *(float2*)(&get3d(W, dY_oc, Wr_fh_fw, ic0, FH_FW, IC));

		//load 1 element from deltaY
		int oh0 = ih0 + dY_fh, ow0 = iw0 + dY_fw;
		float dy;
		load4d_suv(dy, n0, oh0, ow0, dY_oc);

		simdMM2(v, dy, w);
	}
	//when GK % STEP != 0------------------------------------

	*(float2*)(&deltaX[j0 * IC + ic0]) = v;//j -> [n, oh, ow]
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2) GK >= BLOCK_SIZE
//LB = 4: GK > 16
#ifndef DECONV3D_DX_KERNEL_1_2_SUV
#define DECONV3D_DX_KERNEL_1_2_SUV

//LB = 4: Size = 0.757027, Time = 13.148 msec, Performace = 123.646 GFlop/s
//LB = 3: Size = 0.757027, Time = 18.65  msec, Performace = 87.169 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_2_suv(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float   Ws[2][1 << LB << LB];
	__shared__ float2 dYs[2][1 << LB << LB];

	//prepare for GN = IC
	const int ic0 = ((by << LB) + ty) + ic_index;

	//prepared for GM = N * IH * IW
	const int IH_IW = IH * IW;
	int j0 = (((bx << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_ih_iw(j0, n0, ih0, iw0);
	get_n_ih_iw(j1, n1, ih1, iw1);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	ih0 -= oph; iw0 -= opw;
	ih1 -= oph; iw1 -= opw;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = OC * FH_FW;

	//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx, W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	const int Ws_xy = (tx << LB) + ty;
	Ws[buf][Ws_xy] = get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC);

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty, dY_oc, dY_fh, dY_fw;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	int ohp0 = ih0 + dY_fh, owp0 = iw0 + dY_fw;
	int ohp1 = ih1 + dY_fh, owp1 = iw1 + dY_fw;
	const int dYs_yx = (ty << LB) + tx;
	load4d_suv(dYs[buf][dYs_yx].x, n0, ohp0, owp0, dY_oc);
	load4d_suv(dYs[buf][dYs_yx].y, n1, ohp1, owp1, dY_oc);
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = GK >> LB; ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 dy = dYs[buf][(ik << LB) + tx];
			float  w = Ws[buf][(ik << LB) + ty];
			simdMM2(v, w, dy);
		}

		buf ^= 1;
		//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		W_k = (ok << LB) + tx;
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_xy] = get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		dY_k = (ok << LB) + ty;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		ohp0 = ih0 + dY_fh, owp0 = iw0 + dY_fw;
		ohp1 = ih1 + dY_fh, owp1 = iw1 + dY_fw;
		load4d_suv(dYs[buf][dYs_yx].x, n0, ohp0, owp0, dY_oc);
		load4d_suv(dYs[buf][dYs_yx].y, n1, ohp1, owp1, dY_oc);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 dy = dYs[buf][(ik << LB) + tx];
		float  w = Ws[buf][(ik << LB) + ty];
		simdMM2(v, w, dy);
	}

	//when GK % STEP != 0------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		dY_k = k;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		Wr_fh_fw = FH_FW - 1 - dY_k;

		//load 1 elements from W
		float w = get3d(W, dY_oc, Wr_fh_fw, ic0, FH_FW, IC);

		//load 2 elements from deltaY
		int oh0 = ih0 + dY_fh, ow0 = iw0 + dY_fw;
		int oh1 = ih1 + dY_fh, ow1 = iw1 + dY_fw;
		float2 dy;
		load4d_suv(dy.x, n0, oh0, ow0, dY_oc);
		load4d_suv(dy.y, n1, oh1, ow1, dY_oc);

		simdMM2(v, w, dy);
	}
	//when GK % STEP != 0------------------------------------

	j0 *= IC; j1 = j0 + IC;//j -> [n, oh, ow]
	deltaX[j0 + ic0] = v.x;
	deltaX[j1 + ic0] = v.y;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1) GK >= BLOCK_SIZE
//LB = 4: GK > 16
#ifndef DECONV3D_DX_KERNEL_1_1_SUV
#define DECONV3D_DX_KERNEL_1_1_SUV

//LB = 4: Size = 0.757027, Time = 17.786 msec, Performace = 91.4035 GFlop/s
//LB = 3: Size = 0.757027, Time = 26.91  msec, Performace = 60.4126 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_1_suv(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB << LB];
	__shared__ float dYs[2][1 << LB << LB];

	//prepare for GN = IC
	const int ic0 = ((by << LB) + ty) + ic_index;

	//prepared for GM = N * IH * IW
	const int IH_IW = IH * IW;
	int j0 = (bx << LB) + tx + j_index;
	get_n_ih_iw(j0, n0, ih0, iw0);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	ih0 -= oph; iw0 -= opw;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = OC * FH_FW;

	//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx, W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	const int Ws_xy = (tx << LB) + ty;
	Ws[buf][Ws_xy] = get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC);

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty, dY_oc, dY_fh, dY_fw;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	int ohp0 = ih0 + dY_fh, owp0 = iw0 + dY_fw;
	const int dYs_yx = (ty << LB) + tx;
	load4d_suv(dYs[buf][dYs_yx], n0, ohp0, owp0, dY_oc);
	__syncthreads();

	//compute area----------------------------------------------------
	float v = 0.0f;
	for (int ok = 1, OK = GK >> LB; ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float dy = dYs[buf][(ik << LB) + tx];
			float  w = Ws[buf][(ik << LB) + ty];
			v += dy * w;
		}

		buf ^= 1;
		//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		W_k = (ok << LB) + tx;
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_xy] = get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC);

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		dY_k = (ok << LB) + ty;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		ohp0 = ih0 + dY_fh, owp0 = iw0 + dY_fw;
		load4d_suv(dYs[buf][dYs_yx], n0, ohp0, owp0, dY_oc);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float dy = dYs[buf][(ik << LB) + tx];
		float  w = Ws[buf][(ik << LB) + ty];
		v += dy * w;
	}

	//when GK % STEP != 0------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		dY_k = k;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		Wr_fh_fw = FH_FW - 1 - dY_k;

		//load 1 element from W
		float w = get3d(W, dY_oc, Wr_fh_fw, ic0, FH_FW, IC);

		//load 1 element from deltaY
		int oh0 = ih0 + dY_fh, ow0 = iw0 + dY_fw;
		float dy;
		load4d_suv(dy, n0, oh0, ow0, dY_oc);

		v += dy * w;
	}
	//when GK % STEP != 0------------------------------------

	deltaX[j0 * IC + ic0] = v;//j -> [n, oh, ow]
}

#endif

#endif