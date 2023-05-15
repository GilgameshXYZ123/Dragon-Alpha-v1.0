#pragma once

#ifndef CONV_3D_GEMMR_UERNEL_H
#define CONV_3D_GEMMR_UERNEL_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef CONV_3D_GEMMR_UERNEL_CALL
#define CONV_3D_GEMMR_UERNEL_CALL

//======[Common]==============================================
#define conv3dGemm_u88R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dGemm_u88R4(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//======[Pure: Direct Conv]===================================
#define conv3dPure_u48R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_uernel_4_8R<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_u44R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_uernel_4_4R<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//======[IC is power of 2]====================================
#define conv3dGemm_u88R_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R_IC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dGemm_u88R4_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4_IC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//======[FH = FW = 3]=========================================
#define conv3dGemm_u88RW3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RW3<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#define conv3dGemm_u88RW3_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RW3_IC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//------[(OH, OW) % 4 == 0]-----------------------------------
#define conv3dGemm_u88R4W3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4W3<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#define conv3dGemm_u88R4W3_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4W3_IC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//======[FH = FW = 5]=========================================
#define conv3dGemm_u88RW5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RW5<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#define conv3dGemm_u88RW5_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RW5_IC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//------[(OH, OW) % 4 == 0]-----------------------------------s
#define conv3dGemm_u88R4W5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4W5<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#define conv3dGemm_u88R4W5_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4W5_IC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#endif


//======[Common]==============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R
#define CONV_3D_GEMM_UERNEL_8_8R

//LB = 4: Size = 1, Time = 1.48025 msec, Performace = 1450.75 GFlop/s
//LB = 3: Size = 1, Time = 1.60366 msec, Performace = 1339.11 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, (OH, OW) % 4 == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4
#define CONV_3D_GEMM_UERNEL_8_8R4

//LB = 4: Size = 1, Time = 1.4671 msec, Performace = 1463.76 GFlop/s
//LB = 3:Size = 1, Time = 1.61969 msec, Performace = 1325.86 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8R4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;
	const int sw_IC = sw * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//======[Pure: Direct Conv]===================================
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_PURE_UERNEL_4_8R
#define CONV_3D_PURE_UERNEL_4_8R

//[OC = 64]:
//LB = 4: Size = 1.125, Time = 2.17813 msec, Performace = 1109.17 GFlop/s
//LB = 3: Size = 1.125, Time = 2.59467 msec, Performace =  931.109 GFlop/s
//[OC = 32]:
//LB = 3: Size = 1.125, Time = 3.30615 msec, Performace =  730.734 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dPure_uernel_4_8R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty & 1) << 1);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int tihs2 = toh2 * sh - ph, tiws2 = tow2 * sw - pw;
	const int tihs3 = toh3 * sh - ph, tiws3 = tow3 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;
	const int X2 = (((tn2 *IH) + tihs2) * IW + tiws2) * IC;
	const int X3 = (((tn3 *IH) + tihs3) * IW + tiws3) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0);
	float4 v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0);
	float4 v7 = make_float4(0, 0, 0, 0);

	const int Ws_y = (ty >> 1) << 1, Ws_x = (tx << 1) + (ty & 1);
	const int OIC = (IC >> LB), SX = (IW - FW) *IC, SCW = IC * OC;

	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, CW += SCW)
		{
			//load 4 elem from X
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			bool lx2 = LOAD_X(tihs2, tiws2, fh, fw);
			bool lx3 = LOAD_X(tihs3, tiws3, fh, fw);
			int Xic = (tx - ((tx >= STEP) << LB >> 1)) << 1;
			float2 x0 = (lx0 ? *(float2*)(X + X0 + Xic) : F32_2_0);
			float2 x1 = (lx1 ? *(float2*)(X + X1 + Xic) : F32_2_0);
			float2 x2 = (lx2 ? *(float2*)(X + X2 + Xic) : F32_2_0);
			float2 x3 = (lx3 ? *(float2*)(X + X3 + Xic) : F32_2_0);
			Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
			Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
			
			//load 2 elem from W
			int Wic = (ty >> 1) << 1;
			int woffset0 = Wic * OC, woffset1 = woffset0 + OC;
			Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + woffset0);
			Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(CW + woffset1);
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP2; ik++)
				{
					float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]);
					float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

					simdMM4(v0, b0.x, a0);
					simdMM4(v1, b0.y, a0);
					simdMM4(v2, b0.z, a0);
					simdMM4(v3, b0.w, a0);
					simdMM4(v4, b1.x, a0);
					simdMM4(v5, b1.y, a0);
					simdMM4(v6, b1.z, a0);
					simdMM4(v7, b1.w, a0);
				}
				buf ^= 1;

				//load 4 elem from X
				int Xic = (((oic - (tx >= STEP)) << LB >> 1) + tx) << 1;
				float2 x0 = (lx0 ? *(float2*)(X + X0 + Xic) : F32_2_0);
				float2 x1 = (lx1 ? *(float2*)(X + X1 + Xic) : F32_2_0);
				float2 x2 = (lx2 ? *(float2*)(X + X2 + Xic) : F32_2_0);
				float2 x3 = (lx3 ? *(float2*)(X + X3 + Xic) : F32_2_0);
				Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
				Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

				//load 2 elem from W
				int Wic = (((oic << LB) + ty) >> 1) << 1;
				int woffset0 = Wic * OC, woffset1 = woffset0 + OC;
				Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + woffset0);
				Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(CW + woffset1);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP2; ik++)
			{
				float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]);
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

				simdMM4(v0, b0.x, a0);
				simdMM4(v1, b0.y, a0);
				simdMM4(v2, b0.z, a0);
				simdMM4(v3, b0.w, a0);
				simdMM4(v4, b1.x, a0);
				simdMM4(v5, b1.y, a0);
				simdMM4(v6, b1.z, a0);
				simdMM4(v7, b1.w, a0);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0; //oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4;
	*(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6;
	*(float4*)(Y + j7) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_PURE_UKERNEL_4_4R
#define CONV_3D_PURE_UKERNEL_4_4R

//LB = 4: Size = 1.125, Time = 2.42465 msec, Performace = 996.399 GFlop/s
//LB = 3: Size = 1.125, Time = 2.61062 msec, Performace = 925.421 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dPure_uernel_4_4R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB][(2 << LB) + 2];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty & 1) << 1);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);

	const int Ws_y = (ty >> 1) << 1, Ws_x = (tx << 1) + (ty & 1);
	const int Xs_x = (tx >> 1) << 1, Xs_y = (ty << 1) + (tx & 1);
	const int OIC = (IC >> LB), SX = (IW - FW)*IC, SCW = IC * OC;

	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, CW += SCW)
		{
			//load 2 elem from X
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			int Xic = (tx >> 1) << 1;
			float2 x0 = (lx0 ? *(float2*)(X + X0 + Xic) : F32_2_0);
			float2 x1 = (lx1 ? *(float2*)(X + X1 + Xic) : F32_2_0);
			Xs[buf][Xs_x][Xs_y] = float2{ x0.x, x1.x };
			Xs[buf][Xs_x + 1][Xs_y] = float2{ x0.y, x1.y };

			//load 2 elem from W
			int Wic = (ty >> 1) << 1;
			int woffset0 = Wic * OC, woffset1 = woffset0 + OC;
			Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + woffset0);
			Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(CW + woffset1);
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP2; ik++)
				{
					float4 a = *(float4*)(&Ws[buf][ik][tx << 1]);
					float4 b = *(float4*)(&Xs[buf][ik][ty << 1]);

					simdMM4(v0, b.x, a);
					simdMM4(v1, b.y, a);
					simdMM4(v2, b.z, a);
					simdMM4(v3, b.w, a);
				}
				buf ^= 1;

				//load 2 elem from X
				int Xic = (((oic << LB) + tx) >> 1) << 1;
				float2 x0 = (lx0 ? *(float2*)(X + X0 + Xic) : F32_2_0);
				float2 x1 = (lx1 ? *(float2*)(X + X1 + Xic) : F32_2_0);
				Xs[buf][Xs_x][Xs_y] = float2{ x0.x, x1.x };
				Xs[buf][Xs_x + 1][Xs_y] = float2{ x0.y, x1.y };

				//load 2 elem from W
				int Wic = (((oic << LB) + ty) >> 1) << 1;
				int woffset0 = Wic * OC, woffset1 = woffset0 + OC;
				Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + woffset0);
				Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(CW + woffset1);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP2; ik++)
			{
				float4 a = *(float4*)(&Ws[buf][ik][tx << 1]);
				float4 b = *(float4*)(&Xs[buf][ik][ty << 1]);

				simdMM4(v0, b.x, a);
				simdMM4(v1, b.y, a);
				simdMM4(v2, b.z, a);
				simdMM4(v3, b.w, a);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0; //oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
}

#endif


//======[IC is power of 2]====================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R_IC_2POW
#define CONV_3D_GEMM_UERNEL_8_8R_IC_2POW

//LB = 4: Size = 1, Time = 1.48134 msec, Performace = 1449.69 GFlop/s
//LB = 3: Size = 1, Time = 1.55845 msec, Performace = 1377.96 GFlop/s
//LB = 4: Size = 1.125, Time = 1.61774 msec, Performace = 1493.39 GFlop/s
//LB = 3: Size = 1.125, Time = 1.76776 msec, Performace = 1366.66 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8R_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3) << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	int xoffset = (X_fh << LIC)*IW + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
		int xoffset = (X_fh << LIC)*IW + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, (OH, OW) % 4 == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4_IC_2POW
#define CONV_3D_GEMM_UERNEL_8_8R4_IC_2POW

//LB = 4: Size = 1, Time = 1.43693 msec, Performace = 1494.49 GFlop/s
//LB = 3: Size = 1, Time = 1.5329  msec, Performace = 1400.93 GFlop/s
//LB = 4: Size = 1.125, Time = 1.60382 msec, Performace = 1506.36 GFlop/s
//LB = 3: Size = 1.125, Time = 1.74764 msec, Performace = 1382.39 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8R4_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X += X1;
	const int sw_IC = sw << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	int xoffset = (X_fh << LIC)*IW + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
		int xoffset = (X_fh << LIC)*IW + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//======[FH = FW = 3]=========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RW3
#define CONV_3D_GEMM_UERNEL_8_8RW3

//LB = 4: Size = 1.125, Time = 1.63056 msec, Performace = 1481.65 GFlop/s
//LB = 3: Size = 1.125, Time = 1.76776 msec, Performace = 1366.66 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8RW3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0, x1, x2, x3;
		x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RW3_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8RW3_IC2POW

//LB = 4: Size = 1.125, Time = 1.59093 msec, Performace = 1518.56 GFlop/s
//LB = 3: Size = 1.125, Time = 1.77885 msec, Performace = 1358.13 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8RW3_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3) << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//------[(OH, OW) % 4 == 0]-----------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0 
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4W3
#define CONV_3D_GEMM_UERNEL_8_8R4W3

//LB = 4: Size = 1.125, Time = 1.5932  msec, Performace = 1516.39 GFlop/s
//LB = 3: Size = 1.125, Time = 1.75219 msec, Performace = 1378.8 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8R4W3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) * IC;//X += X1;
	const int sw_IC = sw * IC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = (X_fh*IW + X_fw - Idx)*IC + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int woffset = (ty - ((ty >= STEP) << LB >> 1) << 1) * OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int woffset = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1) * OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = (X_fh*IW + X_fw - Idx)*IC + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4W3_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8R4W3_IC2POW

//LB = 4: Size = 1.125, Time = 1.56123 msec, Performace = 1547.44 GFlop/s
//LB = 3: Size = 1.125, Time = 1.68134 msec, Performace = 1436.9 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8R4W3_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X += X1;
	const int sw_IC = sw << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	int woffset = (ty - ((ty >= STEP) << LB >> 1) << 1) * OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int woffset = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1) * OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//======[FH = FW = 5]=========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RW5
#define CONV_3D_GEMM_UERNEL_8_8RW5

//LB = 4: Size = 1.5625, Time = 2.20161 msec, Performace = 1524.08 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.36722 msec, Performace = 1417.46 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8RW5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//prepare for GK = FH * FW * IC
	const int GK = 25 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0, x1, x2, x3;
		x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RW5_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8RW5_IC2POW

//LB = 4: Size = 1.5625, Time = 2.15232 msec, Performace = 1558.99 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.31431 msec, Performace = 1449.87 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8RW5_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3) << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 25 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//------[(OH, OW) % 4 == 0]-----------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4W5
#define CONV_3D_GEMM_UERNEL_8_8R4W5

//LB = 4: Size = 1.5625, Time = 2.17827 msec, Performace = 1540.42 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.33045 msec, Performace = 1439.82 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8R4W5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) * IC;//X += X1;
	const int sw_IC = sw * IC;

	//prepare for GK = FH * FW * IC
	const int GK = 25 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4W5_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8R4W5_IC2POW

//LB = 4: Size = 1.5625, Time = 2.1148  msec, Performace = 1586.65 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.29335 msec, Performace = 1463.12 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8R4W5_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X += X1;
	const int sw_IC = sw << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 25 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif

#endif