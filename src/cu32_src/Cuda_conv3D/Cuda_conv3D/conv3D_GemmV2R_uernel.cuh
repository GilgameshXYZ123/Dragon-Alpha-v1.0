#pragma once

#ifndef CONV_3D_GEMMV2_R_UERNEL_H
#define CONV_3D_GEMMV2_R_UERNEL_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
//Version2: for small input feature
#ifndef CONV_3D_GEMMV2_R_UERNEL_CALL
#define CONV_3D_GEMMV2_R_UERNEL_CALL

//======[Common]=============================================================
#define conv3dGemmV2_u88R(stream, LB, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N) \
	conv3dGemmV2_uernel_8_8R<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw,\
			 oc_index, n_index)

//======[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
#define conv3dGemmV2_u88RW3p1_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_uernel_8_8R_W3P1_ic2pow<LB, (1<<LB>>1), (1 << LB)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw,\
			 oc_index, n_index)

//=========[ph = pw = 1], [FH = FW = 4], [IH, IW > 2]===========================
#define conv3dGemmV2_u88RW4p1_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_uernel_8_8R_W4P1_ic2pow<LB, (1<<LB>>1), (1 << LB)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw,\
			 oc_index, n_index)

//=========[ph = pw = 2], [FH = FW = 5], [IH, IW > 2]===========================
#define conv3dGemmV2_u88RW5p2_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_uernel_8_8R_W5P2_ic2pow<LB, (1<<LB>>1), (1 << LB)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw,\
			 oc_index, n_index)

#endif


//======[Common]=============================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMMV2_UERNEL_8_8R
#define CONV_3D_GEMMV2_UERNEL_8_8R

//for(IH, IW) = (4, 4)
//LB = 4: Size = 2.25, Time = 2.34853 msec, Performace = 2057.39 GFlop/s
//LB = 3: Size = 2.25, Time = 2.4387  msec, Performace = 1981.32 GFlop/s
//for(IH, IW) = (8, 8)
//LB = 4: Size = 2.25, Time = 2.61031 msec, Performace = 1851.06 GFlop/s
//LB = 3: Size = 2.25, Time = 3.05422 msec, Performace = 1582.02 GFlop/s
//for(IH, IW) = (16, 16)
//LB = 4: Size = 2.25, Time = 2.73739 msec, Performace = 1765.12 GFlop/s
//LB = 3: Size = 2.25, Time = 3.21971 msec, Performace = 1500.71 GFlop/s
//for(IH, IW) = (32, 32)
//LB = 4: Size = 2.25, Time = 2.9829  msec, Performace = 1619.84 GFlop/s
//LB = 3: Size = 2.25, Time = 3.13191 msec, Performace = 1542.78 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemmV2_uernel_8_8R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - ph, tow = ow * sw - pw;

	//prepare for GK = FH * FW * IC
	int fhs = -IF_int((toh < 0), toh, 0);
	int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_IC = tFW * IC, GK = tFH * tFW_IC;
	CW += (fhs*FW + fws)*IC*OC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws)*IC;//X[0, fhs, fws, 0]

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	const int Xstride = IH * IW * IC;
	const int X1 = X0 + Xstride;
	const int X2 = X1 + Xstride;
	const int X3 = X2 + Xstride;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	const int SX = (IW - tFW)*IC;
	int fh = X_k / tFW_IC;//X_fh = W_fh = fh
	int xoffset = fh * SX + X_k;
	float2 x0 = *(float2*)(X + X0 + xoffset);
	float2 x1 = *(float2*)(X + X1 + xoffset);
	float2 x2 = *(float2*)(X + X2 + xoffset);
	float2 x3 = *(float2*)(X + X3 + xoffset);
	Xs[buf][(tx << 1)    ][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	const int SW = (FW - tFW)*IC;
	int woffset0 = (fh*SW + W_k)*OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
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

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int fh = X_k / tFW_IC;//X_fh = W_fh = fh
		int xoffset = fh * SX + X_k;
		float2 x0 = *(float2*)(X + X0 + xoffset);
		float2 x1 = *(float2*)(X + X1 + xoffset);
		float2 x2 = *(float2*)(X + X2 + xoffset);
		float2 x3 = *(float2*)(X + X3 + xoffset);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = (fh*SW + W_k)*OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
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

	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//======[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMMV2_UERNEL_8_8R_W3_P1_IC2POW
#define CONV_3D_GEMMV2_UERNEL_8_8R_W3_P1_IC2POW

//for(IH, IW) = (4, 4)
//LB = 4: Size = 2.25, Time = 2.30918 msec, Performace = 2092.45 GFlop/s
//LB = 3: Size = 2.25, Time = 2.41398 msec, Performace = 2001.61 GFlop/s
//for(IH, IW) = (8, 8)
//LB = 4: Size = 2.25, Time = 2.56837 msec, Performace = 1881.28 GFlop/s
//LB = 3: Size = 2.25, Time = 2.99055 msec, Performace = 1615.7  GFlop/s
//for(IH, IW) = (16, 16)
//LB = 4: Size = 2.25, Time = 2.69059 msec, Performace = 1795.83 GFlop/s
//LB = 3: Size = 2.25, Time = 3.15798 msec, Performace = 1530.04 GFlop/s
//for(IH, IW) = (32, 32)
//LB = 4: Size = 2.25, Time = 2.93489 msec, Performace = 1646.35 GFlop/s
//LB = 3: Size = 2.25, Time = 3.08337 msec, Performace = 1567.06 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemmV2_uernel_8_8R_W3P1_ic2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = 3, 2
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - 1, tow = ow * sw - 1;

	//prepare for GK = FH * FW * IC
	int fhs = -IF_int((toh < 0), toh, 0);
	int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	CW += ((fhs * 3 + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]

	int fh_idx = (tFH == 3), fw_idx = (tFW == 3);
	int fhw_offset = ((fh_idx << 1) + fw_idx) * 9;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	const int Xstride = (IH * IW) << LIC;
	const int X1 = X0 + Xstride;
	const int X2 = X1 + Xstride;
	const int X3 = X2 + Xstride;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	const int SX = (IW - tFW) << LIC;
	int Idx = X_k >> LIC; int fh = XIDX_V2_W3P1[fhw_offset + Idx] >> 2;
	int xoffset = fh * SX + X_k;
	float2 x0 = *(float2*)(X + X0 + xoffset);
	float2 x1 = *(float2*)(X + X1 + xoffset);
	float2 x2 = *(float2*)(X + X2 + xoffset);
	float2 x3 = *(float2*)(X + X3 + xoffset);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	const int SW = (3 - tFW) << LIC;
	int woffset0 = (fh*SW + W_k)*OC, woffset1 = woffset0 + OC;
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
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; int fh = XIDX_V2_W3P1[fhw_offset + Idx] >> 2;
		int xoffset = fh * SX + X_k;
		float2 x0 = *(float2*)(X + X0 + xoffset);
		float2 x1 = *(float2*)(X + X1 + xoffset);
		float2 x2 = *(float2*)(X + X2 + xoffset);
		float2 x3 = *(float2*)(X + X3 + xoffset);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = (fh*SW + W_k)*OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
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

	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//=========[ph = pw = 1], [FH = FW = 4], [IH, IW > 2]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMMV2_UERNEL_8_8R_W4_P1_IC2POW
#define CONV_3D_GEMMV2_UERNEL_8_8R_W4_P1_IC2POW

//for(IH, IW) = (4, 4)
//LB = 4: Size = 4, Time = 3.36004 msec, Performace = 2556.5 GFlop/s
//LB = 3: Size = 4, Time = 3.48981 msec, Performace = 2461.43 GFlop/s
//for(IH, IW) = (8, 8)
//LB = 4: Size = 4, Time = 3.99182 msec, Performace = 2151.89 GFlop/s
//LB = 3: Size = 4, Time = 4.8414  msec, Performace = 1774.27 GFlop/s
//for(IH, IW) = (16, 16)
//LB = 4: Size = 4, Time = 4.45306 msec, Performace = 1929    GFlop/s
//LB = 3: Size = 4, Time = 5.47999 msec, Performace = 1567.51 GFlop/s
//for(IH, IW) = (32, 32)
//LB = 4: Size = 4, Time = 4.86731 msec, Performace = 1764.82 GFlop/s
//LB = 3: Size = 4, Time = 5.24569 msec, Performace = 1637.52 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemmV2_uernel_8_8R_W4P1_ic2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = 4, 3
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - 1, tow = ow * sw - 1;

	//prepare for GK = FH * FW * IC
	int fhs = -IF_int((toh < 0), toh, 0);
	int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 4), 4, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 4), 4, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	CW += (((fhs << 2) + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]

	int fh_idx = (tFH == 4), fw_idx = (tFW == 4);
	int fhw_offset = ((fh_idx << 1) + fw_idx) << 4;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	const int Xstride = (IH * IW) << LIC;
	const int X1 = X0 + Xstride;
	const int X2 = X1 + Xstride;
	const int X3 = X2 + Xstride;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	const int SX = (IW - tFW) << LIC;
	int Idx = X_k >> LIC; int fh = XIDX_V2_W4P1[fhw_offset + Idx] >> 2;
	int xoffset = fh * SX + X_k;
	float2 x0 = *(float2*)(X + X0 + xoffset);
	float2 x1 = *(float2*)(X + X1 + xoffset);
	float2 x2 = *(float2*)(X + X2 + xoffset);
	float2 x3 = *(float2*)(X + X3 + xoffset);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	const int SW = (4 - tFW) << LIC;
	int woffset0 = (fh*SW + W_k)*OC , woffset1 = woffset0 + OC;
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
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

			simdMM4(v0, b0.x, a0);  simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0);  simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0);  simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0);  simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0);  simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; int fh = XIDX_V2_W4P1[fhw_offset + Idx] >> 2;
		int xoffset = fh * SX + X_k;
		float2 x0 = *(float2*)(X + X0 + xoffset);
		float2 x1 = *(float2*)(X + X1 + xoffset);
		float2 x2 = *(float2*)(X + X2 + xoffset);
		float2 x3 = *(float2*)(X + X3 + xoffset);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = (fh*SW + W_k)*OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0);  simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0);  simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0);  simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0);  simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0);  simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//=========[ph = pw = 2], [FH = FW = 5], [IH, IW > 2]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMMV2_UERNEL_8_8R_W5_P2_IC2POW
#define CONV_3D_GEMMV2_UERNEL_8_8R_W5_P1_IC2POW

//for(IH, IW) = (4, 4)
//LB = 4: Size = 3.125, Time = 3.01514 msec, Performace = 2225.73 GFlop/s
//LB = 3: Size = 3.125, Time = 3.02448 msec, Performace = 2218.86 GFlop/s
//for(IH, IW) = (8, 8)
//LB = 4: Size = 3.125, Time = 3.13646 msec, Performace = 2139.64 GFlop/s
//LB = 3: Size = 3.125, Time = 3.86114 msec, Performace = 1738.06 GFlop/s
//for(IH, IW) = (16, 16)
//LB = 4: Size = 3.125, Time = 3.51285 msec, Performace = 1910.38 GFlop/s
//LB = 3: Size = 3.125, Time = 4.263   msec, Performace = 1574.22 GFlop/s
//for(IH, IW) = (32, 32)
//LB = 4: Size = 3.125, Time = 3.76817 msec, Performace = 1780.94 GFlop/s
//LB = 3: Size = 3.125, Time = 4.06208 msec, Performace = 1652.08 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemmV2_uernel_8_8R_W5P2_ic2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = 5, 4, 3
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 2
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - 2, tow = ow * sw - 2;

	//prepare for GK = FH * FW * IC
	int fhs = -IF_int((toh < 0), toh, 0);
	int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 5), 5, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 5), 5, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	CW += ((fhs * 5 + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]

	int fh_idx = (tFH == 4) + ((tFH == 5) << 1);
	int fw_idx = (tFW == 4) + ((tFW == 5) << 1);
	int fhw_offset = (fh_idx * 3 + fw_idx) * 25;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	const int Xstride = (IH * IW) << LIC;
	const int X1 = X0 + Xstride;
	const int X2 = X1 + Xstride;
	const int X3 = X2 + Xstride;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	const int SX = (IW - tFW) << LIC;
	int Idx = X_k >> LIC; int fh = XIDX_V2_W5P2[fhw_offset + Idx] >> 3;
	int xoffset = fh * SX + X_k;
	float2 x0 = *(float2*)(X + X0 + xoffset);
	float2 x1 = *(float2*)(X + X1 + xoffset);
	float2 x2 = *(float2*)(X + X2 + xoffset);
	float2 x3 = *(float2*)(X + X3 + xoffset);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	const int SW = (5 - tFW) << LIC;
	int woffset0 = (fh*SW + W_k)*OC, woffset1 = woffset0 + OC;
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
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; int fh = XIDX_V2_W5P2[fhw_offset + Idx] >> 3;
		int xoffset = fh * SX + X_k;
		float2 x0 = *(float2*)(X + X0 + xoffset);
		float2 x1 = *(float2*)(X + X1 + xoffset);
		float2 x2 = *(float2*)(X + X2 + xoffset);
		float2 x3 = *(float2*)(X + X3 + xoffset);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = (fh*SW + W_k)*OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
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

	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif

#endif