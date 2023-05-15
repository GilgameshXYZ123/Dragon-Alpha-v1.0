#pragma once

#ifndef CONV_3D_GEMMV2_KERNEL_H
#define CONV_3D_GEMMV2_KERNEL_H

//Version2: for small input feature
#ifndef CONV_3D_GEMMV2_KERNEL_CALL
#define CONV_3D_GEMMV2_KERNEL_CALL

//LB = log2(BLOCK_SIZE)
//j_index = n_index*OH_OW

//=========[Common]=============================================================
#define conv3dGemmV2_k88(stream, LB, oc_index, n_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N) \
	conv3dGemmV2_kernel_8_8<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw,\
			 oc_index, n_index)

//=========[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
#define conv3dGemmV2_k88W3p1_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_kernel_8_8_W3P1_ic2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, \
			 oc_index, n_index)

//=========[ph = pw = 1], [FH = FW = 4], [IH, IW > 2]===========================
#define conv3dGemmV2_k88W4p1_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_kernel_8_8_W4P1_ic2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, \
			 oc_index, n_index)

//=========[ph = pw = 2], [FH = FW = 5], [IH, IW > 2]===========================
#define conv3dGemmV2_k88W5p2_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_kernel_8_8_W5P2_ic2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, \
			 oc_index, n_index)

#endif


//=========[Common]=============================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_8_8
#define CONV_3D_GEMMV2_KERNEL_8_8

//for(IH, IW) = (4, 4)
//LB = 4: Size = 2.25, Time = 2.59046 msec, Performace = 1865.24  GFlop/s
//LB = 3: Size = 2.25, Time = 5.0839  msec, Performace =  950.419 GFlop/s
//for(IH, IW) = (8, 8)
//k88x4_ic2pow<4>: Size = 2.25, Time = 3.38981 msec, Performace = 1425.4   GFlop/s
//k88x4_ic2pow<3>: Size = 2.25, Time = 5.53039 msec, Performace =  873.688 GFlop/s
//LB = 4: Size = 2.25, Time = 3.2823 msec, Performace = 1472.09  GFlop/s
//LB = 3: Size = 2.25, Time = 7.3949 msec, Performace = 653.401 GFlop/s
//for(IH, IW) = (16, 16)
//k88x4_ic2pow<4>: Size = 2.25, Time = 3.38981 msec, Performace = 1425.4   GFlop/s
//k88x4_ic2pow<3>: Size = 2.25, Time = 5.53039 msec, Performace =  873.688 GFlop/s
//LB = 4: Size = 2.25, Time = 3.66376 msec, Performace = 1318.82 GFlop/s
//LB = 3: Size = 2.25, Time = 3.56857 msec, Performace = 1354    GFlop/s
//LB = 3: Size = 2.25, Time = 3.39598 msec, Performace = 1422.81 GFlop/s
//for(IH, IW) = (32, 32)
//k88x4_ic2pow_tex<4>: Size = 2.25, Time = 3.41826 msec, Performace = 1413.54 GFlop/s
//k88x4_ic2pow_tex<3>: Size = 2.25, Time = 4.01357 msec, Performace = 1203.87 GFlop/s
//LB = 4: Size = 2.25, Time = 3.32787 msec, Performace = 1451.93 GFlop/s
//LB = 3: Size = 2.25, Time = 4.66655 msec, Performace = 1035.42 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_8_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

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
	W += (fhs*FW + fws)*IC;//W[0, fhs, fws, 0]
	X += (fhs*IW + fws)*IC;//X[0, fhs, fws, 0]

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	const int Wstride = FH * FW * IC;
	const int toc0 = (oc0 + ((ty >= STEP) << 2)) * Wstride;
	const int toc1 = toc0 + Wstride;
	const int toc2 = toc1 + Wstride;
	const int toc3 = toc2 + Wstride;

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int fh = W_k / tFW_IC;
	const int SW = (FW - tFW)*IC;
	float4 wv; int woffset = fh * SW + W_k;//X_fh = W_fh = fh
	wv.x = W[toc0 + woffset];
	wv.y = W[toc1 + woffset];
	wv.z = W[toc2 + woffset];
	wv.w = W[toc3 + woffset];
	Ws[buf][ty][tx] = wv;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	const int SX = (IW - tFW)*IC;
	float4 xv; int xoffset = fh * SX + X_k;
	xv.x = X[X0 + xoffset];
	xv.y = X[X1 + xoffset];
	xv.z = X[X2 + xoffset];
	xv.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = xv;
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int fh = W_k / tFW_IC;
		float4 wv; int woffset = fh * SW + W_k;//X_fh = W_fh = fh
		wv.x = W[toc0 + woffset];
		wv.y = W[toc1 + woffset];
		wv.z = W[toc2 + woffset];
		wv.w = W[toc3 + woffset];
		Ws[buf][ty][tx] = wv;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		float4 xv; int xoffset = fh * SX + X_k;
		xv.x = X[X0 + xoffset];
		xv.y = X[X1 + xoffset];
		xv.z = X[X2 + xoffset];
		xv.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = xv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

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


//=========[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_8_8_W3_P1_IC2POW
#define CONV_3D_GEMMV2_KERNEL_8_8_W3_P1_IC2POW

//for(IH, IW) = (4, 4)
//k88W3x4_ic2pow_tex<4>: Size = 2.25, Time = 3.15114 msec, Performace = 1533.36 GFlop/s
//k88W3x4_ic2pow_tex<3>: Size = 2.25, Time = 4.07084 msec, Performace = 1186.94 GFlop/s
//LB = 4: Size = 2.25, Time = 2.48455 msec, Performace = 1944.76 GFlop/s
//LB = 3:Size = 2.25, Time = 5.41242 msec, Performace = 892.732 GFlop/s
//for(IH, IW) = (8, 8)
//k88W3x4_ic2pow_tex<4>: Size = 2.25, Time = 3.14779 msec, Performace = 1535    GFlop/s
//k88W3x4_ic2pow_tex<3>: Size = 2.25, Time = 5.31711 msec, Performace = 908.734 GFlop/s
//LB = 4: Size = 2.25, Time = 3.13859 msec, Performace = 1539.49  GFlop/s
//LB = 3: Size = 2.25, Time = 5.81417 msec, Performace =  831.046 GFlop/s
//for(IH, IW) = (16, 16)
//k88W3x4_ic2pow_tex<4>: Size = 2.25, Time = 3.1358  msec, Performace = 1540.86 GFlop/s
//k88W3x4_ic2pow_tex<3>: Size = 2.25, Time = 3.33038 msec, Performace = 1450.84 GFlop/s
//LB = 4: Size = 2.25, Time = 3.2402  msec, Performace = 1491.22 GFlop/s
//LB = 3: Size = 2.25, Time = 5.54872 msec, Performace = 870.802 GFlop/s
//for(IH, IW) = (32, 32)
//k88W3x4_ic2pow_tex<4>: Size = 2.25, Time = 3.21539 msec, Performace = 1502.72 GFlop/s
//k88W3x4_ic2pow_tex<3>: Size = 2.25, Time = 3.89612 msec, Performace = 1240.17 GFlop/s
//LB = 4: Size = 2.25, Time = 3.2402  msec, Performace = 1491.22 GFlop/s
//LB = 3: Size = 2.25, Time = 4.61641 msec, Performace = 1046.67 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_8_8_W3P1_ic2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//tFH = tFW = 3, 2
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

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
	W += (fhs * 3 + fws) << LIC;//W[0, fhs, fws, 0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]

	int fh_idx = (tFH == 3), fw_idx = (tFW == 3);
	int fhw_offset = ((fh_idx << 1) + fw_idx) * 9;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int Wstride = 9 << LIC;
	int toc0 = (oc0 + ((ty >= STEP) << 2)) * Wstride;
	int toc1 = toc0 + Wstride, toc2 = toc1 + Wstride, toc3 = toc2 + Wstride;

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	int Xstride = (IH * IW) << LIC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int Idx = W_k >> LIC; int fh = XIDX_V2_W3P1[fhw_offset + Idx] >> 2;
	const int SW = (3 - tFW) << LIC;
	float4 w; int woffset = fh * SW + W_k;//X_fh = W_fh = fh
	w.x = W[toc0 + woffset];
	w.y = W[toc1 + woffset];
	w.z = W[toc2 + woffset];
	w.w = W[toc3 + woffset];
	Ws[buf][ty][tx] = w;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	const int SX = (IW - tFW) << LIC;
	float4 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int Idx = W_k >> LIC; int fh = XIDX_V2_W3P1[fhw_offset + Idx] >> 2;
		float4 w; int woffset = fh * SW + W_k;//X_fh = W_fh = fh
		w.x = W[toc0 + woffset];
		w.y = W[toc1 + woffset];
		w.z = W[toc2 + woffset];
		w.w = W[toc3 + woffset];
		Ws[buf][ty][tx] = w;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		float4 x; int xoffset = fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

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
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_8_8_W4_P1_IC2POW
#define CONV_3D_GEMMV2_KERNEL_8_8_W4_P1_IC2POW

//for(IH, IW) = (4, 4)
//LB = 4: Size = 4, Time = 3.2072  msec, Performace = 2678.32 GFlop/s
//LB = 3: Size = 4, Time = 6.58696 msec, Performace = 1304.08 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_8_8_W4P1_ic2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//tFH = tFW = 4, 3
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

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
	W += ((fhs << 2) + fws) << LIC;//W[0, fhs, fws, 0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]

	int fh_idx = (tFH == 4), fw_idx = (tFW == 4);
	int fhw_offset = ((fh_idx << 1) + fw_idx) * 16;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int Wstride = 16 << LIC;
	int toc0 = (oc0 + ((ty >= STEP) << 2)) * Wstride;
	int toc1 = toc0 + Wstride, toc2 = toc1 + Wstride, toc3 = toc2 + Wstride;

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	int Xstride = (IH * IW) << LIC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int Idx = W_k >> LIC; int fh = XIDX_V2_W4P1[fhw_offset + Idx] >> 2;
	const int SW = (4 - tFW) << LIC;
	float4 w; int woffset = fh * SW + W_k;//X_fh = W_fh = fh
	w.x = W[toc0 + woffset];
	w.y = W[toc1 + woffset];
	w.z = W[toc2 + woffset];
	w.w = W[toc3 + woffset];
	Ws[buf][ty][tx] = w;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	const int SX = (IW - tFW) << LIC;
	float4 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int Idx = W_k >> LIC; int fh = XIDX_V2_W4P1[fhw_offset + Idx] >> 2;
		float4 w; int woffset = fh * SW + W_k;//X_fh = W_fh = fh
		w.x = W[toc0 + woffset];
		w.y = W[toc1 + woffset];
		w.z = W[toc2 + woffset];
		w.w = W[toc3 + woffset];
		Ws[buf][ty][tx] = w;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		float4 x; int xoffset = fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

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
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_8_8_W5_P2_IC2POW
#define CONV_3D_GEMMV2_KERNEL_8_8_W5_P2_IC2POW

//for(IH, IW) = (4, 4)
//LB = 4: Size = 3.125, Time = 2.73673 msec, Performace = 2452.16 GFlop/s
//LB = 3: Size = 3.125, Time = 5.61345 msec, Performace = 1195.5 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_8_8_W5P2_ic2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//tFH = tFW = 5, 4, 3
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

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
	const int tFW_IC = tFW << LIC, GK = tFH * tFW_IC;
	W += (fhs * 5 + fws) << LIC;//W[0, fhs, fws, 0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]

	int fh_idx = (tFH == 4) + ((tFH == 5) << 1);
	int fw_idx = (tFW == 4) + ((tFW == 5) << 1);
	int fhw_offset = (fh_idx * 3 + fw_idx) * 25;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int Wstride = 25 << LIC;
	int toc0 = (oc0 + ((ty >= STEP) << 2)) * Wstride;
	int toc1 = toc0 + Wstride, toc2 = toc1 + Wstride, toc3 = toc2 + Wstride;

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	int Xstride = (IH * IW) << LIC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int Idx = W_k >> LIC; int fh = XIDX_V2_W5P2[fhw_offset + Idx] >> 3;
	const int SW = (5 - tFW) << LIC;
	float4 w; int woffset = fh * SW + W_k;//X_fh = W_fh = fh
	w.x = W[toc0 + woffset];
	w.y = W[toc1 + woffset];
	w.z = W[toc2 + woffset];
	w.w = W[toc3 + woffset];
	Ws[buf][ty][tx] = w;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	const int SX = (IW - tFW) << LIC;
	float4 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int Idx = W_k >> LIC; int fh = XIDX_V2_W5P2[fhw_offset + Idx] >> 3;
		float4 w; int woffset = fh * SW + W_k;//X_fh = W_fh = fh
		w.x = W[toc0 + woffset];
		w.y = W[toc1 + woffset];
		w.z = W[toc2 + woffset];
		w.w = W[toc3 + woffset];
		Ws[buf][ty][tx] = w;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		float4 x; int xoffset = fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

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