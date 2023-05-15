#pragma once

#ifndef CONV_3D_GEMMV2_R_KERNEL_EX_H
#define CONV_3D_GEMMV2_R_KERNEL_EX_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
//Version2: for small input feature
#ifndef CONV_3D_GEMMV2_R_KERNEL_EX_CALL
#define CONV_3D_GEMMV2_R_KERNEL_EX_CALL

//LB = log2(BLOCK_SIZE)
//j_index = n_index*OH_OW

//=========[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
#define conv3dGemmV2_k88RW3p1_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_kernel_8_8R_W3P1_ic2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw,\
			 oc_index, n_index)

//=========[ph = pw = 1], [FH = FW = 4], [IH, IW > 2]===========================
#define conv3dGemmV2_k88RW4p1_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_kernel_8_8R_W4P1_ic2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw,\
			 oc_index, n_index)

//=========[ph = pw = 2], [FH = FW = 5], [IH, IW > 2]===========================
#define conv3dGemmV2_k88RW5p2_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_kernel_8_8R_W5P2_ic2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw,\
			 oc_index, n_index)

#endif


//=========[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_8_8R_W3_P1_IC2POW
#define CONV_3D_GEMMV2_KERNEL_8_8R_W3_P1_IC2POW

//for(IH, IW) = (4, 4)
//LB = 4: Size = 2.25, Time = 2.32005 msec, Performace = 2082.65 GFlop/s
//LB = 3: Size = 2.25, Time = 2.64427 msec, Performace = 1827.29 GFlop/s
//for(IH, IW) = (8, 8)
//LB = 4: Size = 2.25, Time = 2.67653 msec, Performace = 1805.26 GFlop/s
//LB = 3: Size = 2.25, Time = 3.24297 msec, Performace = 1489.94 GFlop/s
//for(IH, IW) = (16, 16)
//LB = 4: Size = 2.25, Time = 2.84524 msec, Performace = 1698.22 GFlop/s
//LB = 3: Size = 2.25, Time = 3.4676  msec, Performace = 1393.43 GFlop/s
//for(IH, IW) = (32, 32)
//LB = 4: Size = 2.25, Time = 3.05029 msec, Performace = 1584.06 GFlop/s
//LB = 3: Size = 2.25, Time = 3.40108 msec, Performace = 1420.68 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_8_8R_W3P1_ic2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = 3, 2
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
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
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	const int SX = (IW - tFW) << LIC;
	int Idx = X_k >> LIC; int fh = XIDX_V2_W3P1[fhw_offset + Idx] >> 2;
	float4 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	const int SW = (3 - tFW) << LIC;
	int woffset = (fh*SW + W_k)*OC;//X_fh = W_fh = fh
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k >> LIC; int fh = XIDX_V2_W3P1[fhw_offset + Idx] >> 2;
		float4 x; int xoffset = fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = (fh*SW + W_k)*OC;//X_fh = W_fh = fh
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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
#ifndef CONV_3D_GEMMV2_KERNEL_8_8R_W4_P1_IC2POW
#define CONV_3D_GEMMV2_KERNEL_8_8R_W4_P1_IC2POW

//for(IH, IW) = (4, 4)
//LB = 4: Size = 4, Time = 3.31585 msec, Performace = 2590.57 GFlop/s
//LB = 3: Size = 4, Time = 3.79039 msec, Performace = 2266.24 GFlop/s
//for(IH, IW) = (8, 8)
//LB = 4: Size = 4, Time = 4.15965 msec, Performace = 2065.06 GFlop/s
//LB = 3: Size = 4, Time = 5.25373 msec, Performace = 1635.02 GFlop/s
//for(IH, IW) = (16, 16)
//LB = 4: Size = 4, Time = 4.72585 msec, Performace = 1817.65 GFlop/s
//LB = 3: Size = 4, Time = 5.88097 msec, Performace = 1460.63 GFlop/s
//for(IH, IW) = (32, 32)
//LB = 4: Size = 4, Time = 5.02016 msec, Performace = 1711.09 GFlop/s
//LB = 3: Size = 4, Time = 5.83764 msec, Performace = 1471.47 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_8_8R_W4P1_ic2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = 4, 3
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
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
	CW += ((fhs * 4 + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]

	int fh_idx = (tFH == 4), fw_idx = (tFW == 4);
	int fhw_offset = ((fh_idx << 1) + fw_idx) * 16;

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
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	const int SX = (IW - tFW) << LIC;
	int Idx = X_k >> LIC; int fh = XIDX_V2_W4P1[fhw_offset + Idx] >> 2;
	float4 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	const int SW = (4 - tFW) << LIC;
	int woffset = (fh*SW + W_k)*OC;//X_fh = W_fh = fh
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k >> LIC; int fh = XIDX_V2_W4P1[fhw_offset + Idx] >> 2;
		float4 x; int xoffset = fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = (fh*SW + W_k)*OC;//X_fh = W_fh = fh
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_8_8R_W5_P2_IC2POW
#define CONV_3D_GEMMV2_KERNEL_8_8R_W5_P2_IC2POW

//for(IH, IW) = (4, 4)
//LB = 4: Size = 3.125, Time = 2.99489 msec, Performace = 2240.78 GFlop/s
//LB = 3: Size = 3.125, Time = 3.27543 msec, Performace = 2048.86 GFlop/s
//for(IH, IW) = (8, 8)
//LB = 4: Size = 3.125, Time = 3.23081 msec, Performace = 2077.15 GFlop/s
//LB = 3: Size = 3.125, Time = 4.1503 msec, Performace = 1616.96 GFlop/s
//for(IH, IW) = (16, 16)
//LB = 4: Size = 3.125, Time = 3.68757 msec, Performace = 1819.87 GFlop/s
//LB = 3: Size = 3.125, Time = 4.6189  msec, Performace = 1452.92 GFlop/s
//for(IH, IW) = (32, 32)
//LB = 4: Size = 3.125, Time = 3.87659 msec, Performace = 1731.13 GFlop/s
//LB = 3: Size = 3.125, Time = 4.48821 msec, Performace = 1495.23 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_8_8R_W5P2_ic2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = 5, 4, 3
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 2
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
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	const int SX = (IW - tFW) << LIC;
	int Idx = X_k >> LIC; int fh = XIDX_V2_W5P2[fhw_offset + Idx] >> 3;
	float4 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	const int SW = (5 - tFW) << LIC;
	int woffset = (fh*SW + W_k)*OC;//X_fh = W_fh = fh
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k >> LIC; int fh = XIDX_V2_W5P2[fhw_offset + Idx] >> 3;
		float4 x; int xoffset = fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = (fh*SW + W_k)*OC;//X_fh = W_fh = fh
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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

#endif