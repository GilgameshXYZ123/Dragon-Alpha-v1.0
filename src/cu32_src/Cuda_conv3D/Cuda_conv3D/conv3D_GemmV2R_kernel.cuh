#pragma once

#ifndef CONV_3D_GEMMV2_R_KERNEL_H
#define CONV_3D_GEMMV2_R_KERNEL_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
//Version2: for small input feature
#ifndef CONV_3D_GEMMV2_R_KERNEL_CALL
#define CONV_3D_GEMMV2_R_KERNEL_CALL

//LB = log2(BLOCK_SIZE)
//j_index = n_index*OH_OW

#define conv3dGemmV2_k88R(stream, LB, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N) \
	conv3dGemmV2_kernel_8_8R<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw,\
			 oc_index, n_index)

#define conv3dGemmV2_k84R(stream, LB, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N) \
	conv3dGemmV2_kernel_8_4R<LB, (1<<LB>>1)>\
		<<< dim3(N>>LB>>2, GN>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw,\
			 oc_index, n_index)

#define conv3dGemmV2_k48R(stream, LB, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N) \
	conv3dGemmV2_kernel_4_8R<LB, (1<<LB>>1)>\
		<<< dim3(N>>LB>>3, GN>>LB>>2, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw,\
			 oc_index, n_index)

#define conv3dGemmV2_k44R(stream, LB, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N) \
	conv3dGemmV2_kernel_4_4R<LB, (1<<LB>>1)>\
		<<< dim3(N>>LB>>2, GN>>LB>>2, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw,\
			 oc_index, n_index)

#define conv3dGemmV2_k42R(stream, LB, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N) \
	conv3dGemmV2_kernel_4_2R<LB, (1<<LB>>1)>\
		<<< dim3(N>>LB>>1, GN>>LB>>2, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw,\
			 oc_index, n_index)

#define conv3dGemmV2_k24R(stream, LB, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N) \
	conv3dGemmV2_kernel_2_4R<LB, (1<<LB>>1)>\
		<<< dim3(N>>LB>>2, GN>>LB>>1, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw,\
			 oc_index, n_index)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_8_8R
#define CONV_3D_GEMMV2_KERNEL_8_8R

//for(IH, IW) = (4, 4)
//LB = 4(3*3): Size = 2.25, Time = 2.37451 msec, Performace = 2034.88 GFlop/s
//LB = 3(3*3): Size = 2.25, Time = 2.67422 msec, Performace = 1806.82 GFlop/s
//LB = 4(4*4): Size = 4, Time = 4.04512 msec, Performace = 2123.53 GFlop/s
//LB = 3(4*4): Size = 4, Time = 4.50474 msec, Performace = 1906.86 GFlop/s
//LB = 4(5*5): Size = 3.125, Time = 2.6623 msec, Performace = 2520.71 GFlop/s
//LB = 3(5*5): Size = 3.125, Time = 2.9509 msec, Performace = 2274.18 GFlop/s
//for(IH, IW) = (8, 8)
//LB = 4: Size = 2.25, Time = 2.82515 msec, Performace = 1710.3  GFlop/s
//LB = 3: Size = 2.25, Time = 3.39598 msec, Performace = 1422.81 GFlop/s
//for(IH, IW) = (16, 16)
//kgemmr88R4_ic2pow<4>: Size = 2.25, Time = 3.63116 msec, Performace = 1330.66 GFlop/s
//LB = 4: Size = 2.25, Time = 2.90842 msec, Performace = 1661.33 GFlop/s
//LB = 3: Size = 2.25, Time = 3.56857 msec, Performace = 1354    GFlop/s
//for(IH, IW) = (32, 32)
//kgemmr88R4_ic2pow<4>: Size = 2.25, Time = 3.20933 msec, Performace = 1505.56 GFlop/s
//LB = 4: Size = 2.25, Time = 3.12613 msec, Performace = 1545.63 GFlop/s
//LB = 3: Size = 2.25, Time = 3.51778 msec, Performace = 1373.55 GFlop/s
//kgemmr88R4_ic2pow<4>: Size = 1.125, Time = 1.67776 msec, Performace = 1439.96 GFlop/s
//LB = 4: Size = 1.125, Time = 1.64524 msec, Performace = 1468.43 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_8_8R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
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
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	const int SX = (IW - tFW)*IC;
	int fh = X_k / tFW_IC; 
	float4 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	const int SW = (FW - tFW)*IC;
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
		int fh = X_k / tFW_IC;
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_8_4R
#define CONV_3D_GEMMV2_KERNEL_8_4R

//for(IH, IW) = (4, 4)
//LB = 4: Size = 2.25, Time = 3.08704 msec, Performace = 1565.2 GFlop/s
//LB = 3: Size = 2.25, Time = 3.98827 msec, Performace = 1211.51 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_8_4R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int n_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

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
	const int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	CW += oc0 + ((tx >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	int n0 = (((blockIdx.x << LB) + tx) << 2) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((ty & 1) << 1);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int X1 = X0 + IH * IW * IC;

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	const int SX = (IW - tFW)*IC;
	int fh = X_k / tFW_IC; 
	float2 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);
	int SW = (FW - tFW)*IC;
	int woffset = (fh*SW + W_k)*OC;
	Ws[buf][tx][ty] = *(float4*)(CW + woffset);
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int fh = X_k / tFW_IC;
		float2 x; int xoffset = fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		Xs[buf][Xs_y][Xs_x] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int woffset = (fh*SW + W_k)*OC;
		Ws[buf][tx][ty] = *(float4*)(CW + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
	}

	int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2; *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4; *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6; *(float4*)(Y + Y3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_4_8R
#define CONV_3D_GEMMV2_KERNEL_4_8R

//for(IH, IW) = (4, 4)
//LB = 4: Size = 2.25, Time = 3.79416 msec, Performace = 1273.49  GFlop/s
//LB = 3: Size = 2.25, Time = 5.20537 msec, Performace =  928.241 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_4_8R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int n_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

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
	CW += (fhs * FW + fws)*IC*OC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws)*IC;//X[0, fhs, fws, 0]

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += ((tx & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N
	int n0 = (((blockIdx.x << LB) + tx) << 3) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((ty >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int SW = (FW - tFW)*IC;
	int fh = W_k / tFW_IC;
	int woffset = (fh*SW + W_k)*OC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + woffset);

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	const int SX = (IW - tFW)*IC;
	float4 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][ty][tx] = x;
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, b0.x, a0);
			simdMM4(v2, b0.y, a0);
			simdMM4(v4, b0.z, a0);
			simdMM4(v6, b0.w, a0);
			simdMM4(v8, b1.x, a0);
			simdMM4(v10, b1.y, a0);
			simdMM4(v12, b1.z, a0);
			simdMM4(v14, b1.w, a0);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		int fh = W_k / tFW_IC;
		int woffset = (fh*SW + W_k)*OC;
		Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + woffset);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		float4 x; int xoffset = fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][ty][tx] = x;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);

		simdMM4(v0, b0.x, a0);
		simdMM4(v2, b0.y, a0);
		simdMM4(v4, b0.z, a0);
		simdMM4(v6, b0.w, a0);
		simdMM4(v8, b1.x, a0);
		simdMM4(v10, b1.y, a0);
		simdMM4(v12, b1.z, a0);
		simdMM4(v14, b1.w, a0);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0; 
	*(float4*)(Y + Y1) = v2; 
	*(float4*)(Y + Y2) = v4;  
	*(float4*)(Y + Y3) = v6;
	*(float4*)(Y + Y4) = v8;
	*(float4*)(Y + Y5) = v10; 
	*(float4*)(Y + Y6) = v12; 
	*(float4*)(Y + Y7) = v14; 
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_4_4R
#define CONV_3D_GEMMV2_KERNEL_4_4R

//for(IH, IW) = (4, 4)
//LB = 4: Size = 2.25, Time = 3.85291 msec, Performace = 1254.08  GFlop/s
//LB = 3: Size = 2.25, Time = 6.1692  msec, Performace =  783.219 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_4_4R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int n_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

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
	CW += (fhs * FW + fws)*IC*OC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws)*IC;//X[0, fhs, fws, 0]

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += ((tx & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N
	int n0 = (((blockIdx.x << LB) + tx) << 2) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((ty & 1) << 1);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int X1 = X0 + IH * IW * IC;

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	const int SX = (IW - tFW)*IC; 
	int fh = X_k / tFW_IC; 
	float2 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = x;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1; 
	const int SW = (FW - tFW)*IC;
	int woffset = (fh*SW + W_k)*OC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + woffset);
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0); 
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, b.x, a);
			simdMM4(v1, b.y, a);
			simdMM4(v2, b.z, a);
			simdMM4(v3, b.w, a);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int fh = X_k / tFW_IC;
		float2 x; int xoffset = fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		Xs[buf][Xs_y][Xs_x] = x;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		int woffset = (fh*SW + W_k)*OC;
		Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

		simdMM4(v0, b.x, a);
		simdMM4(v1, b.y, a);
		simdMM4(v2, b.z, a);
		simdMM4(v3, b.w, a);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride;
	int Y2 = Y1 + Ystride;
	int Y3 = Y2 + Ystride;

	*(float4*)(Y + Y0) = v0;  
	*(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2;  
	*(float4*)(Y + Y3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_4_2R
#define CONV_3D_GEMMV2_KERNEL_4_2R

//for(IH, IW) = (4, 4)
//LB = 4: Size = 2.25, Time = 4.85047 msec, Performace = 996.159 GFlop/s
//LB = 3: Size = 2.25, Time = 7.79391 msec, Performace = 619.95 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_4_2R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int n_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];

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
	CW += (fhs * FW + fws)*IC*OC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws)*IC;//X[0, fhs, fws, 0]

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += ((tx & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N
	int n0 = (((blockIdx.x << LB) + tx) << 1) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + (ty & 1);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;

	//load 1 element from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	const int SX = (IW - tFW)*IC;
	int fh = X_k / tFW_IC;
	int xoffset = fh * SX + X_k;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = X[X0 + xoffset];

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int SW = (FW - tFW)*IC;
	int woffset = (fh*SW + W_k)*OC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + woffset);
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);
			simdMM4(v0, b.x, a);
			simdMM4(v1, b.y, a);
		}
		buf ^= 1;

		//load 1 element from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int fh = X_k / tFW_IC;
		int xoffset = fh * SX + X_k;
		Xs[buf][Xs_y][Xs_x] = X[X0 + xoffset];

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		int woffset = (fh*SW + W_k)*OC;
		Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);
		simdMM4(v0, b.x, a);
		simdMM4(v1, b.y, a);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride;

	*(float4*)(Y + Y0) = v0;
	*(float4*)(Y + Y1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_GEMMV2_KERNEL_2_4R
#define CONV_3D_GEMMV2_KERNEL_2_4R

//for(IH, IW) = (4, 4)
//LB = 4: Size = 2.25, Time = 6.05874 msec, Performace = 797.499 GFlop/s
//LB = 3: Size = 2.25, Time = 10.6197 msec, Performace = 454.986 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemmV2_kernel_2_4R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int n_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

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
	CW += (fhs * FW + fws)*IC*OC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws)*IC;//X[0, fhs, fws, 0]

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	CW += (tx & 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N
	int n0 = (((blockIdx.x << LB) + tx) << 2) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((ty & 1) << 1);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int X1 = X0 + IH * IW * IC;

	//load 1 element from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int SW = (FW - tFW)*IC;
	int fh = W_k / tFW_IC;
	int woffset = (fh*SW + W_k)*OC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = CW[woffset];

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	const int SX = (IW - tFW)*IC; 
	float2 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = x;
	__syncthreads();

	//compute area------------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0); 
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			float2 a = *(float2*)(&Ws[buf][ik][ty << 1]);

			simdMM2(v0, b.x, a);
			simdMM2(v1, b.y, a);
			simdMM2(v2, b.z, a);
			simdMM2(v3, b.w, a);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		int fh = W_k / tFW_IC;
		int woffset = (fh*SW + W_k)*OC;
		Ws[buf][Ws_x][Ws_y] = CW[woffset];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		float2 x; int xoffset = fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		Xs[buf][Xs_y][Xs_x] = x;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		float2 a = *(float2*)(&Ws[buf][ik][ty << 1]);

		simdMM2(v0, b.x, a);
		simdMM2(v1, b.y, a);
		simdMM2(v2, b.z, a);
		simdMM2(v3, b.w, a);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride;
	int Y2 = Y1 + Ystride;
	int Y3 = Y2 + Ystride;

	*(float2*)(Y + Y0) = v0;  
	*(float2*)(Y + Y1) = v1;
	*(float2*)(Y + Y2) = v2;  
	*(float2*)(Y + Y3) = v3;
}

#endif

#endif
