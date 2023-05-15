#pragma once

#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_EX2_H
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_EX2_H

//Split K to improve parallism
//we have:
//	(1) FH * FW >= 2
//	(2) GN = OC:           GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW: GM%4 == 0, GM >= 8
//	(4) GK = N  * OH * OW: GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
//	(6) ph = oph, pw = opw
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_EX2_CALL
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_EX2_CALL

//LB = log2(BLOCK_SIZE)
//GZ * FH * FW = gridDim.z, GZ -> pIdx
//j_index = ic_index * FH * FW

//========[OH = OW = 2], [ph = pw = oph = opw = 1], [IH, IW >= 2]========
#define kGemmV2SK88O2P1_LB4(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC) \
	kernel_GemmV2SK_8_8_O2P1_LB4<LB, (1<<LB>>1), ((1<<LB>>1)-1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, oc_index, ic_index)

#define kGemmV2SK88O2P1_LB3(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC) \
	kernel_GemmV2SK_8_8_O2P1_LB3<LB, (1<<LB>>1), ((1<<LB>>1)-1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, oc_index, ic_index)

//========[OH = OW = 4], [ph = pw = oph = opw = 1], [IH, IW >= 4]========
#define kGemmV2SK88O4P1(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC) \
	kernel_GemmV2SK_8_8_O4P1<LB, (1<<LB>>1), ((1<<LB>>1)-1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, oc_index, ic_index)

#define kGemmV2SK88O4P1_n2pow(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, GN, GIC) \
	kernel_GemmV2SK_8_8_O4P1_N2pow<LB, (1<<LB>>1), ((1<<LB>>1)-1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, oc_index, ic_index)

//========[OH = OW = 7], [ph = pw = oph = opw = 1], [IH, IW >= 7]========
#define kGemmV2SK88O7P1(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC) \
	kernel_GemmV2SK_8_8_O7P1<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, oc_index, ic_index)

#define kGemmV2SK88O7P1_n2pow(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, GN, GIC) \
	kernel_GemmV2SK_8_8_O7P1_N2pow<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, oc_index, ic_index)

//========[OH = OW = 8], [ph = pw = oph = opw = 1], [IH, IW >= 8]========
#define kGemmV2SK88O8P1(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC) \
	kernel_GemmV2SK_8_8_O8P1<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, oc_index, ic_index)

#define kGemmV2SK88O8P1_n2pow(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, GN, GIC) \
	kernel_GemmV2SK_8_8_O8P1_N2pow<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, oc_index, ic_index)

#endif


//========[OH = OW = 2], [ph = pw = oph = opw = 1], [IH, IW >= 2]==========
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O2P1_LB4
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O2P1_LB4

//synchronized: 
//LB = 4: Size = 2.25, Time = 2.21 msec, Performace = 2186.35 GFlop/s
//LB = 3: Size = 2.25, Time = 2.61 msec, Performace = 1851.28 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_GemmV2SK_8_8_O2P1_LB4(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,//OH = OW = 2
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw,//oph = opw = 1
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - 1, tfw = fw - 1;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > 2), 2, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > 2), 2, owe);
	int LtOH = (ohe - ohs) >> 1;//log2(tOH)
	int LtOW = (owe - ows) >> 1;//log2(tOW)
	const int LtOH_OW = LtOH + LtOW, GK = N << LtOH_OW;
	const int LtOH_OW_m1 = (1 << LtOH_OW) - 1, LtOW_m1 = (1 << LtOW) - 1;
	deltaY += ((ohs << 1) + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty >= STEP) << 2);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty & STEP_m1) + GK_start;
	int X_n = X_k >> LtOH_OW; X_k &= LtOH_OW_m1;
	int X_oh = X_k >> LtOW, X_ow = X_k & LtOW_m1;
	int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx & STEP_m1) + GK_start;
	int Y_n = Y_k >> LtOH_OW; Y_k &= LtOH_OW_m1;
	int Y_oh = Y_k >> LtOW, Y_ow = Y_k & LtOW_m1;
	int yoffset = ((Y_n << 2) + (Y_oh << 1) + Y_ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB >> 1) + (ty & STEP_m1) + GK_start;
		int X_n = X_k >> LtOH_OW; X_k &= LtOH_OW_m1;
		int X_oh = X_k >> LtOW, X_ow = X_k & LtOW_m1;
		int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB >> 1) + (tx & STEP_m1) + GK_start;
		int Y_n = Y_k >> LtOH_OW; Y_k &= LtOH_OW_m1;
		int Y_oh = Y_k >> LtOW, Y_ow = Y_k & LtOW_m1;
		int yoffset = ((Y_n << 2) + (Y_oh << 1) + Y_ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 3: N % 4 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O2P1_LB3
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O2P1_LB3

//synchronized: 
//LB = 3: Size = 2.25, Time = 2.526 msec, Performace = 1912.84 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_GemmV2SK_8_8_O2P1_LB3(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,//OH = OW = 2
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw,//oph = opw = 1
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - 1, tfw = fw - 1;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > 2), 2, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > 2), 2, owe);
	int LtOH = (ohe - ohs) >> 1;//log2(tOH)
	int LtOW = (owe - ows) >> 1;//log2(tOW)
	const int LtOH_OW = LtOH + LtOW, GK = N << LtOH_OW;
	const int LtOH_OW_m1 = (1 << LtOH_OW) - 1, LtOW_m1 = (1 << LtOW) - 1;
	deltaY += ((ohs << 1) + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty >= STEP) << 2);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty & STEP_m1) + GK_start;
	int X_n = X_k >> LtOH_OW; X_k &= LtOH_OW_m1;
	int X_oh = X_k >> LtOW, X_ow = X_k & LtOW_m1;
	int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx & STEP_m1) + GK_start;
	int Y_n = Y_k >> LtOH_OW; Y_k &= LtOH_OW_m1;
	int Y_oh = Y_k >> LtOW, Y_ow = Y_k & LtOW_m1;
	int yoffset = ((Y_n << 2) + (Y_oh << 1) + Y_ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB >> 1) + (ty & STEP_m1) + GK_start;
		int X_n = X_k >> LtOH_OW; X_k &= LtOH_OW_m1;
		int X_oh = X_k >> LtOW, X_ow = X_k & LtOW_m1;
		int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB >> 1) + (tx & STEP_m1) + GK_start;
		int Y_n = Y_k >> LtOH_OW; Y_k &= LtOH_OW_m1;
		int Y_oh = Y_k >> LtOW, Y_ow = Y_k & LtOW_m1;
		int yoffset = ((Y_n << 2) + (Y_oh << 1) + Y_ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//========[OH = OW = 4], [ph = pw = oph = opw = 1], [IH, IW >= 4]==========
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O4P1
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O4P1

//synchronized: 
//(IH, IW) = 8
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.226 msec, Performace = 1497.78 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.618 msec, Performace = 1335.5 GFlop/s
//LB = 4: Size = 2.25, Time = 2.734 msec, Performace = 1767.31 GFlop/s
//LB = 3: Size = 2.25, Time = 3.352 msec, Performace = 1441.48 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_GemmV2SK_8_8_O4P1(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,//OH = OW = 4
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw,//oph = opw = 1
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - 1, tfw = fw - 1;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > 4), 4, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > 4), 4, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int GK = tOH * tOW * N;
	deltaY += ((ohs << 2) + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	int oh_idx = (tOH == 4), ow_idx = (tOW == 4);
	int ohw_offset = ((oh_idx << 1) + ow_idx) << 4;//stride = 16

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty >= STEP) << 2);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int Idx = X_k / N; char ohw = YIDX_V2_O4P1[ohw_offset + Idx];//when N % STEP == 0, Idx = X_k / N = Y_k / N
	int oh = ohw >> 2, ow = ohw & 3; Idx *= N;
	int xoffset = (X_k - Idx) * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int yoffset = (((Y_k - Idx) << 4) + (oh << 2) + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int Idx = X_k / N;  char ohw = YIDX_V2_O4P1[ohw_offset + Idx];
		int oh = ohw >> 2, ow = ohw & 3; Idx *= N;
		int xoffset = (X_k - Idx) * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int yoffset = (((Y_k - Idx) << 4) + (oh << 2) + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, N is power of 2
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O4P1_N2POW
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O4P1_N2POW

//synchronized: 
//(IH, IW) = 8
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.226 msec, Performace = 1497.78 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.618 msec, Performace = 1335.5 GFlop/s
//LB = 4: Size = 2.25, Time = 2.644 msec, Performace = 1827.47 GFlop/s
//LB = 3: Size = 2.25, Time = 3.066 msec, Performace = 1575.94 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_GemmV2SK_8_8_O4P1_N2pow(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,//tOH = tOW = 4, 3
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw,//oph = opw = 1
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - 1, tfw = fw - 1;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > 4), 4, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > 4), 4, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW << LN, GK = tOH * tOW_N;
	deltaY += ((ohs << 2) + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	int oh_idx = (tOH == 4), ow_idx = (tOW == 4);
	int ohw_offset = ((oh_idx << 1) + ow_idx) << 4;//stride = 16

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty >= STEP) << 2);//X[0, tfh0, tfw0, tic0]

	int LN_m1 = (1 << LN) - 1;
	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int Idx = X_k >> LN; char ohw = YIDX_V2_O4P1[ohw_offset + Idx];
	int oh = ohw >> 2, ow = ohw & 3;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int yoffset = (((Y_k & LN_m1) << 4) + (oh << 2) + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);

	int xoffset = (X_k & LN_m1) * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int Idx = X_k >> LN; char ohw = YIDX_V2_O4P1[ohw_offset + Idx];
		int oh = ohw >> 2, ow = ohw & 3;
		int xoffset = (X_k & LN_m1) * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int yoffset = (((Y_k & LN_m1) << 4) + (oh << 2) + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//========[OH = OW = 7], [ph = pw = oph = opw = 1], [IH, IW >= 7]========
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O7P1
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O7P1

//synchronized: 
//(IH, IW) = 14 -> (OH, OW) = 7
//LB = 4: Size = 0.861328, Time = 1.22  msec, Performace = 1516.14 GFlop/s
//LB = 3: Size = 0.861328, Time = 1.444 msec, Performace = 1280.95 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_8_8_O7P1(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,//OH = OW = 7
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw,//oph = opw = 1
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - 1, tfw = fw - 1;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > 7), 7, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > 7), 7, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int GK = tOH * tOW * N;
	deltaY += ((ohs*7) + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	int oh_idx = (tOH == 7), ow_idx = (tOW == 7);
	int ohw_offset = ((oh_idx << 1) + ow_idx) * 49;//stride = 64

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty >= STEP) << 2);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int Idx = X_k / N; char ohw = YIDX_V2_O7P1X[ohw_offset + Idx];
	int oh = ohw >> 3, ow = ohw & 7; Idx *= N;
	int xoffset = (X_k - Idx) * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int yoffset = (((Y_k - Idx) * 7 + oh) * 7 + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int Idx = X_k / N; char ohw = YIDX_V2_O7P1X[ohw_offset + Idx];
		int oh = ohw >> 3, ow = ohw & 7; Idx *= N;
		int xoffset = (X_k - Idx) * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int yoffset = (((Y_k - Idx) * 7 + oh) * 7 + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int W1 = W0 + Wstride, W2 = W1 + Wstride;
	const int W3 = W2 + Wstride, W4 = W3 + Wstride;
	const int W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O7P1_N2POW
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O7P1_N2POW

//synchronized: 
//(IH, IW) = 14 -> (OH, OW) = 7
//LB = 4: Size = 0.861328, Time = 1.19  msec, Performace = 1554.36 GFlop/s
//LB = 3: Size = 0.861328, Time = 1.334 msec, Performace = 1386.57 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_8_8_O7P1_N2pow(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,//OH = OW = 7
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw,//oph = opw = 1
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - 1, tfw = fw - 1;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > 7), 7, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > 7), 7, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int GK = (tOH * tOW) << LN;
	deltaY += ((ohs * 7) + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	int oh_idx = (tOH == 7), ow_idx = (tOW == 7);
	int ohw_offset = ((oh_idx << 1) + ow_idx) * 49;//stride = 64

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty >= STEP) << 2);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int Idx = X_k >> LN; char ohw = YIDX_V2_O7P1X[ohw_offset + Idx];
	int oh = ohw >> 3, ow = ohw & 7; Idx <<= LN;
	int xoffset = (X_k - Idx) * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int yoffset = (((Y_k - Idx) * 7 + oh) * 7 + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int Idx = X_k >> LN; char ohw = YIDX_V2_O7P1X[ohw_offset + Idx];
		int oh = ohw >> 3, ow = ohw & 7; Idx <<= LN;
		int xoffset = (X_k - Idx) * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int yoffset = (((Y_k - Idx) * 7 + oh) * 7 + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int W1 = W0 + Wstride, W2 = W1 + Wstride;
	const int W3 = W2 + Wstride, W4 = W3 + Wstride;
	const int W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//========[OH = OW = 8], [ph = pw = oph = opw = 1], [IH, IW >= 8]==========
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O8P1
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O8P1

//synchronized: 
//(IH, IW) = 16
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.254 msec, Performace = 1484.89 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.618 msec, Performace = 1335.5 GFlop/s
//LB = 4: Size = 2.25, Time = 3.12  msec, Performace = 1548.67 GFlop/s
//LB = 3: Size = 2.25, Time = 3.754 msec, Performace = 1287.12 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_8_8_O8P1(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,//OH = OW = 8
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw,//oph = opw = 1
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - 1, tfw = fw - 1;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > 8), 8, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > 8), 8, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int GK = tOH * tOW * N;
	deltaY += ((ohs << 3) + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	int oh_idx = (tOH == 8), ow_idx = (tOW == 8);
	int ohw_offset = ((oh_idx << 1) + ow_idx) << 6;//stride = 64

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty >= STEP) << 2);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int Idx = X_k / N; char ohw = YIDX_V2_O8P1X[ohw_offset + Idx];
	int oh = ohw >> 3, ow = ohw & 7; Idx *= N;
	int xoffset = (X_k - Idx) * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int yoffset = (((Y_k - Idx) << 6) + (oh << 3) + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int Idx = X_k / N; char ohw = YIDX_V2_O8P1X[ohw_offset + Idx];
		int oh = ohw >> 3, ow = ohw & 7; Idx *= N;
		int xoffset = (X_k - Idx) * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int yoffset = (((Y_k - Idx) << 6) + (oh << 3) + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0, N is power of 2
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O8P1_N2POW
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_O8P1_N2POW

//synchronized: 
//(IH, IW) = 16: 
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.254 msec, Performace = 1484.89 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.618 msec, Performace = 1335.5 GFlop/s
//LB = 4: Size = 2.25, Time = 3.038 msec, Performace = 1590.47 GFlop/s
//LB = 3: Size = 2.25, Time = 3.478 msec, Performace = 1389.26 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_8_8_O8P1_N2pow(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,//OH = OW = 8
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw,//oph = opw = 1
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - 1, tfw = fw - 1;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > 8), 8, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > 8), 8, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int GK = (tOH * tOW) << LN;
	deltaY += ((ohs << 3) + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	int oh_idx = (tOH == 8), ow_idx = (tOW == 8);
	int ohw_offset = ((oh_idx << 1) + ow_idx) << 6;//stride = 64

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty >= STEP) << 2);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC
	int N_m1 = (1 << LN) - 1;

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int Idx = X_k >> LN; char ohw = YIDX_V2_O8P1X[ohw_offset + Idx];
	int oh = ohw >> 3, ow = ohw & 7;
	
	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int yoffset = (((Y_k & N_m1) << 6) + (oh << 3) + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);

	int xoffset = (X_k & N_m1) * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int Idx = X_k >> LN; char ohw = YIDX_V2_O8P1X[ohw_offset + Idx];
		int oh = ohw >> 3, ow = ohw & 7;
		int xoffset = (X_k & N_m1) * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int yoffset = (((Y_k & N_m1) << 6) + (oh << 3) + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif

#endif