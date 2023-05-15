#pragma once

#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_EX_H
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_EX_H

//Split K to improve parallism
//we have:
//	(1) FH * FW >= 2
//	(2) GN = OC:           GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW: GM%4 == 0, GM >= 8
//	(4) GK = N  * OH * OW: GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
//	(6) ph = oph, pw = opw
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_EX_CALL
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_EX_CALL

//LB = log2(BLOCK_SIZE)
//GZ * FH * FW = gridDim.z, GZ -> pIdx
//j_index = ic_index * FH * FW

//======[N is power of 2]=================================================
#define kGemmV2SK88_n2pow_LB4(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_8_8_N2pow_LB4<LB, (1<<LB>>1), ((1<<LB>>1)-1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK88_n2pow_LB3(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_8_8_N2pow_LB3<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//------------------------------------------------------------------------
#define kGemmV2SK84_n2pow(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_8_4_N2pow<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>2, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK48_n2pow(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_4_8_N2pow<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>2, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK44_n2pow(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_4_4_N2pow<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>2, GN>>LB>>2, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK82_n2pow(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_8_2_N2pow<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>1, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK28_n2pow(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_2_8_N2pow<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>1, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK42_n2pow(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_4_2_N2pow<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>1, GN>>LB>>2, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK24_n2pow(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_2_4_N2pow<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>2, GN>>LB>>1, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#endif


//======[N is power of 2]=================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 3: N % 4 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_N2POW_LB4
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_N2POW_LB4

//synchronized: LB = 4
//(IH, IW) =  4: Size = 2.25, Time = 2.302 msec, Performace = 2098.97 GFlop/s
//(IH, IW) =  8: Size = 2.25, Time = 2.726 msec, Performace = 1772.5  GFlop/s
//(IH, IW) = 16: Size = 2.25, Time = 2.96  msec, Performace = 1632.38 GFlop/s
//(IH, IW) = 32: Size = 2.25, Time = 3.08  msec, Performace = 1568.78 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_GemmV2SK_8_8_N2pow_LB4(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
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
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW << LN, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
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
	const int N_m1 = (1 << LN) - 1;

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty & STEP_m1) + GK_start;
	int oh = X_k / tOW_N, ow = (X_k - oh * tOW_N) >> LN;
	int xoffset = (X_k & N_m1) * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx & STEP_m1) + GK_start;
	int yoffset = (((Y_k & N_m1)*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
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
		int X_k = (ok << LB >> 1) + (ty & STEP_m1) + GK_start;
		int oh = X_k / tOW_N, ow = (X_k - oh * tOW_N) >> LN;
		int xoffset = (X_k & N_m1) * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB >> 1) + (tx & STEP_m1) + GK_start;
		int yoffset = (((Y_k & N_m1)*OH + oh)*OW + ow)*OC;
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 3: N % 4 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_N2POW_LB3
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8_N2POW_LB3

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 2.25, Time = 2.36  msec, Performace = 2047.39 GFlop/s
//LB = 3: Size = 2.25, Time = 2.812 msec, Performace = 1718.29 GFlop/s
//(IH, IW) = 8
//LB = 4: Size = 2.25, Time = 2.786 msec, Performace = 1734.33 GFlop/s
//LB = 3: Size = 2.25, Time = 3.282 msec, Performace = 1472.22 GFlop/s
//(IH, IW) = 16
//LB = 4: Size = 2.25, Time = 3.024 msec, Performace = 1597.83 GFlop/s
//LB = 3: Size = 2.25, Time = 3.55  msec, Performace = 1361.08 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 2.25, Time = 3.138 msec, Performace = 1539.78 GFlop/s
//LB = 3: Size = 2.25, Time = 3.692 msec, Performace = 1308.73 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_8_8_N2pow_LB3(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
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
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW << LN, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
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
	const int N_m1 = (1 << LN) - 1;

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int oh = X_k / tOW_N; X_k -= oh * tOW_N;
	int ow = X_k >> LN, X_n = X_k & N_m1;
	int xoffset = X_n * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n = Y_k & N_m1;
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
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
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k >> LN, X_n = X_k & N_m1;
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n = Y_k & N_m1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
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


//------------------------------------------------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_4_N2POW
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_4_N2POW

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 1.664 msec, Performace = 1451.87 GFlop/s
//LB = 3: Size = 1.125, Time = 1.796 msec, Performace = 1345.17 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 2.16 msec, Performace = 1118.48 GFlop/s
//LB = 3: Size = 1.125, Time = 2.36 msec, Performace = 1023.69 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_8_4_N2pow(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW << LN, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
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

	//prepare for GIC = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty & 1) << 1);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC
	const int N_m1 = (1 << LN) - 1;

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	int oh = X_k / tOW_N; X_k -= oh * tOW_N;
	int ow = X_k >> LN, X_n = X_k & N_m1;
	int xoffset = X_n * IH + oh * IW + ow * IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n = Y_k & N_m1;
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, a0.x, b0);
			simdMM4(v2, a0.y, b0);
			simdMM4(v4, a0.z, b0);
			simdMM4(v6, a0.w, b0);
			simdMM4(v8, a1.x, b0);
			simdMM4(v10, a1.y, b0);
			simdMM4(v12, a1.z, b0);
			simdMM4(v14, a1.w, b0);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k >> LN, X_n = X_k & N_m1;
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n = Y_k & N_m1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, a0.x, b0);
		simdMM4(v2, a0.y, b0);
		simdMM4(v4, a0.z, b0);
		simdMM4(v6, a0.w, b0);
		simdMM4(v8, a1.x, b0);
		simdMM4(v10, a1.y, b0);
		simdMM4(v12, a1.z, b0);
		simdMM4(v14, a1.w, b0);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;
	*(float4*)(deltaW + W1) = v2;
	*(float4*)(deltaW + W2) = v4;
	*(float4*)(deltaW + W3) = v6;
	*(float4*)(deltaW + W4) = v8;
	*(float4*)(deltaW + W5) = v10;
	*(float4*)(deltaW + W6) = v12;
	*(float4*)(deltaW + W7) = v14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_4_8_N2POW
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_4_8_N2POW

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 1.658 msec, Performace = 1457.13 GFlop/s
//LB = 3: Size = 1.125, Time = 1.88  msec, Performace = 1285.06 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 2.166 msec, Performace = 1115.38 GFlop/s
//LB = 3: Size = 1.125, Time = 2.392 msec, Performace = 1010    GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_4_8_N2pow(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW << LN, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
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
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += oc0 + ((tx & 1) << 1);//deltaY[0, 0, 0, toc0]

	//prepare for GIC = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty >= STEP) << 2);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC
	const int N_m1 = (1 << LN) - 1;

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int oh = X_k / tOW_N; X_k -= oh * tOW_N;
	int ow = X_k >> LN, X_n = X_k & N_m1;
	int xoffset = X_n * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	int Y_n = Y_k & N_m1;
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = *(float4*)(&Ys[buf][ik][ty << 1]);

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k >> LN, X_n = X_k & N_m1;
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		int Y_n = Y_k & N_m1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = *(float4*)(&Ys[buf][ik][ty << 1]);

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
	}

	int W1 = W0 + Wstride;
	int W2 = W1 + Wstride;
	int W3 = W2 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_4_4_N2POW
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_4_4_N2POW

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 1.98  msec, Performace = 1220.16 GFlop/s
//LB = 3: Size = 1.125, Time = 2.296 msec, Performace = 1052.23 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 2.606 msec, Performace = 927.06 GFlop/s
//LB = 3: Size = 1.125, Time = 3.102 msec, Performace = 778.826 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_4_4_N2pow(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW << LN, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
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
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += oc0 + ((tx & 1) << 1);//deltaY[0, 0, 0, toc0]

	//prepare for GIC = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty & 1) << 1);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC
	const int N_m1 = (1 << LN) - 1;

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	int oh = X_k / tOW_N; X_k -= oh * tOW_N;
	int ow = X_k >> LN, X_n = X_k & N_m1;
	int xoffset = X_n * IH + oh * IW + ow * IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	int Y_n = Y_k & N_m1;
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);

			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k >> LN, X_n = X_k & N_m1;
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		int Y_n = Y_k & N_m1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}

	int W1 = W0 + Wstride;
	int W2 = W1 + Wstride;
	int W3 = W2 + Wstride;

	*(float4*)(deltaW + W0) = v0;
	*(float4*)(deltaW + W1) = v1;
	*(float4*)(deltaW + W2) = v2;
	*(float4*)(deltaW + W3) = v3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_2_N2POW
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_2_N2POW

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 2.016 msec, Performace = 1198.37 GFlop/s
//LB = 3: Size = 1.125, Time = 2.388 msec, Performace = 1011.69 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 2.606 msec, Performace = 927.06 GFlop/s
//LB = 3: Size = 1.125, Time = 3.214 msec, Performace = 751.686 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_8_2_N2pow(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW << LN, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
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

	//prepare for GIC = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + (ty & 1);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC
	const int N_m1 = (1 << LN) - 1;

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	int oh = X_k / tOW_N; X_k -= oh * tOW_N;
	int ow = X_k >> LN, X_n = X_k & N_m1;
	int xoffset = X_n * IH + oh * IW + ow * IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = X[xoffset];

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n = Y_k & N_m1;
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2  v0 = make_float2(0, 0);
	float2  v2 = make_float2(0, 0);
	float2  v4 = make_float2(0, 0);
	float2  v6 = make_float2(0, 0);
	float2  v8 = make_float2(0, 0);
	float2 v10 = make_float2(0, 0);
	float2 v12 = make_float2(0, 0);
	float2 v14 = make_float2(0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 b0 = *(float2*)(&Xs[buf][ik][tx << 1]);
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

			simdMM2(v0, a0.x, b0);
			simdMM2(v2, a0.y, b0);
			simdMM2(v4, a0.z, b0);
			simdMM2(v6, a0.w, b0);
			simdMM2(v8, a1.x, b0);
			simdMM2(v10, a1.y, b0);
			simdMM2(v12, a1.z, b0);
			simdMM2(v14, a1.w, b0);
		}
		buf ^= 1;

		//load 1 element  from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k >> LN, X_n = X_k & N_m1;
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][Xs_y][Xs_x] = X[xoffset];

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n = Y_k & N_m1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 b0 = *(float2*)(&Xs[buf][ik][tx << 1]);
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

		simdMM2(v0, a0.x, b0);
		simdMM2(v2, a0.y, b0);
		simdMM2(v4, a0.z, b0);
		simdMM2(v6, a0.w, b0);
		simdMM2(v8, a1.x, b0);
		simdMM2(v10, a1.y, b0);
		simdMM2(v12, a1.z, b0);
		simdMM2(v14, a1.w, b0);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float2*)(deltaW + W0) = v0;
	*(float2*)(deltaW + W1) = v2;
	*(float2*)(deltaW + W2) = v4;
	*(float2*)(deltaW + W3) = v6;
	*(float2*)(deltaW + W4) = v8;
	*(float2*)(deltaW + W5) = v10;
	*(float2*)(deltaW + W6) = v12;
	*(float2*)(deltaW + W7) = v14;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_2_8_N2POW
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_2_8_N2POW

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 2.348 msec, Performace = 1028.93  GFlop/s
//LB = 3: Size = 1.125, Time = 2.608 msec, Performace =  926.349 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 3.12  msec, Performace = 774.333 GFlop/s
//LB = 3: Size = 1.125, Time = 3.416 msec, Performace = 707.236 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_2_8_N2pow(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW << LN, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
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
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += oc0 + (tx & 1);//deltaY[0, 0, 0, toc0]

	//prepare for GIC = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty >= STEP) << 2);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC
	const int N_m1 = (1 << LN) - 1;

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int oh = X_k / tOW_N; X_k -= oh * tOW_N;
	int ow = X_k >> LN, X_n = X_k & N_m1;
	int xoffset = X_n * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	int Y_n = Y_k & N_m1;
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = deltaY[yoffset];
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float2 a0 = *(float2*)(&Ys[buf][ik][ty << 1]);
			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k >> LN, X_n = X_k & N_m1;
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 1 element  from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		int Y_n = Y_k & N_m1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][Ys_x][Ys_y] = deltaY[yoffset];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float2 a0 = *(float2*)(&Ys[buf][ik][ty << 1]);
		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
	}

	int W1 = W0 + Wstride;
	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_4_2_N2POW
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_4_2_N2POW

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 2.642 msec, Performace = 914.428 GFlop/s
//LB = 3: Size = 1.125, Time = 3.508 msec, Performace = 688.688 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 3.906 msec, Performace = 618.515 GFlop/s
//LB = 3: Size = 1.125, Time = 4.712 msec, Performace = 512.716 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_4_2_N2pow(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW << LN, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
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
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += oc0 + ((tx & 1) << 1);//deltaY[0, 0, 0, toc0]

	//prepare for GIC = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + (ty & 1);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC
	const int N_m1 = (1 << LN) - 1;

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	int oh = X_k / tOW_N; X_k -= oh * tOW_N;
	int ow = X_k >> LN, X_n = X_k & N_m1;
	int xoffset = X_n * IH + oh * IW + ow * IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = X[xoffset];

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	int Y_n = Y_k & N_m1;
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);
			float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);

			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
			simdMM2(v2, a.z, b);
			simdMM2(v3, a.w, b);
		}
		buf ^= 1;

		//load 1 element  from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k >> LN, X_n = X_k & N_m1;
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][Xs_y][Xs_x] = X[xoffset];

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		int Y_n = Y_k & N_m1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);
		float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
		simdMM2(v2, a.z, b);
		simdMM2(v3, a.w, b);
	}

	int W1 = W0 + Wstride;
	int W2 = W1 + Wstride;
	int W3 = W2 + Wstride;

	*(float2*)(deltaW + W0) = v0;
	*(float2*)(deltaW + W1) = v1;
	*(float2*)(deltaW + W2) = v2;
	*(float2*)(deltaW + W3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_2_4_N2POW
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_2_4_N2POW

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 2.894 msec, Performace = 834.803 GFlop/s
//LB = 3: Size = 1.125, Time = 3.474 msec, Performace = 695.429 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 3.864 msec, Performace = 625.238 GFlop/s
//LB = 3: Size = 1.125, Time = 4.734 msec, Performace = 510.333 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_2_4_N2pow(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW << LN, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
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
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += oc0 + (tx & 1);//deltaY[0, 0, 0, toc0]

	//prepare for GIC = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty & 1) << 1);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC
	const int N_m1 = (1 << LN) - 1;

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	int oh = X_k / tOW_N; X_k -= oh * tOW_N;
	int ow = X_k >> LN, X_n = X_k & N_m1;
	int xoffset = X_n * IH + oh * IW + ow * IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

	//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	int Y_n = Y_k & N_m1;
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = deltaY[yoffset];
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = *(float2*)(&Ys[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k >> LN, X_n = X_k & N_m1;
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		int Y_n = Y_k & N_m1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][Ys_x][Ys_y] = deltaY[yoffset];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = *(float2*)(&Ys[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
	}

	int W1 = W0 + Wstride;
	*(float4*)(deltaW + W0) = v0;
	*(float4*)(deltaW + W1) = v1;
}

#endif

#endif