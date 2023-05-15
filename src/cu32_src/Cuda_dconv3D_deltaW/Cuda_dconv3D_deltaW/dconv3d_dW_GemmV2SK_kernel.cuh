#pragma once

#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_H
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_H

//Split K to improve parallism:
//	(1) FH * FW >= 2
//	(2) GN = OC:           GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW: GM%4 == 0, GM >= 8
//	(4) GK = N  * OH * OW: GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
//	(6) ph = oph, pw = opw
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_CALL
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_CALL

//LB = log2(BLOCK_SIZE)
//GZ * FH * FW = gridDim.z, GZ -> pIdx

//======[Common]==============================================
#define kGemmV2SK88(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_8_8<LB, (1<<LB>>1), ((1<<LB>>1)-1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK84(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_8_4<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>2, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK48(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_4_8<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>2, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK44(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_4_4<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>2, GN>>LB>>2, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK82(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_8_2<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>1, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK28(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_2_8<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>1, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK42(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_4_2<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>1, GN>>LB>>2, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK24(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_2_4<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>2, GN>>LB>>1, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//======[Small]===============================================
#define kGemmV2SK22(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_2_2<LB, (1<<LB)>\
		<<< dim3(GIC>>LB>>1, GN>>LB>>1, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//------------------------------------------------------------
#define kGemmV2SK81(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_8_1<LB, (1<<LB)>\
		<<< dim3(GIC>>LB, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK41(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_4_1<LB, (1<<LB)>\
		<<< dim3(GIC>>LB, GN>>LB>>2, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK21(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_2_1<LB, (1<<LB)>\
		<<< dim3(GIC>>LB, GN>>LB>>1, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//------------------------------------------------------------
#define kGemmV2SK14(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_1_4<LB, (1<<LB)>\
		<<< dim3(GIC>>LB>>2, GN>>LB, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK12(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_1_2<LB, (1<<LB)>\
		<<< dim3(GIC>>LB>>1, GN>>LB, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define kGemmV2SK11(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	kernel_GemmV2SK_1_1<LB, (1<<LB)>\
		<<< dim3(GIC>>LB, GN>>LB, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#endif


//======[Common]==============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_8

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 2.25, Time = 2.41  msec, Performace = 2004.91 GFlop/s
//LB = 3: Size = 2.25, Time = 2.784 msec, Performace = 1735.57 GFlop/s
//(IH, IW) = 8
//k88SK<4>: Size = 2.25, Time = 3.476 msec, Performace = 1390.06 GFlop/s
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.226 msec, Performace = 1497.78 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.618 msec, Performace = 1335.5 GFlop/s
//LB = 4: Size = 2.25, Time = 2.806 msec, Performace = 1721.97 GFlop/s
//LB = 3: Size = 2.25, Time = 3.72  msec, Performace = 1298.88 GFlop/s
//(IH, IW) = 16
//k88SK<4>: Size = 2.25, Time = 3.492 msec, Performace = 1383.69 GFlop/s
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.254 msec, Performace = 1484.89 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.618 msec, Performace = 1335.5 GFlop/s
//LB = 4: Size = 2.25, Time = 3.068 msec, Performace = 1574.91 GFlop/s
//LB = 3: Size = 2.25, Time = 4.016 msec, Performace = 1203.15 GFlop/s
//(IH, IW) = 32
//k88SK<4>: Size = 2.25, Time = 3.484 msec, Performace = 1386.86 GFlop/s
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.246 msec, Performace = 1488.55 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.626 msec, Performace = 1332.55 GFlop/s
//LB = 4: Size = 2.25, Time = 3.172 msec, Performace = 1523.28 GFlop/s
//LB = 3: Size = 2.25, Time = 3.712 msec, Performace = 1301.68 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_GemmV2SK_8_8(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0 + ((ty >= STEP) << 2);//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty & STEP_m1) + GK_start;
	int oh = X_k / tOW_N, km1 = oh * tOW_N;
	int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
	int xoffset = X_n * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx & STEP_m1) + GK_start;
	int Y_n = Y_k - km1;//Y_n = Y_k % N
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
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
		int oh = X_k / tOW_N, km1 = oh * tOW_N;
		int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB >> 1) + (tx & STEP_m1) + GK_start;
		int Y_n = Y_k - km1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_4
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_4

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 1.778 msec, Performace = 1358.78 GFlop/s
//LB = 3: Size = 1.125, Time = 2.028 msec, Performace = 1191.28 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 2.302 msec, Performace = 1049.49  GFlop/s
//LB = 3: Size = 1.125, Time = 2.646 msec, Performace =  913.046 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_8_4(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	int oh = X_k / tOW_N, km1 = oh * tOW_N;
	int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
	int xoffset = X_n * IH + oh * IW + ow * IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n = Y_k - km1;//Y_n = Y_k % N
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
		int oh = X_k / tOW_N, km1 = oh * tOW_N;
		int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n = Y_k - km1;
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
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_4_8
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_4_8

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 1.924 msec, Performace = 1255.68 GFlop/s
//LB = 3: Size = 1.125, Time = 2.178 msec, Performace = 1109.24 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 2.478 msec, Performace = 974.947 GFlop/s
//LB = 3: Size = 1.125, Time = 3.222 msec, Performace = 749.82 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_4_8(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int oh = X_k / tOW_N, km1 = oh * tOW_N;
	int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
	int xoffset = X_n * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	int Y_n = Y_k - km1;//Y_n = Y_k % N
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
		int oh = X_k / tOW_N, km1 = oh * tOW_N;
		int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		int Y_n = Y_k - km1;
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
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_4_4
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_4_4

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 2.084 msec, Performace = 1159.27  GFlop/s
//LB = 3: Size = 1.125, Time = 2.672 msec, Performace =  904.161 GFlop/s
//(IH, IW) = 8
//LB = 4: Size = 1.125, Time = 2.444 msec, Performace = 988.51 GFlop/s
//LB = 3: Size = 1.125, Time = 3.178 msec, Performace = 760.201 GFlop/s
//(IH, IW) = 16
//LB = 4: Size = 1.125, Time = 2.616 msec, Performace = 923.516 GFlop/s
//LB = 3: Size = 1.125, Time = 3.416 msec, Performace = 707.236 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 2.724 msec, Performace = 886.901 GFlop/s
//LB = 3: Size = 1.125, Time = 3.572 msec, Performace = 676.349 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_4_4(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	int oh = X_k / tOW_N, km1 = oh * tOW_N;
	int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
	int xoffset = X_n * IH + oh * IW + ow * IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	int Y_n = Y_k - km1;//Y_n = Y_k % N
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
		int oh = X_k / tOW_N, km1 = oh * tOW_N;
		int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		int Y_n = Y_k - km1;
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
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_2
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_2

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 2.126 msec, Performace = 1136.37  GFlop/s
//LB = 3: Size = 1.125, Time = 2.742 msec, Performace =  881.079 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 2.802 msec, Performace = 862.212 GFlop/s
//LB = 3: Size = 1.125, Time = 3.626 msec, Performace = 666.277 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_8_2(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	int oh = X_k / tOW_N, km1 = oh * tOW_N;
	int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
	int xoffset = X_n * IH + oh * IW + ow * IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = X[xoffset];

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n = Y_k - km1;//Y_n = Y_k % N
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
		int oh = X_k / tOW_N, km1 = oh * tOW_N;
		int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][Xs_y][Xs_x] = X[xoffset];

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n = Y_k - km1;
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
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_2_8
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_2_8

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 2.472 msec, Performace = 977.314 GFlop/s
//LB = 3: Size = 1.125, Time = 2.88  msec, Performace = 838.861 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 3.242 msec, Performace = 745.194 GFlop/s
//LB = 3: Size = 1.125, Time = 3.832 msec, Performace = 630.459 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_2_8(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int oh = X_k / tOW_N, km1 = oh * tOW_N;
	int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
	int xoffset = X_n * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	int Y_n = Y_k - km1;//Y_n = Y_k % N
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
		int oh = X_k / tOW_N, km1 = oh * tOW_N;
		int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 1 element  from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		int Y_n = Y_k - km1;
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
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_4_2
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_4_2

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 2.932 msec, Performace = 823.983 GFlop/s
//LB = 3: Size = 1.125, Time = 4.266 msec, Performace = 566.32 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 3.906 msec, Performace = 618.515 GFlop/s
//LB = 3: Size = 1.125, Time = 5.754 msec, Performace = 419.868 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_4_2(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	int oh = X_k / tOW_N, km1 = oh * tOW_N;
	int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
	int xoffset = X_n * IH + oh * IW + ow * IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = X[xoffset];

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	int Y_n = Y_k - km1;//Y_n = Y_k % N
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
		int oh = X_k / tOW_N, km1 = oh * tOW_N;
		int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][Xs_y][Xs_x] = X[xoffset];

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		int Y_n = Y_k - km1;
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
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_2_4
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_2_4

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 3.176 msec, Performace = 760.68 GFlop/s
//LB = 3: Size = 1.125, Time = 4.256 msec, Performace = 567.65 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 4.186 msec, Performace = 577.143 GFlop/s
//LB = 3: Size = 1.125, Time = 5.718 msec, Performace = 422.511 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_2_4(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	int oh = X_k / tOW_N, km1 = oh * tOW_N;
	int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
	int xoffset = X_n * IH + oh * IW + ow * IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

	//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	int Y_n = Y_k - km1;//Y_n = Y_k % N
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
		int oh = X_k / tOW_N, km1 = oh * tOW_N;
		int ow = (X_k - km1) / N, X_n = X_k - (km1 += ow * N);
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(X + xoffset);

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		int Y_n = Y_k - km1;
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


//======[Small]===============================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_2_2
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_2_2

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 1.125, Time = 3.67  msec, Performace = 658.289 GFlop/s
//LB = 3: Size = 1.125, Time = 5.632 msec, Performace = 428.963 GFlop/s
//(IH, IW) = 6 -> (OH, OW) = 3
//LB = 4: Size = 1.26562, Time = 5.196 msec, Performace = 523.077 GFlop/s
//LB = 3: Size = 1.26562, Time = 7.264 msec, Performace = 374.161 GFlop/s
//(IH, IW) = 32
//LB = 4: Size = 1.125, Time = 5.558 msec, Performace = 434.674 GFlop/s
//LB = 3: Size = 1.125, Time = 7.62  msec, Performace = 317.05 GFlop/s
//(IH, IW) = 30 -> (OH, OW) = 15
//LB = 4: Size = 1.11237, Time = 5.512 msec, Performace = 433.379 GFlop/s
//LB = 3: Size = 1.11237, Time = 7.742 msec, Performace = 308.549 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_2_2(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Xs[2][1 << LB][(1 << LB) + 1];

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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GIC = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0;//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty + GK_start;
	int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
	int X_ow = X_k / N, X_n = X_k - X_ow * N;
	int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
	Xs[buf][ty][tx] = *(float2*)(X + xoffset);

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
	int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
	int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
	Ys[buf][tx][ty] = *(float2*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = Ys[buf][ik][ty];
			float2 b = Xs[buf][ik][tx];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty + GK_start;
		int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
		int X_ow = X_k / N, X_n = X_k - X_ow * N;
		int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
		Xs[buf][ty][tx] = *(float2*)(X + xoffset);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
		int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
		int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
		Ys[buf][tx][ty] = *(float2*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = Ys[buf][ik][ty];
		float2 b = Xs[buf][ik][tx];
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		int X_k = k + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k / N, n = X_k - ow * N;

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int xoffset = n * IH + oh * IW + ow * IC;
		float2 b = *(float2*)(X + xoffset);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int yoffset = ((n*OH + oh)*OW + ow)*OC;
		float2 a = *(float2*)(deltaY + yoffset);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP != 0--------------------------------------------

	int W1 = W0 + Wstride;
	*(float2*)(deltaW + W0) = v0;
	*(float2*)(deltaW + W1) = v1;
}

#endif


//------------------------------------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_8_1
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_8_1

//synchronized: 
//[IC = 16]: LB = 4: Size = 0.158203, Time = 0.762 msec, Performace = 445.851 GFlop/s
//[IC =  8]: LB = 3: Size = 0.158203, Time = 0.874 msec, Performace = 388.717 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_8_1(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(2 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GIC = IC
	int ic0 = ((blockIdx.x << LB) + tx) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0;//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * IC * sh
	IC = sw * IC;//IC -> sw * IC

	//load 8 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
	int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
	int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
	Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset);
	Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset + 4);

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty + GK_start;
	int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
	int X_ow = X_k / N, X_n = X_k - X_ow * N;
	int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
	Xs[buf][ty][tx] = X[xoffset];
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) 
		{
			float4 a0 = Ys[buf][ik][(ty << 1)];
			float4 a1 = Ys[buf][ik][(ty << 1) + 1];
			float  b0 = Xs[buf][ik][tx];

			simdMM4(v0, b0, a0);
			simdMM4(v1, b0, a1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
		int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
		int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
		Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset);
		Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset + 4);
		
		//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty + GK_start;
		int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
		int X_ow = X_k / N, X_n = X_k - X_ow * N;
		int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
		Xs[buf][ty][tx] = X[xoffset];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][(ty << 1)];
		float4 a1 = Ys[buf][ik][(ty << 1) + 1];
		float  b0 = Xs[buf][ik][tx];

		simdMM4(v0, b0, a0);
		simdMM4(v1, b0, a1);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice & (STEP - 1)); k < GK_slice; k++)
	{
		int X_k = k + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k / N, n = X_k - ow * N;

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int yoffset = ((n*OH + oh)*OW + ow)*OC;
		float4 a0 = *(float4*)(deltaY + yoffset);
		float4 a1 = *(float4*)(deltaY + yoffset + 4);

		//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int xoffset = n * IH + oh * IW + ow * IC;
		float b0 = X[xoffset];

		simdMM4(v0, b0, a0);
		simdMM4(v1, b0, a1);
	}
	//when GK % STEP != 0--------------------------------------------

	const int W1 = W0 + Wstride, W2 = W1 + Wstride;
	const int W3 = W2 + Wstride, W4 = W3 + Wstride;
	const int W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	deltaW[W0] = v0.x;
	deltaW[W1] = v0.y;
	deltaW[W2] = v0.z;
	deltaW[W3] = v0.w;
	deltaW[W4] = v1.x;
	deltaW[W5] = v1.y;
	deltaW[W6] = v1.z;
	deltaW[W7] = v1.w;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_4_1
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_4_1

//synchronized: 
//[IC = 16]: LB = 4: Size = 0.158203, Time = 0.772 msec, Performace = 440.076 GFlop/s
//[IC =  8]: LB = 3: Size = 0.158203, Time = 1.03  msec, Performace = 329.843 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_4_1(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GIC = IC
	int ic0 = ((blockIdx.x << LB) + tx) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0;//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * IC * sh
	IC = sw * IC;//IC -> sw * IC

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty + GK_start;
	int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
	int X_ow = X_k / N, X_n = X_k - X_ow * N;
	int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
	Xs[buf][ty][tx] = X[xoffset];

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
	int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
	int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 a = Ys[buf][ik][ty];
			float  b = Xs[buf][ik][tx];
			simdMM4(v0, b, a);
		}
		buf ^= 1;

		//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty + GK_start;
		int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
		int X_ow = X_k / N, X_n = X_k - X_ow * N;
		int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
		Xs[buf][ty][tx] = X[xoffset];

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
		int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
		int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 a = Ys[buf][ik][ty];
		float  b = Xs[buf][ik][tx];
		simdMM4(v0, b, a);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice & (STEP - 1)); k < GK_slice; k++)
	{
		int X_k = k + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k / N, n = X_k - ow * N;

		//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int xoffset = n * IH + oh * IW + ow * IC;
		float b = X[xoffset];

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int yoffset = ((n*OH + oh)*OW + ow)*OC;
		float4 a = *(float4*)(deltaY + yoffset);

		simdMM4(v0, b, a);
	}
	//when GK % STEP != 0--------------------------------------------

	const int W1 = W0 + Wstride;
	const int W2 = W1 + Wstride;
	const int W3 = W2 + Wstride;

	deltaW[W0] = v0.x;
	deltaW[W1] = v0.y;
	deltaW[W2] = v0.z;
	deltaW[W3] = v0.w;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_2_1
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_2_1

//synchronized: 
//(IH, IW) = 6 -> (OH, OW) = 3
//LB = 4: Size = 0.632812, Time = 4.036 msec, Performace = 336.708 GFlop/s
//LB = 3: Size = 0.632812, Time = 6.552 msec, Performace = 207.411 GFlop/s
//[IC = 16]: LB = 4: Size = 0.158203, Time = 1.16  msec, Performace = 292.878 GFlop/s
//[IC =  8]: LB = 3: Size = 0.158203, Time = 1.918 msec, Performace = 177.132 GFlop/s
//(IH, IW) = 30 -> (OH, OW) = 15
//LB = 4: Size = 0.556183, Time = 4.374 msec, Performace = 273.067 GFlop/s
//LB = 3: Size = 0.556183, Time = 7.018 msec, Performace = 170.19  GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_2_1(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GIC = IC
	int ic0 = ((blockIdx.x << LB) + tx)  + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0;//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty + GK_start;
	int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
	int X_ow = X_k / N, X_n = X_k - X_ow * N;
	int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
	Xs[buf][ty][tx] = X[xoffset];

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
	int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
	int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
	Ys[buf][tx][ty] = *(float2*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = Ys[buf][ik][ty];
			float  b = Xs[buf][ik][tx];
			simdMM2(v0, b, a);
		}
		buf ^= 1;

		//load 1 element  from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty + GK_start;
		int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
		int X_ow = X_k / N, X_n = X_k - X_ow * N;
		int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
		Xs[buf][ty][tx] = X[xoffset];

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
		int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
		int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
		Ys[buf][tx][ty] = *(float2*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = Ys[buf][ik][ty];
		float  b = Xs[buf][ik][tx];
		simdMM2(v0, b, a);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		int X_k = k + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k / N, n = X_k - ow * N;

		//load 1 element  from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int xoffset = n * IH + oh * IW + ow * IC;
		float  b = X[xoffset];

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int yoffset = ((n*OH + oh)*OW + ow)*OC;
		float2 a = *(float2*)(deltaY + yoffset);

		simdMM2(v0, b, a);
	}
	//when GK % STEP != 0--------------------------------------------

	int W1 = W0 + Wstride;
	deltaW[W0] = v0.x;
	deltaW[W1] = v0.y;
}

#endif


//------------------------------------------------------------
//(Y: BLOCK_SIZE*1,s X: BLOCK_SIZE*4), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_1_4
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_1_4

//synchronized: 
//(IH, IW) = 6 -> (OH, OW) = 3
//[OC = 16]: LB = 4: Size = 0.158203, Time = 1.016 msec, Performace = 334.388 GFlop/s
//[OC =  8]: LB = 3: Size = 0.158203, Time = 1.344 msec, Performace = 252.782 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_1_4(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB][(1 << LB) + 1];
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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GIC = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0;//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty + GK_start;
	int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
	int X_ow = X_k / N, X_n = X_k - X_ow * N;
	int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
	int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
	int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
	Ys[buf][tx][ty] = deltaY[yoffset];
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  a = Ys[buf][ik][ty];
			float4 b = Xs[buf][ik][tx];
			simdMM4(v0, a, b);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty + GK_start;
		int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
		int X_ow = X_k / N, X_n = X_k - X_ow * N;
		int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 1 element  from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
		int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
		int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
		Ys[buf][tx][ty] = deltaY[yoffset];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  a = Ys[buf][ik][ty];
		float4 b = Xs[buf][ik][tx];
		simdMM4(v0, a, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		int X_k = k + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k / N, n = X_k - ow * N;

		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int xoffset = n * IH + oh * IW + ow * IC;
		float4 b = *(float4*)(X + xoffset);

		//load 1 element from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int yoffset = ((n*OH + oh)*OW + ow)*OC;
		float a = deltaY[yoffset];

		simdMM4(v0, a, b);
	}
	//when GK % STEP != 0--------------------------------------------

	*(float4*)(deltaW + W0) = v0;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_1_2
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_1_2

//synchronized: 
//(IH, IW) = 6 -> (OH, OW) = 3
//LB = 4: Size = 0.632812, Time = 4.488 msec, Performace = 302.797 GFlop/s
//LB = 3: Size = 0.632812, Time = 6.868 msec, Performace = 197.868 GFlop/s
//[OC = 16]: LB = 4: Size = 0.158203, Time = 1.314 msec, Performace = 258.553 GFlop/s
//[OC =  8]: LB = 3: Size = 0.158203, Time = 2.066 msec, Performace = 164.443 GFlop/s
//(IH, IW) = 30 -> (OH, OW) = 15
//LB = 4: Size = 0.556183, Time = 4.76  msec, Performace = 250.923 GFlop/s
//LB = 3: Size = 0.556183, Time = 7.506 msec, Performace = 159.125 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_1_2(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Xs[2][1 << LB][(1 << LB) + 1];

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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GIC = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0;//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty + GK_start;
	int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
	int X_ow = X_k / N, X_n = X_k - X_ow * N;
	int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
	Xs[buf][ty][tx] = *(float2*)(X + xoffset);

	//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
	int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
	int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
	Ys[buf][tx][ty] = deltaY[yoffset];
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  a = Ys[buf][ik][ty];
			float2 b = Xs[buf][ik][tx];
			simdMM2(v0, a, b);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty + GK_start;
		int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
		int X_ow = X_k / N, X_n = X_k - X_ow * N;
		int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
		Xs[buf][ty][tx] = *(float2*)(X + xoffset);

		//load 1 element  from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
		int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
		int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
		Ys[buf][tx][ty] = deltaY[yoffset];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  a = Ys[buf][ik][ty];
		float2 b = Xs[buf][ik][tx];
		simdMM2(v0, a, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		int X_k = k + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k / N, n = X_k - ow * N;

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int xoffset = n * IH + oh * IW + ow * IC;
		float2 b = *(float2*)(X + xoffset);

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int yoffset = ((n*OH + oh)*OW + ow)*OC;
		float a = deltaY[yoffset];

		simdMM2(v0, a, b);
	}
	//when GK % STEP != 0--------------------------------------------

	*(float2*)(deltaW + W0) = v0;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_KERNEL_1_1
#define DECONV3D_DW_GEMM_V2_SK_KERNEL_1_1

//synchronized: 
//(IH, IW) = 6 -> (OH, OW) = 3
//LB = 4: Size = 0.632812, Time =  7.258 msec, Performace = 187.235 GFlop/s
//LB = 3: Size = 0.632812, Time = 12.308 msec, Performace = 110.412 GFlop/s
//(IH, IW) = 30 -> (OH, OW) = 15
//LB = 4: Size = 0.556183, Time =  7.902 msec, Performace = 151.151  GFlop/s
//LB = 3: Size = 0.556183, Time = 13.116 msec, Performace =  91.0638 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmV2SK_1_1(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float Xs[2][1 << LB][(1 << LB) + 1];

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
	int tOW_N = tOW * N, GK = tOH * tOW_N;
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
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GIC = IC
	int ic0 = ((blockIdx.x << LB) + tx) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0;//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load  element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty + GK_start;
	int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
	int X_ow = X_k / N, X_n = X_k - X_ow * N;
	int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
	Xs[buf][ty][tx] = X[xoffset];

	//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
	int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
	int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
	Ys[buf][tx][ty] = deltaY[yoffset];
	__syncthreads();

	//compute area-----------------------------------------------------
	float v = 0;
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float a = Ys[buf][ik][ty];
			float b = Xs[buf][ik][tx];
			v += a * b;
		}
		buf ^= 1;

		//load 1 element  from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty + GK_start;
		int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
		int X_ow = X_k / N, X_n = X_k - X_ow * N;
		int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
		Xs[buf][ty][tx] = X[xoffset];

		//load 1 element  from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
		int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
		int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
		Ys[buf][tx][ty] = deltaY[yoffset];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float a = Ys[buf][ik][ty];
		float b = Xs[buf][ik][tx];
		v += a * b;
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		int X_k = k + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k / N, n = X_k - ow * N;

		//load 1 element  from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int xoffset = n * IH + oh * IW + ow * IC;
		float b = X[xoffset];

		//load 1 element  from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int yoffset = ((n*OH + oh)*OW + ow)*OC;
		float a = deltaY[yoffset];

		v += a * b;
	}
	//when GK % STEP != 0--------------------------------------------

	deltaW[W0] = v;
}

#endif

#endif