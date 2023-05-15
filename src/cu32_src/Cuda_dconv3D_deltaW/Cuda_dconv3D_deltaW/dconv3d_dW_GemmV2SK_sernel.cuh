#pragma once

#ifndef DECONV3D_DW_GEMM_V2_SK_SERNEL_H
#define DECONV3D_DW_GEMM_V2_SK_SERNEL_H

//Split K to improve parallism:
//	(1) FH * FW >= 2
//	(2) GN = OC:           GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW: GM%4 == 0, GM >= 8
//	(4) GK = N  * OH * OW: GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
//	(6) ph = oph, pw = opw
#ifndef DECONV3D_DW_GEMM_V2_SK_SERNEL_CALL
#define DECONV3D_DW_GEMM_V2_SK_SERNEL_CALL

//LB = log2(BLOCK_SIZE)
//GZ * FH * FW = gridDim.z, GZ -> pIdx

//======[Small GM]===========================================
#define sGemmV2SK_8x2_1(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	sernel_GemmV2SK_8x2_1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GIC<<1>>LB, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB>>1, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define sGemmV2SK_4x2_1(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	sernel_GemmV2SK_4x2_1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GIC<<1>>LB, GN>>LB>>2, GZ * FH * FW), dim3(1<<LB>>1, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define sGemmV2SK_2x2_1(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	sernel_GemmV2SK_2x2_1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GIC<<1>>LB, GN>>LB>>1, GZ * FH * FW), dim3(1<<LB>>1, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//======[Small GN]===========================================
#define sGemmV2SK_1_4x2(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	sernel_GemmV2SK_1_4x2<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>2, GN<<1>>LB, GZ * FH * FW), dim3(1<<LB, 1<<LB>>1), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

#define sGemmV2SK_1_2x2(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	sernel_GemmV2SK_1_2x2<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>1, GN<<1>>LB, GZ * FH * FW), dim3(1<<LB, 1<<LB>>1), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)


#endif


//======[Small GM]===========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*0.5), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_SERNEL_8X2_1
#define DECONV3D_DW_GEMM_V2_SK_SERNEL_8X2_1

//synchronized: 
//[IC = 8]: LB = 3: Size = 0.158203, Time = 1.206 msec, Performace = 281.707 GFlop/s
//[IC = 4]: LB = 3: Size = 0.158203, Time = 1.338 msec, Performace = 253.915 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmV2SK_8x2_1(int GZ,
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
	int ic0 = ((blockIdx.x << LB >> 1) + tx) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0;//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * IC * sh
	IC = sw * IC;//IC -> sw * IC

	//load 8 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k1 = tx + GK_start;
	int Y_oh1 = Y_k1 / tOW_N; Y_k1 -= Y_oh1 * tOW_N;
	int Y_ow1 = Y_k1 / N, Y_n1 = Y_k1 - Y_ow1 * N;
	int yoffset1 = ((Y_n1*OH + Y_oh1)*OW + Y_ow1)*OC;
	Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset1);
	Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset1 + 4);

	int Y_k2 = tx + GK_start + STEP2;
	int Y_oh2 = Y_k2 / tOW_N; Y_k2 -= Y_oh2 * tOW_N;
	int Y_ow2 = Y_k2 / N, Y_n2 = Y_k2 - Y_ow2 * N;
	int yoffset2 = ((Y_n2*OH + Y_oh2)*OW + Y_ow2)*OC;
	Ys[buf][tx + STEP2][(ty << 1)] = *(float4*)(deltaY + yoffset2);
	Ys[buf][tx + STEP2][(ty << 1) + 1] = *(float4*)(deltaY + yoffset2 + 4);

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
		int Y_k1 = (ok << LB) + tx + GK_start;
		int Y_oh1 = Y_k1 / tOW_N; Y_k1 -= Y_oh1 * tOW_N;
		int Y_ow1 = Y_k1 / N, Y_n1 = Y_k1 - Y_ow1 * N;
		int yoffset1 = ((Y_n1*OH + Y_oh1)*OW + Y_ow1)*OC;
		Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset1);
		Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset1 + 4);

		int Y_k2 = (ok << LB) + tx + GK_start + STEP2;
		int Y_oh2 = Y_k2 / tOW_N; Y_k2 -= Y_oh2 * tOW_N;
		int Y_ow2 = Y_k2 / N, Y_n2 = Y_k2 - Y_ow2 * N;
		int yoffset2 = ((Y_n2*OH + Y_oh2)*OW + Y_ow2)*OC;
		Ys[buf][tx + STEP2][(ty << 1)] = *(float4*)(deltaY + yoffset2);
		Ys[buf][tx + STEP2][(ty << 1) + 1] = *(float4*)(deltaY + yoffset2 + 4);

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*0.5), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_SERNEL_4X2_1
#define DECONV3D_DW_GEMM_V2_SK_SERNEL_4X2_1

//synchronized: 
//[IC = 8]: LB = 4: Size = 0.158203, Time = 1.63  msec, Performace = 208.429 GFlop/s
//[IC = 4]: LB = 3: Size = 0.158203, Time = 1.706 msec, Performace = 199.143 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmV2SK_4x2_1(int GZ,
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
	int ic0 = ((blockIdx.x << LB >> 1) + tx) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0;//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * IC * sh
	IC = sw * IC;//IC -> sw * IC

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k1 = tx + GK_start;
	int Y_oh1 = Y_k1 / tOW_N; Y_k1 -= Y_oh1 * tOW_N;
	int Y_ow1 = Y_k1 / N, Y_n1 = Y_k1 - Y_ow1 * N;
	int yoffset1 = ((Y_n1*OH + Y_oh1)*OW + Y_ow1)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset1);

	int Y_k2 = tx + GK_start + STEP2;
	int Y_oh2 = Y_k2 / tOW_N; Y_k2 -= Y_oh2 * tOW_N;
	int Y_ow2 = Y_k2 / N, Y_n2 = Y_k2 - Y_ow2 * N;
	int yoffset2 = ((Y_n2*OH + Y_oh2)*OW + Y_ow2)*OC;
	Ys[buf][tx + STEP2][ty] = *(float4*)(deltaY + yoffset2);

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty + GK_start;
	int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
	int X_ow = X_k / N, X_n = X_k - X_ow * N;
	int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
	Xs[buf][ty][tx] = X[xoffset];
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

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k1 = (ok << LB) + tx + GK_start;
		int Y_oh1 = Y_k1 / tOW_N; Y_k1 -= Y_oh1 * tOW_N;
		int Y_ow1 = Y_k1 / N, Y_n1 = Y_k1 - Y_ow1 * N;
		int yoffset1 = ((Y_n1*OH + Y_oh1)*OW + Y_ow1)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset1);

		int Y_k2 = (ok << LB) + tx + GK_start + STEP2;
		int Y_oh2 = Y_k2 / tOW_N; Y_k2 -= Y_oh2 * tOW_N;
		int Y_ow2 = Y_k2 / N, Y_n2 = Y_k2 - Y_ow2 * N;
		int yoffset2 = ((Y_n2*OH + Y_oh2)*OW + Y_ow2)*OC;
		Ys[buf][tx + STEP2][ty] = *(float4*)(deltaY + yoffset2);

		//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty + GK_start;
		int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
		int X_ow = X_k / N, X_n = X_k - X_ow * N;
		int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
		Xs[buf][ty][tx] = X[xoffset];
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


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*0.5), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_SERNEL_2X2_1
#define DECONV3D_DW_GEMM_V2_SK_SERNEL_2X2_1

//synchronized: 
//(IH, IW) = 6 -> (OH, OW) = 3
//[IC = 8]: LB = 3: Size = 0.158203, Time = 2.466 msec, Performace = 137.769 GFlop/s
//[IC = 4]: LB = 3: Size = 0.158203, Time = 2.508 msec, Performace = 135.462 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmV2SK_2x2_1(int GZ,
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
	int ic0 = ((blockIdx.x << LB >> 1) + tx) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	X += (tfh*IW + tfw)*IC + ic0;//X[0, tfh0, tfw0, tic0]

	IW = IW * IC;
	IH = IH * IW;//IH -> IH * IW * IC
	IW = IW * sh;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k1 = tx + GK_start;
	int Y_oh1 = Y_k1 / tOW_N; Y_k1 -= Y_oh1 * tOW_N;
	int Y_ow1 = Y_k1 / N, Y_n1 = Y_k1 - Y_ow1 * N;
	int yoffset1 = ((Y_n1*OH + Y_oh1)*OW + Y_ow1)*OC;
	Ys[buf][tx][ty] = *(float2*)(deltaY + yoffset1);

	int Y_k2 = tx + GK_start + STEP2;
	int Y_oh2 = Y_k2 / tOW_N; Y_k2 -= Y_oh2 * tOW_N;
	int Y_ow2 = Y_k2 / N, Y_n2 = Y_k2 - Y_ow2 * N;
	int yoffset2 = ((Y_n2*OH + Y_oh2)*OW + Y_ow2)*OC;
	Ys[buf][tx + STEP2][ty] = *(float2*)(deltaY + yoffset2);
	
	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty + GK_start;
	int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
	int X_ow = X_k / N, X_n = X_k - X_ow * N;
	int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
	Xs[buf][ty][tx] = X[xoffset];
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

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k1 = (ok << LB) + tx + GK_start;
		int Y_oh1 = Y_k1 / tOW_N; Y_k1 -= Y_oh1 * tOW_N;
		int Y_ow1 = Y_k1 / N, Y_n1 = Y_k1 - Y_ow1 * N;
		int yoffset1 = ((Y_n1*OH + Y_oh1)*OW + Y_ow1)*OC;
		Ys[buf][tx][ty] = *(float2*)(deltaY + yoffset1);

		int Y_k2 = (ok << LB) + tx + GK_start + STEP2;
		int Y_oh2 = Y_k2 / tOW_N; Y_k2 -= Y_oh2 * tOW_N;
		int Y_ow2 = Y_k2 / N, Y_n2 = Y_k2 - Y_ow2 * N;
		int yoffset2 = ((Y_n2*OH + Y_oh2)*OW + Y_ow2)*OC;
		Ys[buf][tx + STEP2][ty] = *(float2*)(deltaY + yoffset2);
		
		//load 1 element  from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty + GK_start;
		int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
		int X_ow = X_k / N, X_n = X_k - X_ow * N;
		int xoffset = X_n * IH + X_oh * IW + X_ow * IC;
		Xs[buf][ty][tx] = X[xoffset];
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

	const int W1 = W0 + Wstride;
	deltaW[W0] = v0.x;
	deltaW[W1] = v0.y;
}

#endif


//======[Small GN]===========================================
//(Y: BLOCK_SIZE*0.5, X: BLOCK_SIZE*4), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_SERNEL_1_4X2
#define DECONV3D_DW_GEMM_V2_SK_SERNEL_1_4X2

//synchronized: 
//(IH, IW) = 6 -> (OH, OW) = 3
//[OC = 8]: LB = 3: Size = 0.158203, Time = 1.902 msec, Performace = 178.622 GFlop/s
//[OC = 4]: LB = 3: Size = 0.158203, Time = 1.932 msec, Performace = 175.848 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmV2SK_1_4x2(int GZ,
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
	int oc0 = ((blockIdx.y << LB >> 1) + ty) + oc_index;
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
	int X_k1 = ty + GK_start;
	int X_oh1 = X_k1 / tOW_N; X_k1 -= X_oh1 * tOW_N;
	int X_ow1 = X_k1 / N, X_n1 = X_k1 - X_ow1 * N;
	int xoffset1 = X_n1 * IH + X_oh1 * IW + X_ow1 * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset1);

	int X_k2 = ty + GK_start + STEP2;
	int X_oh2 = X_k2 / tOW_N; X_k2 -= X_oh2 * tOW_N;
	int X_ow2 = X_k2 / N, X_n2 = X_k2 - X_ow2 * N;
	int xoffset2 = X_n2 * IH + X_oh2 * IW + X_ow2 * IC;
	Xs[buf][ty + STEP2][tx] = *(float4*)(X + xoffset2);

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
		int X_k1 = (ok << LB) + ty + GK_start;
		int X_oh1 = X_k1 / tOW_N; X_k1 -= X_oh1 * tOW_N;
		int X_ow1 = X_k1 / N, X_n1 = X_k1 - X_ow1 * N;
		int xoffset1 = X_n1 * IH + X_oh1 * IW + X_ow1 * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset1);

		int X_k2 = (ok << LB) + ty + GK_start + STEP2;
		int X_oh2 = X_k2 / tOW_N; X_k2 -= X_oh2 * tOW_N;
		int X_ow2 = X_k2 / N, X_n2 = X_k2 - X_ow2 * N;
		int xoffset2 = X_n2 * IH + X_oh2 * IW + X_ow2 * IC;
		Xs[buf][ty + STEP2][tx] = *(float4*)(X + xoffset2);

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


//(Y: BLOCK_SIZE*0.5, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4: N > 16
//LB = 3: N >  8
#ifndef DECONV3D_DW_GEMM_V2_SK_SERNEL_1_2X2
#define DECONV3D_DW_GEMM_V2_SK_SERNEL_1_2X2

//synchronized: 
//(IH, IW) = 6 -> (OH, OW) = 3
//[OC = 8]: LB = 3: Size = 0.158203, Time = 2.886 msec, Performace = 117.72  GFlop/s
//[OC = 4]: LB = 3: Size = 0.158203, Time = 2.87  msec, Performace = 118.376 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmV2SK_1_2x2(int GZ,
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
	int oc0 = ((blockIdx.y << LB >> 1) + ty) + oc_index;
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
	int X_k1 = ty + GK_start;
	int X_oh1 = X_k1 / tOW_N; X_k1 -= X_oh1 * tOW_N;
	int X_ow1 = X_k1 / N, X_n1 = X_k1 - X_ow1 * N;
	int xoffset1 = X_n1 * IH + X_oh1 * IW + X_ow1 * IC;
	Xs[buf][ty][tx] = *(float2*)(X + xoffset1);

	int X_k2 = ty + GK_start + STEP2;
	int X_oh2 = X_k2 / tOW_N; X_k2 -= X_oh2 * tOW_N;
	int X_ow2 = X_k2 / N, X_n2 = X_k2 - X_ow2 * N;
	int xoffset2 = X_n2 * IH + X_oh2 * IW + X_ow2 * IC;
	Xs[buf][ty + STEP2][tx] = *(float2*)(X + xoffset2);

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
		int X_k1 = (ok << LB) + ty + GK_start;
		int X_oh1 = X_k1 / tOW_N; X_k1 -= X_oh1 * tOW_N;
		int X_ow1 = X_k1 / N, X_n1 = X_k1 - X_ow1 * N;
		int xoffset1 = X_n1 * IH + X_oh1 * IW + X_ow1 * IC;
		Xs[buf][ty][tx] = *(float2*)(X + xoffset1);

		int X_k2 = (ok << LB) + ty + GK_start + STEP2;
		int X_oh2 = X_k2 / tOW_N; X_k2 -= X_oh2 * tOW_N;
		int X_ow2 = X_k2 / N, X_n2 = X_k2 - X_ow2 * N;
		int xoffset2 = X_n2 * IH + X_oh2 * IW + X_ow2 * IC;
		Xs[buf][ty + STEP2][tx] = *(float2*)(X + xoffset2);

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

#endif