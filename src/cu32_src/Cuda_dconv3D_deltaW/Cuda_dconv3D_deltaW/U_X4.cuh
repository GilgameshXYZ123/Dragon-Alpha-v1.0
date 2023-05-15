

//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: N % 8 == 0
#ifndef JERNEL_V2_V1
#define JERNEL_V2_V1

#define jernelV2_v1(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GIC) \
	jernel_8_8_V2_v1<LB, (1<<LB>>1), (1<<LB), ((1<<LB>>1)-1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//synchronized: 
//(IH, IW) = 16
//LB = 4: Size = 1.125, Time = 1.612 msec, Performace = 1498.71 GFlop/s
//LB = 3: Size = 1.125, Time = 1.842 msec, Performace = 1311.57 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void jernel_8_8_V2_v1(int GZ,
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
//LB = 4: N % 8 == 0
#ifndef JERNEL_V2_V2
#define JERNEL_V2_V2

#define jernelV2_v2(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	jernel_8_8_V2_v2<LB, (1<<LB>>1), (1<<LB), ((1<<LB>>1)-1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//synchronized: 
//(IH, IW) = 16
//LB = 4: Size = 1.125, Time = 1.612 msec, Performace = 1498.71 GFlop/s
//LB = 3: Size = 1.125, Time = 1.842 msec, Performace = 1311.57 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void jernel_8_8_V2_v2(int GZ,
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