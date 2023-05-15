#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_S1_H
#define DECONV3D_DX_ZERO_PADDING_UERNEL_S1_H

//Unsparse Matrix Method:
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_S1_CALL
#define DECONV3D_DX_ZERO_PADDING_UERNEL_S1_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]==============================================
#define u88As1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, N,IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//======[OC is power of 2]====================================
#define u88As1_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1_OC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, N,IC,LOC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//========[FH = 3, FW = 3]====================================
#define u88As1W3(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1_W3<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,IH,IW, N,IC,OC, (2-ph),(2-pw),\
			ic_index,j_index)

//------[OC is power of 2]------------------------------------
#define u88As1W3_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1_W3_OC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,IH,IW, N,IC,LOC, (2-ph),(2-pw),\
			ic_index,j_index)

#define u88s1W3x4_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8_s1_W3_x4_OC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,(IH*IW),IW, IC,LOC, (2-ph),(2-pw),\
			ic_index,j_index)

//========[FH = 5, FW = 5]====================================
#define u88As1W5(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1_W5<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (24*IC)), deltaX,IH,IW, N,IC,OC, (4-ph),(4-pw),\
			ic_index,j_index)

//------[OC is power of 2]------------------------------------
#define u88As1W5_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1_W5_OC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (24*IC)), deltaX,IH,IW, N,IC,LOC, (4-ph),(4-pw),\
			ic_index,j_index)

#define u88s1W5x4_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8_s1_W5_x4_OC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (24*IC)), deltaX,(IH*IW),IW, IC,LOC, (4-ph),(4-pw),\
			ic_index,j_index)

#endif


//======[Common]=======================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1

//for(IH, IW) = 32:
//LB = 4: Size = 1.125, Time = 1.564 msec, Performace = 1544.71 GFlop/s
//LB = 3: Size = 1.125, Time = 1.672 msec, Performace = 1444.93 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPadding_uernel_8_8A_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int IW_N = IW * N;
	get_ih_iw_n(j0, ih0, iw0, n0);
	const int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;
	const int Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int tih = ih0 - oph, tiw = iw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for GK = FH * FW * OC
	const int FW_OC = FW * OC, GK = FH * FW_OC, Wstride = FH * FW * IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_oc * Wstride;//[fhr = fwr = 0]
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + Wstride);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - FW)*OC;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int fh = W_k / FW_OC, fw = (W_k -= fh * FW_OC) / OC;

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		float4 x; int yoffset = Y_k + fh * SY;
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		int W_oc = W_k - fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + Wstride);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//======[OC is power of 2]=============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_OC2POW

//for(IH, IW) = 32:
//LB = 4: Size = 1.125, Time = 1.518 msec, Performace = 1591.51 GFlop/s
//LB = 3: Size = 1.125, Time = 1.588 msec, Performace = 1521.36 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPadding_uernel_8_8A_s1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int IW_N = IW * N;
	get_ih_iw_n(j0, ih0, iw0, n0);
	const int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;
	const int Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int tih = ih0 - oph, tiw = iw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for GK = FH * FW * OC
	const int FW_OC = FW << LOC, GK = FH * FW_OC, Wstride = FH * FW * IC;
	const int OC_m1 = (1 << LOC) - 1;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_oc * Wstride;//[fhr = fwr = 0]
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + Wstride);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - FW) << LOC;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int fh = Y_k / FW_OC, fw = (Y_k - fh * FW_OC) >> LOC;
		float4 x; int yoffset = Y_k + fh * SY;
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int W_oc = W_k & OC_m1;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + Wstride);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//======[FH = FW = 3]==================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W3
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W3

//for(IH, IW) = 32:
//LB = 4: Size = 1.125, Time = 1.522 msec, Performace = 1587.33 GFlop/s
//LB = 3: Size = 1.125, Time = 1.592 msec, Performace = 1517.54 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPadding_uernel_8_8A_s1_W3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int IW_N = IW * N;
	get_ih_iw_n(j0, ih0, iw0, n0);
	const int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;
	const int Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int tih = ih0 - oph, tiw = iw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for GK = FH * FW * OC
	const int GK = 9 * OC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_oc * 9 * IC;//[fhr = fwr = 0]
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - 3)*OC;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int Idx = Y_k / OC; char fhw = YIDX_W33[Idx];
		int fh = fhw >> 2, fw = fhw & 3;
		int yoffset = Y_k + fh * SY; 
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int W_oc = W_k - (fh * 3 + fw)*OC;
		int woffset = ((W_oc * 3 - fh) * 3 - fw)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//------[OC is power of 2]---------------------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0
//LB = 4: GK % 16 == 0
//LB = 3: GK %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W3_OC2POW
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W3_OC2POW

//for(IH, IW) = 8:
//LB = 4: Size = 1.125, Time = 1.746 msec, Performace = 1383.69 GFlop/s
//LB = 3: Size = 1.125, Time = 1.808 msec, Performace = 1336.24 GFlop/s
//for(IH, IW) = 16:
//LB = 4: Size = 1.125, Time = 1.466 msec, Performace = 1647.97 GFlop/s
//LB = 3: Size = 1.125, Time = 1.598 msec, Performace = 1511.84 GFlop/s
//for(IH, IW) = 32:
//LB = 4: Size = 1.125, Time = 1.492 msec, Performace = 1619.25 GFlop/s
//LB = 3: Size = 1.125, Time = 1.546 msec, Performace = 1562.69 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPadding_uernel_8_8A_s1_W3_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int IW_N = IW * N;
	get_ih_iw_n(j0, ih0, iw0, n0);
	const int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;
	const int Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int tih = ih0 - oph, tiw = iw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for GK = FH * FW * OC
	const int GK = 9 << LOC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_oc * 9 * IC;//[fhr = fwr = 0]
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - 3) << LOC;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int Idx = Y_k >> LOC; char fhw = YIDX_W33[Idx];
		int fh = fhw >> 2, fw = fhw & 3;
		int yoffset = Y_k + fh * SY;
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int W_oc = W_k - ((fh * 3 + fw) << LOC);
		int woffset = ((W_oc * 3 - fh) * 3 - fw)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, (IH, IW) % 4 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_W3_X4_OC_2POW
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_W3_X4_OC_2POW

//for(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.728 msec, Performace = 1398.1 GFlop/s
//LB = 3: Size = 1.125, Time = 1.684 msec, Performace = 1434.63 GFlop/s
//for(IH, IW) = 8:
//LB = 4: Size = 1.125, Time = 1.512 msec, Performace = 1597.83 GFlop/s
//LB = 3: Size = 1.125, Time = 1.604 msec, Performace = 1506.18 GFlop/s
//for(IH, IW) = 16:
//LB = 4: Size = 1.125, Time = 1.546 msec, Performace = 1562.69 GFlop/s
//LB = 3: Size = 1.125, Time = 1.584 msec, Performace = 1525.2  GFlop/s
//for(IH, IW) = 32:
//LB = 4: Size = 1.125, Time = 1.496 msec, Performace = 1614.92 GFlop/s
//LB = 3: Size = 1.125, Time = 1.582 msec, Performace = 1527.13 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPadding_uernel_8_8_s1_W3_x4_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	deltaY += ((tn0*OH + tih0)*OW + tiw1) << LOC;

	//prepare for GK = FH * FW * OC
	const int GK = 9 << LOC, OC = (1 << LOC), OC_m1 = OC - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_k = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	int Idx = Y_k >> LOC; char fhw = YIDX_W33[Idx];
	int fh = fhw >> 2, fw = fhw & 3;
	int yoffset = ((fh*OW + fw - Idx) << LOC) + Y_k;
	bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
	bool ly0 = ly && (tiw0 >= -fw) && (tiw0 < OW - fw);
	bool ly1 = ly && (tiw1 >= -fw) && (tiw1 < OW - fw);
	bool ly2 = ly && (tiw2 >= -fw) && (tiw2 < OW - fw);
	bool ly3 = ly && (tiw3 >= -fw) && (tiw3 < OW - fw);
	float2 x0 = (ly0 ? *(float2*)(deltaY + yoffset - OC) : F32_2_0);//Y0
	float2 x1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);//Y1
	float2 x2 = (ly2 ? *(float2*)(deltaY + yoffset + OC) : F32_2_0);//Y2
	float2 x3 = (ly3 ? *(float2*)(deltaY + yoffset + (OC << 1)) : F32_2_0);//Y3
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = ((W_k & OC_m1) * 9 - fh * 3 - fw)*IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9*IC);
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int Idx = Y_k >> LOC; char fhw = YIDX_W33[Idx];
		int fh = fhw >> 2, fw = fhw & 3;
		int yoffset = ((fh*OW + fw - Idx) << LOC) + Y_k;
		bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
		bool ly0 = ly && (tiw0 >= -fw) && (tiw0 < OW - fw);
		bool ly1 = ly && (tiw1 >= -fw) && (tiw1 < OW - fw);
		bool ly2 = ly && (tiw2 >= -fw) && (tiw2 < OW - fw);
		bool ly3 = ly && (tiw3 >= -fw) && (tiw3 < OW - fw);
		float2 x0 = (ly0 ? *(float2*)(deltaY + yoffset - OC) : F32_2_0);//Y0
		float2 x1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);//Y1
		float2 x2 = (ly2 ? *(float2*)(deltaY + yoffset + OC) : F32_2_0);//Y2
		float2 x3 = (ly3 ? *(float2*)(deltaY + yoffset + (OC << 1)) : F32_2_0);//Y3
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = ((W_k & OC_m1) * 9 - fh * 3 - fw)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//======[FH = FW = 5]==================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W5
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W5

//for(IH, IW) = 8:
//LB = 4: Size = 1.5625, Time = 2.138 msec, Performace = 1569.43 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.236 msec, Performace = 1500.65 GFlop/s
//for(IH, IW) = 16:
//LB = 4: Size = 1.5625, Time = 2.114 msec, Performace = 1587.25 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.184 msec, Performace = 1536.37 GFlop/s
//for(IH, IW) = 32:
//LB = 4: Size = 3.125, Time = 3.91  msec, Performace = 1716.34 GFlop/s
//LB = 3: Size = 3.125, Time = 4.156 msec, Performace = 1614.75 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPadding_uernel_8_8A_s1_W5(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 5
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int IW_N = IW * N;
	get_ih_iw_n(j0, ih0, iw0, n0);
	const int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;
	const int Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int tih = ih0 - oph, tiw = iw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for GK = FH * FW * OC
	const int GK = 25 * OC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_oc * 25 * IC;//[fhr = fwr = 0]
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - 5)*OC;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int Idx = Y_k / OC; char fhw = YIDX_W55[Idx];
		int fh = fhw >> 3, fw = fhw & 7;
		int yoffset = Y_k + fh * SY;
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int W_oc = W_k - (fh * 5 + fw)*OC;
		int woffset = ((W_oc * 5 - fh) * 5 - fw)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//------[OC is power of 2]---------------------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0
//LB = 4: GK % 16 == 0
//LB = 3: GK %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W5_OC2POW
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W5_OC2POW

//for(IH, IW) = 8:
//LB = 4: Size = 1.5625, Time = 2.044 msec, Performace = 1641.61 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.156 msec, Performace = 1556.33 GFlop/s
//for(IH, IW) = 16:
//LB = 4: Size = 1.5625, Time = 2.058 msec, Performace = 1630.44 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.088 msec, Performace = 1607.01 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPadding_uernel_8_8A_s1_W5_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 5
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int IW_N = IW * N;
	get_ih_iw_n(j0, ih0, iw0, n0);
	const int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;
	const int Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int tih = ih0 - oph, tiw = iw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for GK = FH * FW * OC
	const int GK = 25 << LOC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_oc * 25 * IC;//[fhr = fwr = 0]
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - 5) << LOC;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int Idx = Y_k >> LOC; char fhw = YIDX_W55[Idx];
		int fh = fhw >> 3, fw = fhw & 7;
		int yoffset = Y_k + fh * SY;
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int W_oc = W_k - ((fh * 5 + fw) << LOC);
		int woffset = ((W_oc * 5 - fh) * 5 - fw)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, (IH, IW) % 4 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_W5_X4_OC2POW
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_W5_X4_OC2POW

//for(IH, IW) = 8
//LB = 4: Size = 1.5625, Time = 2.052 msec, Performace = 1635.21 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.226 msec, Performace = 1507.39 GFlop/s
//for(IH, IW) = 16:
//LB = 4: Size = 1.5625, Time = 2.128 msec, Performace = 1576.81 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.112 msec, Performace = 1588.75 GFlop/s
//for(IH, IW) = 32:
//LB = 4: Size = 3.125, Time = 3.814 msec, Performace = 1759.54 GFlop/s
//LB = 3: Size = 3.125, Time = 4.068 msec, Performace = 1649.68 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPadding_uernel_8_8_s1_W5_x4_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 5
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	deltaY += ((tn0*OH + tih0)*OW + tiw1) << LOC;

	//prepare for GK = FH * FW * OC
	const int GK = 25 << LOC, OC = (1 << LOC), OC_m1 = OC - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_k = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	int Idx = Y_k >> LOC; char fhw = YIDX_W55[Idx];
	int fh = fhw >> 3, fw = fhw & 7;
	int yoffset = ((fh*OW + fw - Idx) << LOC) + Y_k;
	bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
	bool ly0 = ly && (tiw0 >= -fw) && (tiw0 < OW - fw);
	bool ly1 = ly && (tiw1 >= -fw) && (tiw1 < OW - fw);
	bool ly2 = ly && (tiw2 >= -fw) && (tiw2 < OW - fw);
	bool ly3 = ly && (tiw3 >= -fw) && (tiw3 < OW - fw);
	float2 x0 = (ly0 ? *(float2*)(deltaY + yoffset - OC) : F32_2_0);//Y0
	float2 x1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);//Y1
	float2 x2 = (ly2 ? *(float2*)(deltaY + yoffset + OC) : F32_2_0);//Y2
	float2 x3 = (ly3 ? *(float2*)(deltaY + yoffset + (OC << 1)) : F32_2_0);//Y3
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = ((W_k & OC_m1) * 25 - fh * 5 - fw)*IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int Idx = Y_k >> LOC; char fhw = YIDX_W55[Idx];
		int fh = fhw >> 3, fw = fhw & 7;
		int yoffset = ((fh*OW + fw - Idx) << LOC) + Y_k;
		bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
		bool ly0 = ly && (tiw0 >= -fw) && (tiw0 < OW - fw);
		bool ly1 = ly && (tiw1 >= -fw) && (tiw1 < OW - fw);
		bool ly2 = ly && (tiw2 >= -fw) && (tiw2 < OW - fw);
		bool ly3 = ly && (tiw3 >= -fw) && (tiw3 < OW - fw);
		float2 x0 = (ly0 ? *(float2*)(deltaY + yoffset - OC) : F32_2_0);//Y0
		float2 x1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);//Y1
		float2 x2 = (ly2 ? *(float2*)(deltaY + yoffset + OC) : F32_2_0);//Y2
		float2 x3 = (ly3 ? *(float2*)(deltaY + yoffset + (OC << 1)) : F32_2_0);//Y3
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = ((W_k & OC_m1) * 25 - fh * 5 - fw)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif

#endif