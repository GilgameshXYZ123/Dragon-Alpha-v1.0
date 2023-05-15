#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_H
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_H

//Unsparse Matrix Method: 
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_CALL
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]============================================================
#define uV2_88s1(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N) \
	zeroPaddingV2_uernel_8_8_s1<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, (FH - ph - 1),(FW - pw - 1),\
			 ic_index,n_index)

//======[OC is power of 2]==================================================
#define uV2_88s1_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOC, ph, pw, GN, N) \
	zeroPaddingV2_uernel_8_8_s1_OC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,LOC, (FH - ph - 1),(FW - pw - 1),\
			 ic_index,n_index)

//======[ph = pw = 1], [FH = FW = 3], [OH, OW > 1] -> [oph = opw = 1]=======
#define uV2_88s1W3P1(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N) \
	zeroPaddingV2_uernel_8_8_s1_W3P1<LB, (1<<LB>>1),(1<<LB)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,OC,\
			 ic_index,n_index)

#define uV2_88s1W3P1_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	zeroPaddingV2_uernel_8_8_s1_W3P1_OC2pow<LB, (1<<LB>>1),(1<<LB)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,LOC,\
			 ic_index,n_index)

//=====[ph = pw = 2], [FH = FW = 5], [OH, OW > 2] -> [oph = opw = 2]============
#define uV2_88s1W5P2(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N) \
	zeroPaddingV2_uernel_8_8_s1_W5P2<LB, (1<<LB>>1),(1<<LB)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,OC,\
			 ic_index,n_index)

#define uV2_88s1W5P2_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	zeroPaddingV2_uernel_8_8_s1_W5P2_OC2pow<LB, (1<<LB>>1),(1<<LB)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,LOC,\
			 ic_index,n_index)

#endif


//======[Common]============================================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1

//(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.242 msec, Performace = 1945.18 GFlop/s
//LB = 3: Size = 1.125, Time = 1.322 msec, Performace = 1827.47 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.125, Time = 1.316 msec, Performace = 1835.8 GFlop/s
//LB = 3: Size = 1.125, Time = 1.46  msec, Performace = 1654.74 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.125, Time = 1.464 msec, Performace = 1650.22 GFlop/s
//LB = 3: Size = 1.125, Time = 1.598 msec, Performace = 1511.84 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPaddingV2_uernel_8_8_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws))*IC + ic0 + ((ty >= STEP) << 2);//W[0, -fhs, -fws, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + tiw + fws)*OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	FW = FW * IC;//FW -> FW * IC
	FH = FH * FW;//FH -> FH * FW * IC
	IC = OC * FH + IC;//IC -> OC*FH*FW*IC + IC

	//load 4 elem from deltaY[N, OH, OW, OC]
	int Y_k = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = OC * (OW - tFW);
	int fh = Y_k / tFW_OC;
	int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int W_fw = (W_k -= fh * tFW_OC) / OC;
	int woffset = W_k * FH - fh * FW - W_fw * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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

		//load 4 elem from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int fh = Y_k / tFW_OC;
		int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int W_fw = (W_k -= fh * tFW_OC) / OC;
		int woffset = W_k * FH - fh * FW - W_fw * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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


//======[OC is power of 2]==================================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_OC2POW

//(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.202 msec, Performace = 2009.92 GFlop/s
//LB = 3: Size = 1.125, Time = 1.286 msec, Performace = 1878.63 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.125, Time = 1.264 msec, Performace = 1911.33 GFlop/s
//LB = 3: Size = 1.125, Time = 1.378 msec, Performace = 1753.21 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.125, Time = 1.416 msec, Performace = 1706.16 GFlop/s
//LB = 3: Size = 1.125, Time = 1.444 msec, Performace = 1673.07 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPaddingV2_uernel_8_8_s1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws))*IC + ic0 + ((ty >= STEP) << 2);//W[0, -fhs, -fws, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + tiw + fws) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	FW = FW * IC;//FW -> FW * IC
	FH = FH * FW;//FH -> FH * FW * IC
	IC = (FH << LOC) + IC;//IC -> OC*FH*FW*IC + IC

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_k = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - tFW) << LOC;
	int fh = Y_k / tFW_OC;
	int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int W_fw = (W_k -= fh * tFW_OC) >> LOC;
	int woffset = W_k * FH - fh * FW - W_fw * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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

		//load 4 elem from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int fh = Y_k / tFW_OC;
		int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int W_fw = (W_k -= fh * tFW_OC) >> LOC;
		int woffset = W_k * FH - fh * FW - W_fw * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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


//======[ph = pw = 1], [FH = FW = 3], [OH, OW > 1] -> [oph = opw = 1]=======
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W3_P1
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W3_P1

//(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.224 msec, Performace = 1973.79 GFlop/s
//LB = 3: Size = 1.125, Time = 1.292 msec, Performace = 1869.91 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.125, Time = 1.308 msec, Performace = 1847.03 GFlop/s
//LB = 3: Size = 1.125, Time = 1.402 msec, Performace = 1723.19 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.125, Time = 1.45 msec, Performace = 1666.15 GFlop/s
//LB = 3: Size = 1.125, Time = 1.47 msec, Performace = 1643.48 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPaddingV2_uernel_8_8_s1_W3P1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//tFH = tFW = 2 or 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,//oph = opw = 1
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - 1, tiw = iw - 1;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;

	int fh_idx = (tFH == 3), fw_idx = (tFW == 3);
	int fhw_offset = ((fh_idx << 1) + fw_idx) * 9;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((2 - fhs) * 3 + (2 - fws))*IC + ic0 + ((ty >= STEP) << 2);//Wr[0, fhs, frws, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + fws + tiw)*OC;
	const int Ystride = OH * OW *OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	int FW = 3 * IC;//FW -> FW * IC = 3*IC
	int FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = (9*OC + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 81*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 3640, It's hard to out of bound of memory 

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_k = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - tFW)*OC;
	int Idx = Y_k / OC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
	int fh = fhw >> 2, fw = fhw & 3;
	int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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

		//load 4 elem from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int Idx = Y_k / OC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
		int fh = fhw >> 2, fw = fhw & 3;
		int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, OC is power of 2
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W3_P1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W3_P1_OC2POW

//(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.19  msec, Performace = 2030.18 GFlop/s
//LB = 3: Size = 1.125, Time = 1.288 msec, Performace = 1875.71 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.125, Time = 1.258 msec, Performace = 1920.44 GFlop/s
//LB = 3: Size = 1.125, Time = 1.358 msec, Performace = 1779.03 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.125, Time = 1.392 msec, Performace = 1735.57 GFlop/s
//LB = 3: Size = 1.125, Time = 1.42  msec, Performace = 1701.35 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPaddingV2_uernel_8_8_s1_W3P1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//tFH = tFW = 2 or 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,//oph = opw = 1
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - 1, tiw = iw - 1;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;

	int fh_idx = (tFH == 3), fw_idx = (tFW == 3);
	int fhw_offset = ((fh_idx << 1) + fw_idx) * 9;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((2 - fhs) * 3 + (2 - fws))*IC + ic0 + ((ty >= STEP) << 2);//Wr[0, fhs, frws, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + fws + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	int FW = 3 * IC;//FW -> FW * IC = 3*IC
	int FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = ((9 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 81*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 3640, It's hard to out of bound of memory 

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_k = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - tFW) << LOC;
	int Idx = Y_k >> LOC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
	int fh = fhw >> 2, fw = fhw & 3;
	int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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

		//load 4 elem from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int Idx = Y_k >> LOC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
		int fh = fhw >> 2, fw = fhw & 3;
		int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		
		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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


//=====[ph = pw = 2], [FH = FW = 5], [OH, OW > 2] -> [oph = opw = 2]========
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W5_P2
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W5_P2

//(IH, IW) = 4:
//LB = 4: Size = 1.5625, Time = 1.18 msec, Performace = 2843.6 GFlop/s
//LB = 3: Size = 1.5625, Time = 1.22 msec, Performace = 2750.36 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.5625, Time = 1.522 msec, Performace = 2204.63 GFlop/s
//LB = 3: Size = 1.5625, Time = 1.614 msec, Performace = 2078.96 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.5625, Time = 1.734 msec, Performace = 1935.09 GFlop/s
//LB = 3: Size = 1.5625, Time = 1.84  msec, Performace = 1823.61 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPaddingV2_uernel_8_8_s1_W5P2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//tFH = tFW = 5, 4, 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,//oph = opw = 2
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - 2, tiw = iw - 2;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 5), 5, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 5), 5, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((4 - fhs) * 5 + (4 - fws))*IC + ic0 + ((ty >= STEP) << 2);//W[0, -fhs, -fws, tic0]

	int fh_idx = (tFH == 4) + ((tFH == 5) << 1);
	int fw_idx = (tFW == 4) + ((tFW == 5) << 1);
	int fhw_offset = ((fh_idx * 3) + fw_idx) * 25;

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + tiw + fws)*OC;//deltaY[0, fhs, fws, 0]
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	int FW =  5 * IC;//FW -> FW * IC = 5*IC
	int FH = 25 * IC;//FH -> FH * FW * IC = 25*IC
	IC = (25 * OC + 1)*IC;//IC -> OC * FH * FW * IC + IC = (25*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) =FH * FH * FW * IC * FW * OC = 625*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 1310, It's hard to out of bound of memory 

	//load 4 elem from deltaY[N, OH, OW, OC]
	int Y_k = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = OC * (OW - tFW);
	int Idx = Y_k / OC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
	int fh = fhw >> 3, fw = fhw & 7;
	int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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

		//load 4 elem from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int Idx = Y_k / OC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
		int fh = fhw >> 3, fw = fhw & 7;
		int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, OC is power of 2
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W5_P2_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W5_P2_OC2POW

//(IH, IW) = 4:
//LB = 4: Size = 1.5625, Time = 1.152 msec, Performace = 2912.71 GFlop/s
//LB = 3: Size = 1.5625, Time = 1.18  msec, Performace = 2843.6 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.5625, Time = 1.46  msec, Performace = 2298.25 GFlop/s
//LB = 3: Size = 1.5625, Time = 1.562 msec, Performace = 2148.17 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.5625, Time = 1.744 msec, Performace = 1923.99 GFlop/s
//LB = 3: Size = 1.5625, Time = 1.776 msec, Performace = 1889.33 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void zeroPaddingV2_uernel_8_8_s1_W5P2_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//tFH = tFW = 5, 4, 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,//oph = opw = 2
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - 2, tiw = iw - 2;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 5), 5, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 5), 5, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((4 - fhs) * 5 + (4 - fws))*IC + ic0 + ((ty >= STEP) << 2);//W[0, -fhs, -fws, tic0]

	int fh_idx = (tFH == 4) + ((tFH == 5) << 1);
	int fw_idx = (tFW == 4) + ((tFW == 5) << 1);
	int fhw_offset = ((fh_idx * 3) + fw_idx) * 25;

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + tiw + fws) << LOC;//deltaY[0, fhs, fws, 0]
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	int FW =  5 * IC;//FW -> FW * IC = 5*IC
	int FH = 25 * IC;//FH -> FH * FW * IC = 25*IC
	IC = ((25 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (25*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) =FH * FH * FW * IC * FW * OC = 625*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 1310, It's hard to out of bound of memory 

	//load 4 elem from deltaY[N, OH, OW, OC]
	int Y_k = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - tFW) << LOC;
	int Idx = Y_k >> LOC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
	int fh = fhw >> 3, fw = fhw & 7;
	int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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

		//load 4 elem from deltaY[N, OH, OW, OC]
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int Idx = Y_k >> LOC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
		int fh = fhw >> 3, fw = fhw & 7;
		int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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

#endif
