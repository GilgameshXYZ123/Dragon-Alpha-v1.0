#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_S1_EX_H
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_S1_EX_H

//Unsparse Matrix Method
//We have:
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_S1_EX_CALL
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_S1_EX_CALL

//LB = log2(BLOCK_SIZE)

#define kV2_88s1_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOC, ph, pw, GN, N) \
	zeroPaddingV2_kernel_8_8_s1_OC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,LOC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

#define kV2_84s1_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOC, ph, pw, GN, N) \
	zeroPaddingV2_kernel_8_4_s1_OC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>2, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,LOC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

#define kV2_48s1_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOC, ph, pw, GN, N) \
	zeroPaddingV2_kernel_4_8_s1_OC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,LOC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

#define kV2_44s1_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOC, ph, pw, GN, N) \
	zeroPaddingV2_kernel_4_4_s1_OC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, N>>LB>>2, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,LOC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4, N % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_8_S1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_8_S1_OC2POW

//(IH, IW) = 2:
//LB = 4: Size = 1.125, Time = 0.814 msec, Performace = 2967.96 GFlop/s
//LB = 3: Size = 1.125, Time = 0.928 msec, Performace = 2603.36 GFlop/s
//(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.122 msec, Performace = 2153.23 GFlop/s
//LB = 3: Size = 1.125, Time = 1.338 msec, Performace = 1805.62 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.125, Time = 1.318 msec, Performace = 1833.02 GFlop/s
//LB = 3: Size = 1.125, Time = 1.494 msec, Performace = 1617.08 GFlop/s
//[(IH, IW) = 16:]
//LB = 4: Size = 1.125 , Time = 1.5 msec, Performace = 1610.61 GFlop/s
//LB = 3: Size = 0.5625, Time = 0.848 msec, Performace = 1424.48 GFlop/s
//(IH, IW) = 32:
//k88W3x4_oc2pow<4>: Size = 2.25, Time = 3.266 msec, Performace = 1479.44 GFlop/s
//LB = 4: Size = 2.25, Time = 3.168 msec, Performace = 1525.2 GFlop/s
//LB = 3: Size = 2.25, Time = 3.408 msec, Performace = 1417.79 GFlop/s
//(IH, IW) = 64:
//k88W3x4_oc2pow<4>: Size = 9, Time = 13.96 msec, Performace = 1384.48 GFlop/s
//LB = 4: Size = 9, Time = 13.8 msec, Performace = 1400.53 GFlop/s
//(IH, IW) = 128:
//k88W3x4_oc2pow<4>: Size = 36, Time = 52.58 msec, Performace = 1470.32 GFlop/s
//LB = 4: Size = 36, Time = 51.2 msec, Performace = 1509.95 GFlop/s
template<int LB, int STEP>
__global__ void zeroPaddingV2_kernel_8_8_s1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

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
	deltaY += (fhs*OW + fws) << LOC;//deltaY[0, fhs, fws, 0]
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws)) * IC;//W[0, -fhs, -fws, 0]

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	int Ystride = (OH * OW) << LOC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	FW = FW * IC;//FW -> FW * IC
	FH = FH * FW;//FH -> FH * FW * IC
	IC = (FH << LOC) + IC;//IC -> OC*FH*FW*IC + IC

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = (OW - tFW) << LOC;
	int fh = Y_k / tFW_OC;
	float4 x; int yoffset = fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_fw = (W_k -= fh * tFW_OC) >> LOC;
	int woffset = W_k * FH - fh * FW - W_fw * IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

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

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int fh = Y_k / tFW_OC;
		float4 x; int yoffset = fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fw = (W_k -= fh * tFW_OC) >> LOC;
		int woffset = W_k * FH - fh * FW - W_fw * IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;
	int X4 = X3 + Xstride, X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*4), N % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4, N % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_4_S1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_4_S1_OC2POW

//(IH, IW) = 2:
//LB = 4: Size = 1.125, Time = 1.026 msec, Performace = 2354.7  GFlop/s
//LB = 3: Size = 1.125, Time = 1.222 msec, Performace = 1977.02 GFlop/s
//(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.448 msec, Performace = 1668.45 GFlop/s
//LB = 3: Size = 1.125, Time = 1.724 msec, Performace = 1401.35 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.125, Time = 1.972 msec, Performace = 1225.11 GFlop/s
//LB = 3: Size = 1.125, Time = 2.14  msec, Performace = 1128.93 GFlop/s
template<int LB, int STEP>
__global__ void zeroPaddingV2_kernel_8_4_s1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

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
	deltaY += (fhs*OW + fws) << LOC;//deltaY[0, fhs, fws, 0]
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws)) * IC;//W[0, -fhs, -fws, 0]

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 2) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx & 1) << 1);
	int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	int Ystride = (OH * OW) << LOC;
	int Y1 = Y0 + Ystride;

	FW = FW * IC;//FW -> FW * IC
	FH = FH * FW;//FH -> FH * FW * IC
	IC = (FH << LOC) + IC;//IC -> OC*FH*FW*IC + IC

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx >> 1;
	const int SY = (OW - tFW) << LOC;
	int fh = Y_k / tFW_OC;
	float2 x; int yoffset = fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = x;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int W_fw = (W_k -= fh * tFW_OC) >> LOC;
	int woffset = W_k * FH - fh * FW - W_fw * IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok << LB) + tx) >> 1;
		int fh = Y_k / tFW_OC;
		float2 x; int yoffset = fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		Ys[buf][Ys_x][Ys_y] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fw = (W_k -= fh * tFW_OC) >> LOC;
		int woffset = W_k * FH - fh * FW - W_fw * IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
	}

	int X1 = X0 + Xstride;
	int X2 = X1 + Xstride;
	int X3 = X2 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4, N % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_4_8_S1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_4_8_S1_OC2POW

//(IH, IW) = 2:
//LB = 4: Size = 1.125, Time = 0.992 msec, Performace = 2435.4 GFlop/s
//LB = 3: Size = 1.125, Time = 1.164 msec, Performace = 2075.53 GFlop/s
//(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.422 msec, Performace = 1698.96 GFlop/s
//LB = 3: Size = 1.125, Time = 1.706 msec, Performace = 1416.13 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.125, Time = 1.842 msec, Performace = 1311.57 GFlop/s
//LB = 3: Size = 1.125, Time = 2.018 msec, Performace = 1197.18 GFlop/s
template<int LB, int STEP>
__global__ void zeroPaddingV2_kernel_4_8_s1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

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
	deltaY += (fhs*OW + fws) << LOC;//deltaY[0, fhs, fws, 0]
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws)) * IC;//W[0, -fhs, -fws, 0]

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	W += ic0 + ((ty & 1) << 1);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	int Ystride = (OH * OW) << LOC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	FW = FW * IC;//FW -> FW * IC
	FH = FH * FW;//FH -> FH * FW * IC
	IC = (FH << LOC) + IC;//IC -> OC*FH*FW*IC + IC

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = (OW - tFW) << LOC;
	int fh = Y_k / tFW_OC;
	float4 x; int yoffset = fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty >> 1; //Ws: with the same ty
	int W_fw = (W_k -= fh * tFW_OC) >> LOC;
	int woffset = W_k * FH - fh * FW - W_fw * IC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(W + woffset);
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);
			simdMM4(v2, y0.y, w0);
			simdMM4(v4, y0.z, w0);
			simdMM4(v6, y0.w, w0);
			simdMM4(v8, y1.x, w0);
			simdMM4(v10, y1.y, w0);
			simdMM4(v12, y1.z, w0);
			simdMM4(v14, y1.w, w0);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int fh = Y_k / tFW_OC;
		float4 x; int yoffset = fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok << LB) + ty) >> 1;
		int W_fw = (W_k -= fh * tFW_OC) >> LOC;
		int woffset = W_k * FH - fh * FW - W_fw * IC;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(W + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0);
		simdMM4(v2, y0.y, w0);
		simdMM4(v4, y0.z, w0);
		simdMM4(v6, y0.w, w0);
		simdMM4(v8, y1.x, w0);
		simdMM4(v10, y1.y, w0);
		simdMM4(v12, y1.z, w0);
		simdMM4(v14, y1.w, w0);
	}

	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;
	int X4 = X3 + Xstride, X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;
	*(float4*)(deltaX + X1) = v2;
	*(float4*)(deltaX + X2) = v4;
	*(float4*)(deltaX + X3) = v6;
	*(float4*)(deltaX + X4) = v8;
	*(float4*)(deltaX + X5) = v10;
	*(float4*)(deltaX + X6) = v12;
	*(float4*)(deltaX + X7) = v14;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*4), N % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4, N % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_4_4_S1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_4_4_S1_OC2POW

//(IH, IW) = 2:
//LB = 4: Size = 1.125, Time = 1.106 msec, Performace = 2184.38 GFlop/s
//LB = 3: Size = 1.125, Time = 1.38  msec, Performace = 1750.67 GFlop/s
//(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.606 msec, Performace = 1504.31  GFlop/s
//LB = 3: Size = 1.125, Time = 2.152 msec, Performace = 1122.64 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.125, Time = 2.076 msec, Performace = 1163.74  GFlop/s
//LB = 3: Size = 1.125, Time = 2.584 msec, Performace =  934.953 GFlop/s
template<int LB, int STEP>
__global__ void zeroPaddingV2_kernel_4_4_s1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

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
	deltaY += (fhs*OW + fws) << LOC;//deltaY[0, fhs, fws, 0]
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws)) * IC;//W[0, -fhs, -fws, 0]

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	W += ic0 + ((ty & 1) << 1);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 2) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx & 1) << 1);
	int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	int Ystride = (OH * OW) << LOC;
	int Y1 = Y0 + Ystride;

	FW = FW * IC;//FW -> FW * IC
	FH = FH * FW;//FH -> FH * FW * IC
	IC = (FH << LOC) + IC;//IC -> OC*FH*FW*IC + IC

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx >> 1;
	const int SY = (OW - tFW) << LOC;
	int fh = Y_k / tFW_OC;
	float2 x; int yoffset = fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = x;

	//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty >> 1; //Ws: with the same ty
	int W_fw = (W_k -= fh * tFW_OC) >> LOC;
	int woffset = W_k * FH - fh * FW - W_fw * IC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(W + woffset);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
			float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w);
			simdMM4(v2, y.z, w);
			simdMM4(v3, y.w, w);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok << LB) + tx) >> 1;
		int fh = Y_k / tFW_OC;
		float2 x; int yoffset = fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		Ys[buf][Ys_x][Ys_y] = x;

		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok << LB) + ty) >> 1;
		int W_fw = (W_k -= fh * tFW_OC) >> LOC;
		int woffset = W_k * FH - fh * FW - W_fw * IC;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(W + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y.x, w);
		simdMM4(v1, y.y, w);
		simdMM4(v2, y.z, w);
		simdMM4(v3, y.w, w);
	}

	int X1 = X0 + Xstride;
	int X2 = X1 + Xstride;
	int X3 = X2 + Xstride;

	*(float4*)(deltaX + X0) = v0;
	*(float4*)(deltaX + X1) = v1;
	*(float4*)(deltaX + X2) = v2;
	*(float4*)(deltaX + X3) = v3;
}

#endif

#endif
