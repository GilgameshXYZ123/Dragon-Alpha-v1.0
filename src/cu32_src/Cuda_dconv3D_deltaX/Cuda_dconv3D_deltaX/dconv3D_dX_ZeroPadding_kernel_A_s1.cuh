#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_A_S1_H
#define DECONV3D_DX_ZERO_PADDING_KERNEL_A_S1_H

//[Upgrade to A] Unsparse Matrix Method:
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
//(6) N % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_A_S1_CALL
#define DECONV3D_DX_ZERO_PADDING_KERNEL_A_S1_CALL

//======[Common]==============================================
#define k88As1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8A_s1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, N,IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//======[OC is power of 2]====================================
#define k88As1_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8A_s1_OC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, N,IC,LOC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//=======[FH = FW = 3]=========================================
#define k88As1W3(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8A_s1_W3<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,IH,IW, N,IC,OC, (2-ph),(2-pw),\
			ic_index,j_index)

#define k88As1W3_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8A_s1_W3_OC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,IH,IW, N,IC,LOC, (2-ph),(2-pw),\
			ic_index,j_index)

//=======[FH = FW = 5]==========================================
#define k88As1W5(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8A_s1_W5<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (24*IC)), deltaX,IH,IW, N,IC,OC, (4-ph),(4-pw),\
			ic_index,j_index)

#define k88As1W5_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8A_s1_W5_OC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (24*IC)), deltaX,IH,IW, N,IC,LOC, (4-ph),(4-pw),\
			ic_index,j_index)

#endif


//======[Common]==============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0,
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1

//for(IH, IW) = 32:
//LB = 4: Size = 1.125, Time = 1.648 msec, Performace = 1465.97 GFlop/s
//LB = 3: Size = 1.125, Time = 1.94  msec, Performace = 1245.32 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8A_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

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
	const int FW_OC = FW * OC, GK = FH * FW_OC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);//[fhr = fwr = 0]
	int woffset = W_oc * FH*FW*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = tx - ((tx >= STEP) << LB >> 1);//[fhr = fwr = 0]
	const int SY = (OW - FW)*OC;
	float4 x; bool ly = LOAD_Y(tih, tiw, 0, 0);
	x.x = (ly ? deltaY[Y0 + Y_oc] : 0);
	x.y = (ly ? deltaY[Y1 + Y_oc] : 0);
	x.z = (ly ? deltaY[Y2 + Y_oc] : 0);
	x.w = (ly ? deltaY[Y3 + Y_oc] : 0);
	Ys[buf][tx][ty] = x;
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

		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int fh = W_k / FW_OC, fw = (W_k -= fh * FW_OC) / OC;

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int yoffset = Y_k + fh * SY; bool ly = LOAD_Y(tih, tiw, fh, fw);
		x.x = (ly ? deltaY[Y0 + yoffset] : 0);
		x.y = (ly ? deltaY[Y1 + yoffset] : 0);
		x.z = (ly ? deltaY[Y2 + yoffset] : 0);
		x.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_oc = W_k - fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
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


//======[OC is power of 2]====================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0,
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_OC2POW

//LB = 4: Size = 1.125, Time = 1.574 msec, Performace = 1534.89 GFlop/s
//LB = 3: Size = 1.125, Time = 1.8   msec, Performace = 1342.18 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8A_s1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

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
	const int FW_OC = FW << LOC, GK = FH * FW_OC, OC_m1 = (1 << LOC) - 1;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);//[fhr = fwr = 0]
	int woffset = W_oc * FH*FW*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = tx - ((tx >= STEP) << LB >> 1);//[fhr = fwr = 0]
	const int SY = (OW - FW) << LOC;
	float4 x; bool ly = LOAD_Y(tih, tiw, 0, 0);
	x.x = (ly ? deltaY[Y0 + Y_oc] : 0);
	x.y = (ly ? deltaY[Y1 + Y_oc] : 0);
	x.z = (ly ? deltaY[Y2 + Y_oc] : 0);
	x.w = (ly ? deltaY[Y3 + Y_oc] : 0);
	Ys[buf][tx][ty] = x;
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

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int fh = Y_k / FW_OC, fw = (Y_k - fh * FW_OC) >> LOC;
		int yoffset = Y_k + fh * SY; bool ly = LOAD_Y(tih, tiw, fh, fw);
		x.x = (ly ? deltaY[Y0 + yoffset] : 0);
		x.y = (ly ? deltaY[Y1 + yoffset] : 0);
		x.z = (ly ? deltaY[Y2 + yoffset] : 0);
		x.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k & OC_m1;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
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


//======[FH = FW == 3]========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0,
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W3
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W3

//for(IH, IW) = 32:
//LB = 4: Size = 1.125, Time = 1.552 msec, Performace = 1556.65 GFlop/s
//LB = 3: Size = 1.125, Time = 1.836 msec, Performace = 1315.86 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8A_s1_W3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

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
	int W_oc = ty - ((ty >= STEP) << LB >> 1);//[fhr = fwr = 0]
	int woffset = W_oc * 9 * IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = tx - ((tx >= STEP) << LB >> 1);//[fhr = fwr = 0]
	const int SY = (OW - 3)*OC;
	float4 x; bool ly = LOAD_Y(tih, tiw, 0, 0);
	x.x = (ly ? deltaY[Y0 + Y_oc] : 0);
	x.y = (ly ? deltaY[Y1 + Y_oc] : 0);
	x.z = (ly ? deltaY[Y2 + Y_oc] : 0);
	x.w = (ly ? deltaY[Y3 + Y_oc] : 0);
	Ys[buf][tx][ty] = x;
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

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = Y_k / OC; char fhw = YIDX_W33[Idx];
		int fh = fhw >> 2, fw = fhw & 3;
		int yoffset = Y_k + fh * SY;
		float4 x; bool ly = LOAD_Y(tih, tiw, fh, fw);
		x.x = (ly ? deltaY[Y0 + yoffset] : 0);
		x.y = (ly ? deltaY[Y1 + yoffset] : 0);
		x.z = (ly ? deltaY[Y2 + yoffset] : 0);
		x.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k - (fh * 3 + fw)*OC;
		int woffset = ((W_oc * 3 - fh) * 3 - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0, OC is power 2
//LB = 4: OC % 8 == 0,
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W3_OC2POW
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W3_OC2POW

//k88s1W3x4<4>: Size = 1.125, Time = 1.522 msec, Performace = 1587.33 GFlop/s
//LB = 4: Size = 1.125, Time = 1.512 msec, Performace = 1597.83 GFlop/s
//LB = 3: Size = 1.125, Time = 1.74  msec, Performace = 1388.46 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8A_s1_W3_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

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
	const int GK = 9 << LOC, OC_m1 = (1 << LOC) - 1;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);//[fhr = fwr = 0]
	int woffset = W_oc * 9 * IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = tx - ((tx >= STEP) << LB >> 1);//[fhr = fwr = 0]
	const int SY = (OW - 3) << LOC;
	float4 x; bool ly = LOAD_Y(tih, tiw, 0, 0);
	x.x = (ly ? deltaY[Y0 + Y_oc] : 0);
	x.y = (ly ? deltaY[Y1 + Y_oc] : 0);
	x.z = (ly ? deltaY[Y2 + Y_oc] : 0);
	x.w = (ly ? deltaY[Y3 + Y_oc] : 0);
	Ys[buf][tx][ty] = x;
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

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = Y_k >> LOC; char fhw = YIDX_W33[Idx];
		int fh = fhw >> 2, fw = fhw & 3;
		int yoffset = Y_k + fh * SY; bool ly = LOAD_Y(tih, tiw, fh, fw);
		x.x = (ly ? deltaY[Y0 + yoffset] : 0);
		x.y = (ly ? deltaY[Y1 + yoffset] : 0);
		x.z = (ly ? deltaY[Y2 + yoffset] : 0);
		x.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k & OC_m1;
		int woffset = ((W_oc * 3 - fh) * 3 - fw)*IC;
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


//======[FH = FW == 5]========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0,
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W5
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W5

//k88s1W5x4<4>: Size = 1.5625, Time = 2.196 msec, Performace = 1527.98 GFlop/s
//k88s1W5x4<3>: Size = 1.5625, Time = 2.714 msec, Performace = 1236.35 GFlop/s
//LB = 4: Size = 1.5625, Time = 2.06  msec, Performace = 1628.86 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.536 msec, Performace = 1323.12 GFlop/s
//for(IH, IW) = 8: 
//LB = 4: Size = 1.5625, Time = 2.072 msec, Performace = 1619.42 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.504 msec, Performace = 1340.03 GFlop/s
//for(IH, IW) = 16:
//LB = 4: Size = 1.5625, Time = 2.138 msec, Performace = 1569.43 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.508 msec, Performace = 1337.9  GFlop/s
//for(IH, IW) = 32:
//LB = 4: Size = 3.125, Time = 3.952 msec, Performace = 1698.1 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8A_s1_W5(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 5
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

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
	int W_oc = ty - ((ty >= STEP) << LB >> 1);//[fhr = fwr = 0]
	int woffset = W_oc * 25 * IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = tx - ((tx >= STEP) << LB >> 1);//[fhr = fwr = 0]
	const int SY = (OW - 5)*OC;
	float4 x; bool ly = LOAD_Y(tih, tiw, 0, 0);
	x.x = (ly ? deltaY[Y0 + Y_oc] : 0);
	x.y = (ly ? deltaY[Y1 + Y_oc] : 0);
	x.z = (ly ? deltaY[Y2 + Y_oc] : 0);
	x.w = (ly ? deltaY[Y3 + Y_oc] : 0);
	Ys[buf][tx][ty] = x;
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

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = Y_k / OC; char fhw = YIDX_W55[Idx];
		int fh = fhw >> 3, fw = fhw & 7;
		int yoffset = Y_k + fh * SY; bool ly = LOAD_Y(tih, tiw, fh, fw);
		x.x = (ly ? deltaY[Y0 + yoffset] : 0);
		x.y = (ly ? deltaY[Y1 + yoffset] : 0);
		x.z = (ly ? deltaY[Y2 + yoffset] : 0);
		x.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k - (fh * 5 + fw) * OC;
		int woffset = ((W_oc * 5 - fh) * 5 - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4: OC % 8 == 0,
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W5_OC2POW
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W5_OC2POW

//k88s1W5x4_oc2pow<4>: Size = 1.5625, Time = 2.034 msec, Performace = 1649.68 GFlop/s
//k88s1W5x4_oc2pow<3>: Size = 1.5625, Time = 2.392 msec, Performace = 1402.78 GFlop/s
//LB = 4: Size = 1.5625, Time = 2.022 msec, Performace = 1659.47 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.318 msec, Performace = 1447.56 GFlop/s
//for(IH, IW) = 8:
//LB = 4: Size = 1.5625, Time = 2.028 msec, Performace = 1654.56 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.27  msec, Performace = 1478.17 GFlop/s
//for(IH, IW) = 16:
//LB = 4: Size = 1.5625, Time = 2.018 msec, Performace = 1662.76 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.344 msec, Performace = 1431.5 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8A_s1_W5_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 5
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

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
	const int GK = 25 << LOC, OC_m1 = (1 << LOC) - 1;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);//[fhr = fwr = 0]
	int woffset = W_oc * 25 * IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = tx - ((tx >= STEP) << LB >> 1);//[fhr = fwr = 0]
	const int SY = (OW - 5) << LOC;
	float4 x; bool ly = LOAD_Y(tih, tiw, 0, 0);
	x.x = (ly ? deltaY[Y0 + Y_oc] : 0);
	x.y = (ly ? deltaY[Y1 + Y_oc] : 0);
	x.z = (ly ? deltaY[Y2 + Y_oc] : 0);
	x.w = (ly ? deltaY[Y3 + Y_oc] : 0);
	Ys[buf][tx][ty] = x;
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

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = Y_k >> LOC; char fhw = YIDX_W55[Idx];
		int fh = fhw >> 3, fw = fhw & 7;
		int yoffset = Y_k + fh * SY; bool ly = LOAD_Y(tih, tiw, fh, fw);
		x.x = (ly ? deltaY[Y0 + yoffset] : 0);
		x.y = (ly ? deltaY[Y1 + yoffset] : 0);
		x.z = (ly ? deltaY[Y2 + yoffset] : 0);
		x.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k & OC_m1;
		int woffset = ((W_oc * 5 - fh) * 5 - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
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