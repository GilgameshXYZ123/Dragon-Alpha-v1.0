#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_A_S1_TEXTURE_H
#define DECONV3D_DX_ZERO_PADDING_KERNEL_A_S1_TEXTURE_H

//[Upgrade to A] Unsparse Matrix Method: s
//We have:x
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
//(6) N % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_A_S1_TEXTURE_CALL
#define DECONV3D_DX_ZERO_PADDING_KERNEL_A_S1_TEXTURE_CALL

//======[Common]==============================================
#define k88As1_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8A_s1_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, N,IC,OC, (FH-ph-1),(FW-1-pw),\
			ic_index,j_index)

#define k88As1_oc2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8A_s1_OC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, N,IC,LOC, (FH-ph-1),(FW-1-pw),\
			ic_index,j_index)

//======[FH = FW == 3]========================================
#define k88As1W3_oc2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8A_s1_W3_OC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,IH,IW, N,IC,LOC, (2-ph),(2-pw),\
			ic_index,j_index)

//======[FH = FW == 5]========================================
#define k88As1W5_oc2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8A_s1_W5_OC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (24*IC)), deltaX,IH,IW, N,IC,LOC, ph,pw,\
			ic_index,j_index)

#endif


//======[Common]==============================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_TEXTURE

//k88s1<4>: Size = 1.125, Time = 1.732 msec, Performace = 1394.87 GFlop/s
//k88s1<3>: Size = 1.125, Time = 2.096 msec, Performace = 1152.63 GFlop/s
//LB = 4: Size = 1.125, Time = 1.624 msec, Performace = 1487.63 GFlop/s
//LB = 3: Size = 1.125, Time = 1.898 msec, Performace = 1272.88 GFlop/s
//LB = 4: Size = 1, Time = 1.47 msec, Performace = 1460.87 GFlop/s
//LB = 3: Size = 1, Time = 1.708 msec, Performace = 1257.31 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8A_s1_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
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
	x.x = tex1Dfetch<float>(deltaY, Y0 + Y_oc);
	x.y = tex1Dfetch<float>(deltaY, Y1 + Y_oc);
	x.z = tex1Dfetch<float>(deltaY, Y2 + Y_oc);
	x.w = tex1Dfetch<float>(deltaY, Y3 + Y_oc);
	zero_float4(x, ly); Ys[buf][tx][ty] = x;
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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int fh = W_k / FW_OC, fw = (W_k -= fh * FW_OC) / OC;
		int W_oc = W_k - fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int yoffset = Y_k + fh * SY; bool ly = LOAD_Y(tih, tiw, fh, fw);
		x.x = tex1Dfetch<float>(deltaY, Y0 + yoffset);
		x.y = tex1Dfetch<float>(deltaY, Y1 + yoffset);
		x.z = tex1Dfetch<float>(deltaY, Y2 + yoffset);
		x.w = tex1Dfetch<float>(deltaY, Y3 + yoffset);
		zero_float4(x, ly); Ys[buf][tx][ty] = x;
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


//======[OC is power of 2]====================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_OC2POW_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_OC2POW_TEXTURE

//LB = 4: Size = 1.125, Time = 1.538 msec, Performace = 1570.82 GFlop/s
//LB = 3: Size = 1.125, Time = 1.782 msec, Performace = 1355.73 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8A_s1_OC2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
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
	const int FW_OC = FW << LOC, GK = FH * FW_OC;
	const int OC_m1 = (1 << LOC) - 1;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);//[fhr = fwr = 0]
	int woffset = W_oc * FH*FW*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = tx - ((tx >= STEP) << LB >> 1);//[fhr = fwr = 0]
	const int SY = (OW - FW) << LOC;
	float4 x; bool ly = LOAD_Y(tih, tiw, 0, 0);
	x.x = tex1Dfetch<float>(deltaY, Y0 + Y_oc);
	x.y = tex1Dfetch<float>(deltaY, Y1 + Y_oc);
	x.z = tex1Dfetch<float>(deltaY, Y2 + Y_oc);
	x.w = tex1Dfetch<float>(deltaY, Y3 + Y_oc);
	zero_float4(x, ly); Ys[buf][tx][ty] = x;
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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int fh = W_k / FW_OC, fw = (W_k - fh * FW_OC) >> LOC;
		int W_oc = W_k & OC_m1;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int yoffset = Y_k + fh * SY; bool ly = LOAD_Y(tih, tiw, fh, fw);
		x.x = tex1Dfetch<float>(deltaY, Y0 + yoffset);
		x.y = tex1Dfetch<float>(deltaY, Y1 + yoffset);
		x.z = tex1Dfetch<float>(deltaY, Y2 + yoffset);
		x.w = tex1Dfetch<float>(deltaY, Y3 + yoffset);
		zero_float4(x, ly); Ys[buf][tx][ty] = x;
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


//======[FH = FW == 3]=======================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W3_OC2POW_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W3_OC2POW_TEXTURE

//LB = 4: Size = 1.125, Time = 1.508 msec, Performace = 1602.07 GFlop/s
//LB = 3: Size = 1.125, Time = 1.712 msec, Performace = 1411.17 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8A_s1_W3_OC2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
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
	x.x = tex1Dfetch<float>(deltaY, Y0 + Y_oc);
	x.y = tex1Dfetch<float>(deltaY, Y1 + Y_oc);
	x.z = tex1Dfetch<float>(deltaY, Y2 + Y_oc);
	x.w = tex1Dfetch<float>(deltaY, Y3 + Y_oc);
	zero_float4(x, ly); Ys[buf][tx][ty] = x;
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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int Idx = W_k >> LOC; char fhw = YIDX_W33[Idx];
		int fh = fhw >> 2, fw = fhw & 3;
		int W_oc = W_k & OC_m1;
		int woffset = ((W_oc * 3 - fh) * 3 - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int yoffset = Y_k + fh * SY; bool ly = LOAD_Y(tih, tiw, fh, fw);
		x.x = tex1Dfetch<float>(deltaY, Y0 + yoffset);
		x.y = tex1Dfetch<float>(deltaY, Y1 + yoffset);
		x.z = tex1Dfetch<float>(deltaY, Y2 + yoffset);
		x.w = tex1Dfetch<float>(deltaY, Y3 + yoffset);
		zero_float4(x, ly); Ys[buf][tx][ty] = x;
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


//======[FH = FW == 5]=======================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W5_OC2POW_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8A_S1_W5_OC2POW_TEXTURE

//LB = 4: Size = 1.5625, Time = 2.016 msec, Performace = 1664.41 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.274 msec, Performace = 1475.57 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8A_s1_W5_OC2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 5
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int LOC,
	int ph, int pw,
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
	const int oph = 4 - ph, opw = 4 - pw;
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
	x.x = tex1Dfetch<float>(deltaY, Y0 + Y_oc);
	x.y = tex1Dfetch<float>(deltaY, Y1 + Y_oc);
	x.z = tex1Dfetch<float>(deltaY, Y2 + Y_oc);
	x.w = tex1Dfetch<float>(deltaY, Y3 + Y_oc);
	zero_float4(x, ly); Ys[buf][tx][ty] = x;
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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int Idx = W_k >> LOC; char fhw = YIDX_W55[Idx];
		int fh = fhw >> 3, fw = fhw & 7;
		int W_oc = W_k & OC_m1;
		int woffset = ((W_oc * 5 - fh) * 5 - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int yoffset = Y_k + fh * SY; bool ly = LOAD_Y(tih, tiw, fh, fw);
		x.x = tex1Dfetch<float>(deltaY, Y0 + yoffset);
		x.y = tex1Dfetch<float>(deltaY, Y1 + yoffset);
		x.z = tex1Dfetch<float>(deltaY, Y2 + yoffset);
		x.w = tex1Dfetch<float>(deltaY, Y3 + yoffset);
		zero_float4(x, ly); Ys[buf][tx][ty] = x;
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

#endif