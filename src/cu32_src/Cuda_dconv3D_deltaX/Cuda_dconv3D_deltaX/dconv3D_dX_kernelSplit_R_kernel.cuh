#pragma once

#ifndef DCONV3D_DX_KERNEL_SPLIT_R_KERNEL_H
#define DCONV3D_DX_KERNEL_SPLIT_R_KERNEL_H

//R: remode: W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]
//CWstride = CFH * CFW * OC * IC
#ifndef DCONV3D_DX_KERNEL_SPLIT_R_CALL
#define DCONV3D_DX_KERNEL_SPLIT_R_CALL

//LB = log2(BLOCK_SIZE)

#define ks88R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ks_kernel_8_8R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ks84R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ks_kernel_8_4R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>2), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ks48R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ks_kernel_4_8R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ks44R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ks_kernel_4_4R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>2), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ks42R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ks_kernel_4_2R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>1), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ks24R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ks_kernel_2_4R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>1), (GM>>LB>>1), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ks22R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ks_kernel_2_2R<LB, (1<<LB)>\
		<<< dim3((GN>>LB>>1), (GM>>LB>>1), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ks21R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ks_kernel_2_1R<LB, (1<<LB)>\
		<<< dim3((GN>>LB>>1), (GM>>LB), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ks12R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ks_kernel_1_2R<LB, (1<<LB)>\
		<<< dim3((GN>>LB), (GM>>LB>>1), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ks11R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ks_kernel_1_1R<LB, (1<<LB)>\
		<<< dim3((GN>>LB), (GM>>LB), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#endif


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_8_8R
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_8_8R

//LB = 4: Size = 1.05579, Time = 0.63 msec, Performace = 3598.86 GFlop/s
//LB = 3: Size = 1.05579, Time = 0.71 msec, Performace = 3193.36 GFlop/s
template<int LB, int STEP>
__global__ void ks_kernel_8_8R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index, j4 = j0 + 4;
	int IW_slice = KS_IW_slice(IW, sw), IW_slice_N = IW_slice * N;
	Ims_ih_iw_n(j0, ih0, iw0, n0);
	Ims_ih_iw_n(j4, ih4, iw4, n4);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC; bool wrt0 = WRT_X(ih0, iw0);
	int X4 = ((n4*IH + ih4)*IW + iw4)*IC; bool wrt4 = WRT_X(ih4, iw4);
	bool flagX = (tx >= STEP);
	int tn0 = IF_int(flagX, n4, n0);
	int tohs0 = (IF_int(flagX, ih4, ih0) + ph) / sh - oph;
	int tows0 = (IF_int(flagX, iw4, iw0) + pw) / sw - opw; 
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	Ys[buf][tx][ty] = KS_SaveYs4(deltaY, tohs0, tows0, Y_fhr, Y_fwr, OH, OW,
		yoffset, Y0, Y1, Y2, Y3);

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
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
		buf ^= 1;

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		Ys[buf][tx][ty] = KS_SaveYs4(deltaY, tohs0, tows0, Y_fhr, Y_fwr, OH, OW,
			yoffset, Y0, Y1, Y2, Y3);

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
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

	int Xstride = IH * IW * IC;

	float* dst0 = IF_int(wrt0, deltaX, HOLE);
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;
	X0 = IF_int(wrt0, X0, (X0 & 255));
	X1 = IF_int(wrt0, X1, (X1 & 255));
	X2 = IF_int(wrt0, X2, (X2 & 255));
	X3 = IF_int(wrt0, X3, (X3 & 255));
	*(float4*)(dst0 + X0) = v0;  *(float4*)(dst0 + X0 + 4) = v1;
	*(float4*)(dst0 + X1) = v2;  *(float4*)(dst0 + X1 + 4) = v3;
	*(float4*)(dst0 + X2) = v4;  *(float4*)(dst0 + X2 + 4) = v5;
	*(float4*)(dst0 + X3) = v6;  *(float4*)(dst0 + X3 + 4) = v7;

	float *dst4 = IF_int(wrt4, deltaX, HOLE);
	int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;
	X4 = IF_int(wrt4, X4, (X4 & 255));
	X5 = IF_int(wrt4, X5, (X5 & 255));
	X6 = IF_int(wrt4, X6, (X6 & 255));
	X7 = IF_int(wrt4, X7, (X7 & 255));
	*(float4*)(dst4 + X4) = v8;  *(float4*)(dst4 + X4 + 4) = v9;
	*(float4*)(dst4 + X5) = v10; *(float4*)(dst4 + X5 + 4) = v11;
	*(float4*)(dst4 + X6) = v12; *(float4*)(dst4 + X6 + 4) = v13;
	*(float4*)(dst4 + X7) = v14; *(float4*)(dst4 + X7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*4), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_8_4R
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_8_4R

//LB = 4: Size = 1.05579, Time = 0.69  msec, Performace = 3285.92 GFlop/s
//LB = 3: Size = 1.05579, Time = 0.778 msec, Performace = 2914.25 GFlop/s
template<int LB, int STEP>
__global__ void ks_kernel_8_4R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int IW_slice = KS_IW_slice(IW, sw), IW_slice_N = IW_slice * N;
	Ims_ih_iw_n(j0, ih0, iw0, n0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC; bool wrt0 = WRT_X(ih0, iw0);
	int tn0 = n0 + ((tx & 1) << 1);
	int tohs0 = (ih0 + ph) / sh - oph;
	int tows0 = (iw0 + pw) / sw - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;
	int Y1 = Y0 + OH * OW * OC;

	//load 2 elem from deltaY
	int Y_k = tx >> 1;
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = KS_SaveYs2(deltaY, tohs0, tows0, Y_fhr, Y_fwr, OH, OW,
		yoffset, Y0, Y1);

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
	__syncthreads();

	//compute area-----------------------------------------------------
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

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		}
		buf ^= 1;

		//load 2 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		Ys[buf][Ys_x][Ys_y] = KS_SaveYs2(deltaY, tohs0, tows0, Y_fhr, Y_fwr, OH, OW,
			yoffset, Y0, Y1);

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
	}

	int Xstride = IH * IW * IC;

	float* dst0 = IF_int(wrt0, deltaX, HOLE);
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;
	X0 = IF_int(wrt0, X0, (X0 & 255));
	X1 = IF_int(wrt0, X1, (X1 & 255));
	X2 = IF_int(wrt0, X2, (X2 & 255));
	X3 = IF_int(wrt0, X3, (X3 & 255));
	*(float4*)(dst0 + X0) = v0;  *(float4*)(dst0 + X0 + 4) = v1;
	*(float4*)(dst0 + X1) = v2;  *(float4*)(dst0 + X1 + 4) = v3;
	*(float4*)(dst0 + X2) = v4;  *(float4*)(dst0 + X2 + 4) = v5;
	*(float4*)(dst0 + X3) = v6;  *(float4*)(dst0 + X3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_4_8R
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_4_8R

//LB = 4: Size = 1.05579, Time = 0.768 msec, Performace = 2952.19 GFlop/s
//LB = 3: Size = 1.05579, Time = 0.83  msec, Performace = 2731.67 GFlop/s
template<int LB, int STEP>
__global__ void ks_kernel_4_8R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index, j4 = j0 + 4;
	int IW_slice = KS_IW_slice(IW, sw), IW_slice_N = IW_slice * N;
	Ims_ih_iw_n(j0, ih0, iw0, n0);
	Ims_ih_iw_n(j4, ih4, iw4, n4);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC; bool wrt0 = WRT_X(ih0, iw0);
	int X4 = ((n4*IH + ih4)*IW + iw4)*IC; bool wrt4 = WRT_X(ih4, iw4);
	bool flagX = (tx >= STEP);
	int tn0 = IF_int(flagX, n4, n0);
	int tohs0 = (IF_int(flagX, ih4, ih0) + ph) / sh - oph;
	int tows0 = (IF_int(flagX, iw4, iw0) + pw) / sw - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	Ys[buf][tx][ty] = KS_SaveYs4(deltaY, tohs0, tows0, Y_fhr, Y_fwr, OH, OW,
		yoffset, Y0, Y1, Y2, Y3);

	//load 2 elem from W
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

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

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		Ys[buf][tx][ty] = KS_SaveYs4(deltaY, tohs0, tows0, Y_fhr, Y_fwr, OH, OW,
			yoffset, Y0, Y1, Y2, Y3);

		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

		simdMM4(v0, y0.x, w0);
		simdMM4(v2, y0.y, w0);
		simdMM4(v4, y0.z, w0);
		simdMM4(v6, y0.w, w0);
		simdMM4(v8, y1.x, w0);
		simdMM4(v10, y1.y, w0);
		simdMM4(v12, y1.z, w0);
		simdMM4(v14, y1.w, w0);
	}

	int Xstride = IH * IW * IC;

	float* dst0 = IF_int(wrt0, deltaX, HOLE);
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;
	X0 = IF_int(wrt0, X0, (X0 & 255));
	X1 = IF_int(wrt0, X1, (X1 & 255));
	X2 = IF_int(wrt0, X2, (X2 & 255));
	X3 = IF_int(wrt0, X3, (X3 & 255));
	*(float4*)(dst0 + X0) = v0; 
	*(float4*)(dst0 + X1) = v2; 
	*(float4*)(dst0 + X2) = v4;
	*(float4*)(dst0 + X3) = v6;

	float *dst4 = IF_int(wrt4, deltaX, HOLE);
	int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;
	X4 = IF_int(wrt4, X4, (X4 & 255));
	X5 = IF_int(wrt4, X5, (X5 & 255));
	X6 = IF_int(wrt4, X6, (X6 & 255));
	X7 = IF_int(wrt4, X7, (X7 & 255));
	*(float4*)(dst4 + X4) = v8; 
	*(float4*)(dst4 + X5) = v10;
	*(float4*)(dst4 + X6) = v12;
	*(float4*)(dst4 + X7) = v14;
}

#endif


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*4), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_4_4R
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_4_4R

//LB = 4: Size = 1.05579, Time = 0.792 msec, Performace = 2862.73 GFlop/s
//LB = 3: Size = 1.05579, Time = 1.03  msec, Performace = 2201.25 GFlop/s
template<int LB, int STEP>
__global__ void ks_kernel_4_4R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int IW_slice = KS_IW_slice(IW, sw), IW_slice_N = IW_slice * N;
	Ims_ih_iw_n(j0, ih0, iw0, n0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC; bool wrt0 = WRT_X(ih0, iw0);
	int tn0 = n0 + ((tx & 1) << 1);
	int tohs0 = (ih0 + ph) / sh - oph;
	int tows0 = (iw0 + pw) / sw - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;
	int Y1 = Y0 + OH * OW * OC;

	//load 2 elem from deltaY
	int Y_k = tx >> 1;
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = KS_SaveYs2(deltaY, tohs0, tows0, Y_fhr, Y_fwr, OH, OW,
		yoffset, Y0, Y1);
	
	//load 2 elem from W
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
	__syncthreads();

	//compute area-------------------------------------------------
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

			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w);
			simdMM4(v2, y.z, w);
			simdMM4(v3, y.w, w);
		}
		buf ^= 1;

		//load 2 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		Ys[buf][Ys_x][Ys_y] = KS_SaveYs2(deltaY, tohs0, tows0, Y_fhr, Y_fwr, OH, OW,
			yoffset, Y0, Y1);
		
		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

		simdMM4(v0, y.x, w);
		simdMM4(v1, y.y, w);
		simdMM4(v2, y.z, w);
		simdMM4(v3, y.w, w);
	}
	
	float* dst0 = IF_int(wrt0, deltaX, HOLE);
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;
	X0 = IF_int(wrt0, X0, (X0 & 255));
	X1 = IF_int(wrt0, X1, (X1 & 255));
	X2 = IF_int(wrt0, X2, (X2 & 255));
	X3 = IF_int(wrt0, X3, (X3 & 255));
	
	*(float4*)(dst0 + X0) = v0;
	*(float4*)(dst0 + X1) = v1;
	*(float4*)(dst0 + X2) = v2;
	*(float4*)(dst0 + X3) = v3;
	
}

#endif


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*2), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_4_2R
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_4_2R

//LB = 4: Size = 1.05579, Time = 1.154 msec, Performace = 1964.72 GFlop/s
//LB = 3: Size = 1.05579, Time = 1.564 msec, Performace = 1449.67 GFlop/s
template<int LB, int STEP>
__global__ void ks_kernel_4_2R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index;
	int IW_slice = KS_IW_slice(IW, sw), IW_slice_N = IW_slice * N;
	Ims_ih_iw_n(j0, ih0, iw0, n0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC; bool wrt0 = WRT_X(ih0, iw0);
	int tn0 = n0 + (tx & 1);
	int tohs0 = (ih0 + ph) / sh - oph;
	int tows0 = (iw0 + pw) / sw - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;

	//load 1 elem from deltaY
	int Y_k = tx >> 1;
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	Ys[buf][Ys_x][Ys_y] = (ly0 ? deltaY[Y0 + yoffset] : 0);

	//load 2 elem from W
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
	__syncthreads();

	//compute area-------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 y = *(float2*)(&Ys[buf][ik][ty << 1]);
			float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);
			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w);
		}
		buf ^= 1;

		//load 1 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		Ys[buf][Ys_x][Ys_y] = (ly0 ? deltaY[Y0 + yoffset] : 0);

		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 y = *(float2*)(&Ys[buf][ik][ty << 1]);
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);
		simdMM4(v0, y.x, w);
		simdMM4(v1, y.y, w);
	}

	int X1 = X0 + IH * IW * IC;
	if (wrt0) {
		*(float4*)(deltaX + X0) = v0;
		*(float4*)(deltaX + X1) = v1;
	}
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*4), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_2_4R
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_2_4R

//LB = 4: Size = 1.05579, Time = 2.102 msec, Performace = 1078.63  GFlop/s
//LB = 3: Size = 1.05579, Time = 3.072 msec, Performace =  738.048 GFlop/s
template<int LB, int STEP>
__global__ void ks_kernel_2_4R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	CW += ic0 + (ty & 1);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int IW_slice = KS_IW_slice(IW, sw), IW_slice_N = IW_slice * N;
	Ims_ih_iw_n(j0, ih0, iw0, n0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC; bool wrt0 = WRT_X(ih0, iw0);
	int tn0 = n0 + ((tx & 1) << 1);
	int tohs0 = (ih0 + ph) / sh - oph;
	int tows0 = (iw0 + pw) / sw - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;
	int Y1 = Y0 + OH * OW * OC;

	//load 2 elem from deltaY
	int Y_k = tx >> 1;
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = KS_SaveYs2(deltaY, tohs0, tows0, Y_fhr, Y_fwr, OH, OW,
		yoffset, Y0, Y1);

	//load 1 elem from W
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = CW[W_k * IC];
	__syncthreads();

	//compute area-------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
			float2 w = *(float2*)(&Ws[buf][ik][tx << 1]);

			simdMM2(v0, y.x, w);
			simdMM2(v1, y.y, w);
			simdMM2(v2, y.z, w);
			simdMM2(v3, y.w, w);
		}
		buf ^= 1;

		//load 2 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		Ys[buf][Ys_x][Ys_y] = KS_SaveYs2(deltaY, tohs0, tows0, Y_fhr, Y_fwr, OH, OW,
			yoffset, Y0, Y1);

		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = CW[W_k * IC];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
		float2 w = *(float2*)(&Ws[buf][ik][tx << 1]);

		simdMM2(v0, y.x, w);
		simdMM2(v1, y.y, w);
		simdMM2(v2, y.z, w);
		simdMM2(v3, y.w, w);
	}

	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;
	if (wrt0) {
		*(float2*)(deltaX + X0) = v0;
		*(float2*)(deltaX + X1) = v1;
		*(float2*)(deltaX + X2) = v2;
		*(float2*)(deltaX + X3) = v3;
	}
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*2), OC >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_2_2R
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_2_2R

//LB = 4: Size = 1.05579, Time = 1.626 msec, Performace = 1394.39 GFlop/s
//LB = 3: Size = 1.05579, Time = 2.2   msec, Performace = 1030.58 GFlop/s
//LB = 4: Size = 1.02279, Time = 2.714 msec, Performace = 809.296 GFlop/s
//LB = 3: Size = 1.02279, Time = 2.602 msec, Performace = 844.132 GFlop/s
template<int LB, int STEP>
__global__ void ks_kernel_2_2R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	CW += ic0;//CW[0, 0, 0, ic0]
	deltaX += ic0;//deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index, j1 = j0 + 1;
	int IW_slice = KS_IW_slice(IW, sw), IW_slice_N = IW_slice * N;
	Ims_ih_iw_n(j0, ih0, iw0, n0);
	Ims_ih_iw_n(j1, ih1, iw1, n1);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC; bool wrt0 = WRT_X(ih0, iw0);
	int X1 = ((n1*IH + ih1)*IW + iw1)*IC; bool wrt1 = WRT_X(ih1, iw1);
	int ohs0 = (ih0 + ph) / sh - oph, ows0 = (iw0 + pw) / sw - opw;
	int ohs1 = (ih1 + ph) / sh - oph, ows1 = (iw1 + pw) / sw - opw;
	int Y0 = ((n0*OH + ohs0)*OW + ows0) * OC;
	int Y1 = ((n1*OH + ohs1)*OW + ows1) * OC;

	//load 2 elem from W
	int W_k = ty;
	Ws[buf][ty][tx] = *(float2*)(CW + W_k * IC);

	//load 2 elem from deltaY
	int Y_k = tx;
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
	bool ly1 = (ohs1 >= -Y_fhr) && (ohs1 < OH - Y_fhr) && (ows1 >= -Y_fwr) && (ows1 < OW - Y_fwr);
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 y = Ys[buf][ik][ty];
			float2 w = Ws[buf][ik][tx];
			simdMM2(v0, y.x, w);
			simdMM2(v1, y.y, w);
		}
		buf ^= 1;

		//load 2 elem from W
		int W_k = (ok << LB) + ty;
		Ws[buf][ty][tx] = *(float2*)(CW + W_k * IC);

		//load 2 elem from Y
		int Y_k = (ok << LB) + tx;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
		bool ly1 = (ohs1 >= -Y_fhr) && (ohs1 < OH - Y_fhr) && (ows1 >= -Y_fwr) && (ows1 < OW - Y_fwr);
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 y = Ys[buf][ik][ty];
		float2 w = Ws[buf][ik][tx];
		simdMM2(v0, y.x, w);
		simdMM2(v1, y.y, w);
	}

	//when GK % STEP != 0-------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		//load 2 elem from W
		float2 w = *(float2*)(CW + k * IC);

		//load 2 elem from Y
		float2 y; int Y_k = k;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
		bool ly1 = (ohs1 >= -Y_fhr) && (ohs1 < OH - Y_fhr) && (ows1 >= -Y_fwr) && (ows1 < OW - Y_fwr);
		y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);

		simdMM2(v0, y.x, w);
		simdMM2(v1, y.y, w);
	}
	//when GK % STEP != 0-------------------------------------------

	if (wrt0) *(float2*)(deltaX + X0) = v0;
	if (wrt1) *(float2*)(deltaX + X1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*1), OC >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_2_1R
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_2_1R

//LB = 4: Size = 1.02279, Time = 4.084 msec, Performace = 537.814 GFlop/s
//LB = 3: Size = 1.02279, Time = 3.99  msec, Performace = 550.484 GFlop/s
template<int LB, int STEP>
__global__ void ks_kernel_2_1R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	CW += ic0;//CW[0, 0, 0, ic0]
	deltaX += ic0;//deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) + j_index;
	int IW_slice = KS_IW_slice(IW, sw), IW_slice_N = IW_slice * N;
	Ims_ih_iw_n(j0, ih0, iw0, n0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC; bool wrt0 = WRT_X(ih0, iw0);
	int ohs0 = (ih0 + ph) / sh - oph, ows0 = (iw0 + pw) / sw - opw;
	int Y0 = ((n0*OH + ohs0)*OW + ows0) * OC;

	//load 2 elem from W
	int W_k = ty;
	Ws[buf][ty][tx] = *(float2*)(CW + W_k * IC);

	//load 2 elem from deltaY
	int Y_k = tx;
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
	Ys[buf][tx][ty] = (ly0 ? deltaY[Y0 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  y = Ys[buf][ik][ty];
			float2 w = Ws[buf][ik][tx];
			simdMM2(v0, y, w);
		}
		buf ^= 1;

		//load 2 elem from W
		int W_k = (ok << LB) + ty;
		Ws[buf][ty][tx] = *(float2*)(CW + W_k * IC);

		//load 1 elem from Y
		int Y_k = (ok << LB) + tx;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
		Ys[buf][tx][ty] = (ly0 ? deltaY[Y0 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  y = Ys[buf][ik][ty];
		float2 w = Ws[buf][ik][tx];
		simdMM2(v0, y, w);
	}

	//when GK % STEP != 0-------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		//load 2 elem from W
		float2 w = *(float2*)(CW + k * IC);

		//load 1 elem from Y
		float y; int Y_k = k;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
		y = (ly0 ? deltaY[Y0 + yoffset] : 0);

		simdMM2(v0, y, w);
	}
	//when GK % STEP != 0-------------------------------------------

	if (wrt0) *(float2*)(deltaX + X0) = v0;
}

#endif


//(Y: BLOCK_SIZE*1, X:BLOCK_SIZE*2), OC >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_1_2R
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_1_2R

//LB = 4: Size = 1.02279, Time = 4.81  msec, Performace = 456.638 GFlop/s
//LB = 3: Size = 1.02279, Time = 4.618 msec, Performace = 475.624 GFlop/s
template<int LB, int STEP>
__global__ void ks_kernel_1_2R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) + ic_index;
	CW += ic0;//CW[0, 0, 0, ic0]
	deltaX += ic0;//deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index, j1 = j0 + 1;
	int IW_slice = KS_IW_slice(IW, sw), IW_slice_N = IW_slice * N;
	Ims_ih_iw_n(j0, ih0, iw0, n0);
	Ims_ih_iw_n(j1, ih1, iw1, n1);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC; bool wrt0 = WRT_X(ih0, iw0);
	int X1 = ((n1*IH + ih1)*IW + iw1)*IC; bool wrt1 = WRT_X(ih1, iw1);
	int ohs0 = (ih0 + ph) / sh - oph, ows0 = (iw0 + pw) / sw - opw;
	int ohs1 = (ih1 + ph) / sh - oph, ows1 = (iw1 + pw) / sw - opw;
	int Y0 = ((n0*OH + ohs0)*OW + ows0) * OC;
	int Y1 = ((n1*OH + ohs1)*OW + ows1) * OC;

	//load 1 elem from W
	int W_k = ty;
	Ws[buf][ty][tx] = CW[W_k * IC];

	//load 2 elem from deltaY
	int Y_k = tx;
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
	bool ly1 = (ohs1 >= -Y_fhr) && (ohs1 < OH - Y_fhr) && (ows1 >= -Y_fwr) && (ows1 < OW - Y_fwr);
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 y = Ys[buf][ik][ty];
			float  w = Ws[buf][ik][tx];
			simdMM2(v0, w, y);
		}
		buf ^= 1;

		//load 1 elem from W
		int W_k = (ok << LB) + ty;
		Ws[buf][ty][tx] = CW[W_k * IC];

		//load 2 elem from Y
		int Y_k = (ok << LB) + tx;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
		bool ly1 = (ohs1 >= -Y_fhr) && (ohs1 < OH - Y_fhr) && (ows1 >= -Y_fwr) && (ows1 < OW - Y_fwr);
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 y = Ys[buf][ik][ty];
		float  w = Ws[buf][ik][tx];
		simdMM2(v0, w, y);
	}

	//when GK % STEP != 0-------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		//load 1 elem from W
		float w = CW[k * IC];

		//load 2 elem from Y
		float2 y; int Y_k = k;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
		bool ly1 = (ohs1 >= -Y_fhr) && (ohs1 < OH - Y_fhr) && (ows1 >= -Y_fwr) && (ows1 < OW - Y_fwr);
		y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);

		simdMM2(v0, w, y);
	}
	//when GK % STEP != 0-------------------------------------------

	if (wrt0) deltaX[X0] = v0.x;
	if (wrt1) deltaX[X1] = v0.y;
}

#endif


//(Y: BLOCK_SIZE*1, X:BLOCK_SIZE*1), OC >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_1_1R
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_1_1R

//LB = 4: Size = 1.02279, Time = 7.682 msec, Performace = 285.919 GFlop/s
//LB = 3: Size = 1.02279, Time = 7.36  msec, Performace = 298.428 GFlop/s
template<int LB, int STEP>
__global__ void ks_kernel_1_1R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) + ic_index;
	CW += ic0;//CW[0, 0, 0, ic0]
	deltaX += ic0;//deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) + j_index;
	int IW_slice = KS_IW_slice(IW, sw), IW_slice_N = IW_slice * N;
	Ims_ih_iw_n(j0, ih0, iw0, n0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC; bool wrt0 = WRT_X(ih0, iw0);
	int ohs0 = (ih0 + ph) / sh - oph, ows0 = (iw0 + pw) / sw - opw;
	int Y0 = ((n0*OH + ohs0)*OW + ows0) * OC;

	//load 1 elem from W
	int W_k = ty;
	Ws[buf][ty][tx] = CW[W_k * IC];

	//load 1 elem from deltaY
	int Y_k = tx;
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
	Ys[buf][tx][ty] = (ly0 ? deltaY[Y0 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
	float v = 0;
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float y = Ys[buf][ik][ty];
			float w = Ws[buf][ik][tx];
			v += y * w;
		}
		buf ^= 1;

		//load 1 elem from W
		int W_k = (ok << LB) + ty;
		Ws[buf][ty][tx] = CW[W_k * IC];

		//load 1 elem from Y
		int Y_k = (ok << LB) + tx;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
		Ys[buf][tx][ty] = (ly0 ? deltaY[Y0 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float y = Ys[buf][ik][ty];
		float w = Ws[buf][ik][tx];
		v += y * w;
	}

	//when GK % STEP != 0-------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		//load 1 elem from W
		float w = CW[k * IC];

		//load 1 elem from Y
		float y; int Y_k = k;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
		y = (ly0 ? deltaY[Y0 + yoffset] : 0);

		v += y * w;
	}
	//when GK % STEP != 0-------------------------------------------

	if (wrt0) deltaX[X0] = v;
}

#endif

#endif