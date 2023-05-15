#pragma once

#ifndef DCONV3D_DX_KERNEL_SPLIT_IMSR_KERNEL_H
#define DCONV3D_DX_KERNEL_SPLIT_IMSR_KERNEL_H

//IMS: INPUT_MOD_STEP
//R: remode: W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]
//(IH, IW) % (sh, sw) == 0
//CWstride = CFH * CFW * OC * IC
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMSR_CALL
#define DCONV3D_DX_KERNEL_SPLIT_IMSR_CALL

#define ksIms_88R8(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_8_8R8<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ksIms_88R4(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_8_8R4<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ksIms_88R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_8_8R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

//=====================================================================================
#define ksIms_84R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_8_4R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>2), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ksIms_48R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_4_8R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ksIms_44R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_4_4R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>2), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ksIms_42R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_4_2R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>1), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ksIms_24R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_2_4R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>1), (GM>>LB>>2), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ksIms_22R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_2_2R<LB, (1<<LB)>\
		<<< dim3((GN>>LB>>1), (GM>>LB>>1), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ksIms_21R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_2_1R<LB, (1<<LB)>\
		<<< dim3((GN>>LB>>1), (GM>>LB), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ksIms_12R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_1_2R<LB, (1<<LB)>\
		<<< dim3((GN>>LB), (GM>>LB>>1), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#define ksIms_11R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_1_1R<LB, (1<<LB)>\
		<<< dim3((GN>>LB), (GM>>LB), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, sh, sw, ph, pw, ic_index, j_index)

#endif


//(IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_8R8
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_8R8

//LB = 4: Size = 1.53125, Time = 0.706 msec, Performace = 4657.7 GFlop/s
//LB = 4: Size = 1.125, Time = 0.638 msec, Performace = 3786.71 GFlop/s
//LB = 3: Size = 1.125, Time = 0.696 msec, Performace = 3471.15 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_8_8R8(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
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

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = ((n0*IH_IW_slice*sh*sw) + ih0 * IW_slice*sw + iw0)*IC + ic0;
	int tohs0 = (ih0 + ph) / sh - oph;
	int tows0 = (iw0 + ((tx >= STEP) << 3) + pw) / sw - opw;
	int tows1 = tows0 + 1, tows2 = tows0 + 2, tows3 = tows0 + 3;
	int Y1 = ((n0*OH + tohs0)*OW + tows1) * OC;
	OH -= tohs0;//OH = OH - tohs0 = OH_m_tohs0

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int OW_OC = OW * OC;
	Ys[buf][tx][ty] = Ims4x_loadYs4(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
		Y1, tohs0, tows0, tows1, tows2, tows3);
	__syncthreads();

	//compute area-------------------------------------------------
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
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

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

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = Ims4x_loadYs4(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
			Y1, tohs0, tows0, tows1, tows2, tows3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	IC *= sw;
	int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	int X4 = X3 + IC, X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

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


//(IH_slice, IW_slice) % 4 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_8R4
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_8R4

//LB = 4: Size = 1.53125, Time = 0.706 msec, Performace = 4657.7 GFlop/s
//LB = 4: Size = 1.125, Time = 0.644 msec, Performace = 3751.43 GFlop/s
//LB = 3: Size = 1.125, Time = 0.698 msec, Performace = 3461.2 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_8_8R4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
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

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index, j4 = j0 + 4;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j4, n4, ih4, iw4);
	IH_IW_slice *= sh * sw; //IH_IW_slice -> IH * IW
	IW_slice *= sw;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih0 * IW_slice + iw0)*IC + ic0;
	int X4 = ((n4*IH_IW_slice) + ih4 * IW_slice + iw4)*IC + ic0;

	bool flagX = (tx >= STEP);
	int tohs0 = (IF_int(flagX, ih4, ih0) + ph) / sh - oph;
	int tows0 = (IF_int(flagX, iw4, iw0) + pw) / sw - opw;
	int tows1 = tows0 + 1, tows2 = tows0 + 2, tows3 = tows0 + 3;
	int Y1 = ((n0*OH + tohs0)*OW + tows1) * OC;
	OH -= tohs0;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int OW_OC = OW * OC;
	Ys[buf][tx][ty] = Ims4x_loadYs4(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
		Y1, tohs0, tows0, tows1, tows2, tows3);
	__syncthreads();

	//compute area-------------------------------------------------
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
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

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

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = Ims4x_loadYs4(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
			Y1, tohs0, tows0, tows1, tows2, tows3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	IC *= sw;
	int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	int X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_8R
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_8R

//LB = 4: Size = 1.53125, Time = 0.738 msec, Performace = 4455.74 GFlop/s
//LB = 4: Size = 1.125, Time = 0.672 msec, Performace = 3595.12 GFlop/s
//LB = 3: Size = 1.125, Time = 0.724 msec, Performace = 3336.9 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_8_8R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
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

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	Ims_n_ih_iw(tj0, tn0, tohs0, tows0);
	Ims_n_ih_iw(tj1, tn1, tohs1, tows1);
	Ims_n_ih_iw(tj2, tn2, tohs2, tows2);
	Ims_n_ih_iw(tj3, tn3, tohs3, tows3);
	tohs0 = (tohs0 + ph) / sh - oph, tows0 = (tows0 + pw) / sw - opw;
	tohs1 = (tohs1 + ph) / sh - oph, tows1 = (tows1 + pw) / sw - opw;
	tohs2 = (tohs2 + ph) / sh - oph, tows2 = (tows2 + pw) / sw - opw;
	tohs3 = (tohs3 + ph) / sh - oph, tows3 = (tows3 + pw) / sw - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((tn1*OH + tohs1)*OW + tows1) * OC;
	int Y2 = ((tn2*OH + tohs2)*OW + tows2) * OC;
	int Y3 = ((tn3*OH + tohs3)*OW + tows3) * OC;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int OW_OC = OW * OC;
	Ys[buf][tx][ty] = Ims_loadYs4(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
		Y0, tohs0, tows0,
		Y1, tohs1, tows1,
		Y2, tohs2, tows2,
		Y3, tohs3, tows3);
	__syncthreads();

	//compute area-------------------------------------------------
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
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

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

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);   

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = Ims_loadYs4(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
			Y0, tohs0, tows0,
			Y1, tohs1, tows1,
			Y2, tohs2, tows2,
			Y3, tohs3, tows3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	deltaX += ic0 + (ihs*sw*IW_slice + iws)*IC; IC *= sw;
	int alpha = sh * IC, X0 = alpha * j0;
	int X1 = X0 + alpha, X2 = X1 + alpha, X3 = X2 + alpha;
	int X4 = X3 + alpha, X5 = X4 + alpha, X6 = X5 + alpha, X7 = X6 + alpha;
	int beta = IC - alpha;//(1 - sh)*IC = IC - sh*IC = IC - alpha
	X0 += beta * ((j0    ) % IW_slice);
	X1 += beta * ((j0 + 1) % IW_slice);
	X2 += beta * ((j0 + 2) % IW_slice);
	X3 += beta * ((j0 + 3) % IW_slice);
	X4 += beta * ((j0 + 4) % IW_slice);
	X5 += beta * ((j0 + 5) % IW_slice);
	X6 += beta * ((j0 + 6) % IW_slice);
	X7 += beta * ((j0 + 7) % IW_slice);

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

//==============================================================
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*4), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_4R
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_4R

//LB = 4: Size = 1.53125, Time = 0.908 msec, Performace = 3621.51 GFlop/s
//LB = 4: Size = 1.125, Time = 0.724 msec, Performace = 3336.9 GFlop/s
//LB = 3: Size = 1.125, Time = 0.852 msec, Performace = 2835.59 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_8_4R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

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
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	CW += ic0 + ((ty >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 2;
	int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	Ims_n_ih_iw(tj0, tn0, tohs0, tows0);
	Ims_n_ih_iw(tj1, tn1, tohs1, tows1);
	tohs0 = (tohs0 + ph) / sh - oph, tows0 = (tows0 + pw) / sw - opw;
	tohs1 = (tohs1 + ph) / sh - oph, tows1 = (tows1 + pw) / sw - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((tn1*OH + tohs1)*OW + tows1) * OC;

	//load 2 elem from deltaY
	int Y_k = tx >> 1;
	int OW_OC = OW * OC;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = Ims_loadYs2(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
		Y0, tohs0, tows0,
		Y1, tohs1, tows1);

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
	__syncthreads();

	//compute area-------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		}
		buf ^= 1;

		//load 2 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][Ys_x][Ys_y] = Ims_loadYs2(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
			Y0, tohs0, tows0,
			Y1, tohs1, tows1);

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
	}

	deltaX += ic0 + (ihs*sw*IW_slice + iws)*IC; IC *= sw;
	int alpha = sh * IC, X0 = alpha * j0;
	int X1 = X0 + alpha, X2 = X1 + alpha, X3 = X2 + alpha;
	int beta = IC - alpha;//(1 - sh)*IC = IC - sh*IC = IC - alpha
	X0 += beta * ((j0) % IW_slice);
	X1 += beta * ((j0 + 1) % IW_slice);
	X2 += beta * ((j0 + 2) % IW_slice);
	X3 += beta * ((j0 + 3) % IW_slice);

	*(float4*)(deltaX + X0) = v0; *(float4*)(deltaX + X0 + 4) = v1; 
	*(float4*)(deltaX + X1) = v2; *(float4*)(deltaX + X1 + 4) = v3; 
	*(float4*)(deltaX + X2) = v4; *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6; *(float4*)(deltaX + X3 + 4) = v7; 
}

#endif


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_4_8R
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_4_8R

//LB = 4: Size = 1.53125, Time = 0.868 msec, Performace = 3788.4 GFlop/s
//LB = 4: Size = 1.125, Time = 0.736 msec, Performace = 3282.5 GFlop/s
//LB = 3: Size = 1.125, Time = 0.888 msec, Performace = 2720.63 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_4_8R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//followed k8

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

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	Ims_n_ih_iw(tj0, tn0, tohs0, tows0);
	Ims_n_ih_iw(tj1, tn1, tohs1, tows1);
	Ims_n_ih_iw(tj2, tn2, tohs2, tows2);
	Ims_n_ih_iw(tj3, tn3, tohs3, tows3);
	tohs0 = (tohs0 + ph) / sh - oph, tows0 = (tows0 + pw) / sw - opw;
	tohs1 = (tohs1 + ph) / sh - oph, tows1 = (tows1 + pw) / sw - opw;
	tohs2 = (tohs2 + ph) / sh - oph, tows2 = (tows2 + pw) / sw - opw;
	tohs3 = (tohs3 + ph) / sh - oph, tows3 = (tows3 + pw) / sw - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((tn1*OH + tohs1)*OW + tows1) * OC;
	int Y2 = ((tn2*OH + tohs2)*OW + tows2) * OC;
	int Y3 = ((tn3*OH + tohs3)*OW + tows3) * OC;

	//load 2 elem from W
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int OW_OC = OW * OC;
	Ys[buf][tx][ty] = Ims_loadYs4(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
		Y0, tohs0, tows0,
		Y1, tohs1, tows1,
		Y2, tohs2, tows2,
		Y3, tohs3, tows3);
	__syncthreads();

	//compute area-------------------------------------------------
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
			float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

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

		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = Ims_loadYs4(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
			Y0, tohs0, tows0,
			Y1, tohs1, tows1,
			Y2, tohs2, tows2,
			Y3, tohs3, tows3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0);
		simdMM4(v2, y0.y, w0);
		simdMM4(v4, y0.z, w0);
		simdMM4(v6, y0.w, w0);
		simdMM4(v8, y1.x, w0);
		simdMM4(v10, y1.y, w0);
		simdMM4(v12, y1.z, w0);
		simdMM4(v14, y1.w, w0);
	}

	deltaX += ic0 + (ihs*sw*IW_slice + iws)*IC; IC *= sw;
	int alpha = sh * IC, X0 = alpha * j0;
	int X1 = X0 + alpha, X2 = X1 + alpha, X3 = X2 + alpha;
	int X4 = X3 + alpha, X5 = X4 + alpha, X6 = X5 + alpha, X7 = X6 + alpha;
	int beta = IC - alpha;//(1 - sh)*IC = IC - sh*IC = IC - alpha
	X0 += beta * ((j0) % IW_slice);
	X1 += beta * ((j0 + 1) % IW_slice);
	X2 += beta * ((j0 + 2) % IW_slice);
	X3 += beta * ((j0 + 3) % IW_slice);
	X4 += beta * ((j0 + 4) % IW_slice);
	X5 += beta * ((j0 + 5) % IW_slice);
	X6 += beta * ((j0 + 6) % IW_slice);
	X7 += beta * ((j0 + 7) % IW_slice);

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


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*4), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_4_4R
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_4_4R

//LB = 4: Size = 1.53125, Time = 1.108 msec, Performace = 2967.81 GFlop/s
//LB = 4: Size = 1.125, Time = 0.89  msec, Performace = 2714.52 GFlop/s
//LB = 3: Size = 1.125, Time = 1.112 msec, Performace = 2172.59 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_4_4R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
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

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	Ims_n_ih_iw(tj0, tn0, tohs0, tows0);
	Ims_n_ih_iw(tj1, tn1, tohs1, tows1);
	tohs0 = (tohs0 + ph) / sh - oph, tows0 = (tows0 + pw) / sw - opw;
	tohs1 = (tohs1 + ph) / sh - oph, tows1 = (tows1 + pw) / sw - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((tn1*OH + tohs1)*OW + tows1) * OC;

	//load 2 elem from W
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);

	//load 2 elem from deltaY
	int Y_k = tx >> 1;
	int OW_OC = OW * OC;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = Ims_loadYs2(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
		Y0, tohs0, tows0,
		Y1, tohs1, tows1);
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

		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);

		//load 2 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][Ys_x][Ys_y] = Ims_loadYs2(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
			Y0, tohs0, tows0,
			Y1, tohs1, tows1);
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

	deltaX += ic0 + (ihs*sw*IW_slice + iws)*IC; IC *= sw;
	int alpha = sh * IC, X0 = alpha * j0;
	int X1 = X0 + alpha, X2 = X1 + alpha, X3 = X2 + alpha;
	int beta = IC - alpha;//(1 - sh)*IC = IC - sh*IC = IC - alpha
	X0 += beta * ((j0    ) % IW_slice);
	X1 += beta * ((j0 + 1) % IW_slice);
	X2 += beta * ((j0 + 2) % IW_slice);
	X3 += beta * ((j0 + 3) % IW_slice);

	*(float4*)(deltaX + X0) = v0;  
	*(float4*)(deltaX + X1) = v1;
	*(float4*)(deltaX + X2) = v2; 
	*(float4*)(deltaX + X3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*2), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_4_2R
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_4_2R

//LB = 4: Size = 1.53125, Time = 1.494 msec, Performace = 2201.03 GFlop/s
//LB = 4: Size = 1.125, Time = 1.216 msec, Performace = 1986.78 GFlop/s
//LB = 3: Size = 1.125, Time = 1.586 msec, Performace = 1523.28 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_4_2R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
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

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index;
	int tj0 = j0 + (tx & 1);
	Ims_n_ih_iw(tj0, tn0, tohs0, tows0);
	tohs0 = (tohs0 + ph) / sh - oph, tows0 = (tows0 + pw) / sw - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;

	//load 2 elem from W
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);

	//load 1 elem from deltaY
	int Y_k = tx >> 1;
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = (ly0 ? deltaY[Y0 + yoffset] : 0);
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

		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);

		//load 1 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		Ys[buf][Ys_x][Ys_y] = (ly0 ? deltaY[Y0 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 y = *(float2*)(&Ys[buf][ik][ty << 1]);
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);
		simdMM4(v0, y.x, w);
		simdMM4(v1, y.y, w);
	}

	deltaX += ic0 + (ihs*sw*IW_slice + iws)*IC; IC *= sw;
	int alpha = sh * IC, X0 = alpha * j0;
	int X1 = X0 + alpha;
	int beta = IC - alpha;//(1 - sh)*IC = IC - sh*IC = IC - alpha
	X0 += beta * ((j0) % IW_slice);
	X1 += beta * ((j0 + 1) % IW_slice);

	*(float4*)(deltaX + X0) = v0;
	*(float4*)(deltaX + X1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*4), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_2_4R
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_2_4R

//LB = 4: Size = 1.53125, Time = 1.51 msec, Performace = 2177.7 GFlop/s
//LB = 4: Size = 1.125, Time = 1.284 msec, Performace = 1881.56 GFlop/s
//LB = 3: Size = 1.125, Time = 1.774 msec, Performace = 1361.85 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_2_4R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
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

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	Ims_n_ih_iw(tj0, tn0, tohs0, tows0);
	Ims_n_ih_iw(tj1, tn1, tohs1, tows1);
	tohs0 = (tohs0 + ph) / sh - oph, tows0 = (tows0 + pw) / sw - opw;
	tohs1 = (tohs1 + ph) / sh - oph, tows1 = (tows1 + pw) / sw - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((tn1*OH + tohs1)*OW + tows1) * OC;

	//load 1 elem from W
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = CW[W_k * IC];

	//load 2 elem from deltaY
	int Y_k = tx >> 1;
	int OW_OC = OW * OC;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = Ims_loadYs2(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
		Y0, tohs0, tows0,
		Y1, tohs1, tows1);
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

		//load 1 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = CW[W_k * IC];

		//load 2 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][Ys_x][Ys_y] = Ims_loadYs2(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
			Y0, tohs0, tows0,
			Y1, tohs1, tows1);
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

	deltaX += ic0 + (ihs*sw*IW_slice + iws)*IC; IC *= sw;
	int alpha = sh * IC, X0 = alpha * j0;
	int X1 = X0 + alpha, X2 = X1 + alpha, X3 = X2 + alpha;
	int beta = IC - alpha;//(1 - sh)*IC = IC - sh*IC = IC - alpha
	X0 += beta * ((j0) % IW_slice);
	X1 += beta * ((j0 + 1) % IW_slice);
	X2 += beta * ((j0 + 2) % IW_slice);
	X3 += beta * ((j0 + 3) % IW_slice);

	*(float2*)(deltaX + X0) = v0;
	*(float2*)(deltaX + X1) = v1;
	*(float2*)(deltaX + X2) = v2;
	*(float2*)(deltaX + X3) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*2), OC >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_2_2R
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_2_2R

//LB = 4: Size = 1.53125, Time = 1.83 msec, Performace = 1796.9 GFlop/s
//LB = 4: Size = 1.08984, Time = 2.664 msec, Performace = 878.537 GFlop/s
//LB = 3: Size = 1.08984, Time = 2.42 msec, Performace = 967.116 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_2_2R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
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
	CW += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index, j1 = j0 + 1;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	IH_IW_slice *= sh * sw; //IH_IW_slice -> IH * IW
	IW_slice *= sw;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih0 * IW_slice + iw0)*IC + ic0;
	int X1 = ((n1*IH_IW_slice) + ih1 * IW_slice + iw1)*IC + ic0;

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

	*(float2*)(deltaX + X0) = v0;
	*(float2*)(deltaX + X1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*1), OC >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_2_1R
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_2_1R

//LB = 4: Size = 1.53125, Time = 2.848 msec, Performace = 1154.61 GFlop/s
//LB = 3: Size = 1.08984, Time = 3.78  msec, Performace =  619.159 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_2_1R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
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
	CW += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) + j_index;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	IH_IW_slice *= sh * sw; //IH_IW_slice -> IH * IW
	IW_slice *= sw;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih0 * IW_slice + iw0)*IC + ic0;

	int ohs0 = (ih0 + ph) / sh - oph, ows0 = (iw0 + pw) / sw - opw;
	int Y0 = ((n0*OH + ohs0)*OW + ows0) * OC;

	//load 2 elem from W
	int W_k = ty;
	Ws[buf][ty][tx] = *(float2*)(CW + W_k * IC);

	//load 1 elem from deltaY
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

	*(float2*)(deltaX + X0) = v0;
}

#endif


//(Y: BLOCK_SIZE*1, X:BLOCK_SIZE*2), OC >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_1_2R
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_1_2R

//LB = 4: Size = 1.53125, Time = 2.662 msec, Performace = 1235.29  GFlop/s
//LB = 3: Size = 1.08984, Time = 4.386 msec, Performace =  533.612 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_1_2R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
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
	CW += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index, j1 = j0 + 1;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	IH_IW_slice *= sh * sw; //IH_IW_slice -> IH * IW
	IW_slice *= sw;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih0 * IW_slice + iw0)*IC + ic0;
	int X1 = ((n1*IH_IW_slice) + ih1 * IW_slice + iw1)*IC + ic0;

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

	deltaX[X0] = v0.x;
	deltaX[X1] = v0.y;
}

#endif


//(Y: BLOCK_SIZE*1, X:BLOCK_SIZE*1), OC >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_1_1R
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_1_1R

//LB = 4: Size = 1.53125, Time = 4.528 msec, Performace = 726.222 GFlop/s
//LB = 3: Size = 1.08984, Time = 7.3   msec, Performace = 320.606 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_1_1R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
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
	CW += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) + j_index;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	IH_IW_slice *= sh * sw; //IH_IW_slice -> IH * IW
	IW_slice *= sw;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih0 * IW_slice + iw0)*IC + ic0;

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

	deltaX[X0] = v;
}

#endif

#endif
