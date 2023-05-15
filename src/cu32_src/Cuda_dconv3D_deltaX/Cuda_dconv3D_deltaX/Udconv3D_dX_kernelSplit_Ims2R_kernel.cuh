#pragma once

//IMS: INPUT_MOD_STEP
//R: remode: W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]
//S2: sh = sw = 2
//(IH, IW) % (sh, sw) == 0
//CWstride = CFH * CFW * OC * IC
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2R_KERNEL_H
#define DCONV3D_DX_KERNEL_SPLIT_IMS2R_KERNEL_H


#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2R_CALL
#define DCONV3D_DX_KERNEL_SPLIT_IMS2R_CALL

#define UksIms2_88R8(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_8R8<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_88R4(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_8R4<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_88R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_8R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, ph, pw, ic_index, j_index)

//=============================================================================================
#define UksIms2_84R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_4R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>2), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_48R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_4_8R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_44R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_4_4R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>2), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_42R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_4_2R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>1), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_24R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_2_4R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>1), (GM>>LB>>2), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, ph, pw, ic_index, j_index)

#endif


//(IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R8
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R8

//LB = 4: Size = 1.53125, Time = 0.7 msec, Performace = 4697.62 GFlop/s
//LB = 4: Size = 1.125, Time = 0.62 msec, Performace = 3896.64 GFlop/s
//LB = 3: Size = 1.125, Time = 0.676 msec, Performace = 3573.84 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_8R8(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz >> 1, x = bz & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = ((n0*(IH_IW_slice << 2)) + ih0 * (IW_slice << 1) + iw0)*IC + ic0;
	int tohs0 = ((ih0 + ph) >> 1) - oph;
	int tows0 = ((iw0 + ((tx >= STEP) << 3) + pw) >> 1) - opw;
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

	IC <<= 1;
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
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R4
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R4

//LB = 4: Size = 1.53125, Time = 0.7 msec, Performace = 4697.62 GFlop/s
//LB = 4: Size = 1.125, Time = 0.622 msec, Performace = 3884.11 GFlop/s
//LB = 3: Size = 1.125, Time = 0.68 msec, Performace = 3552.82 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_8R4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz >> 1, x = bz & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index, j4 = j0 + 4;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j4, n4, ih4, iw4);
	IH_IW_slice <<= 2;//IH_IW_slice -> IH * IW
	IW_slice <<= 1;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih0 * IW_slice + iw0)*IC + ic0;
	int X4 = ((n4*IH_IW_slice) + ih4 * IW_slice + iw4)*IC + ic0;

	bool flagX = (tx >= STEP);
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
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

	IC <<= 1;
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
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R

//LB = 4: Size = 1.53125, Time = 0.716 msec, Performace = 4592.65 GFlop/s
//LB = 4: Size = 1.125, Time = 0.652 msec, Performace = 3705.4  GFlop/s
//LB = 3: Size = 1.125, Time = 0.702 msec, Performace = 3441.48 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_8R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz >> 1, x = bz & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	Ims2_n_ih_iw(tj0, tn0, tohs0, tows0);
	Ims2_n_ih_iw(tj1, tn1, tohs1, tows1);
	Ims2_n_ih_iw(tj2, tn2, tohs2, tows2);
	Ims2_n_ih_iw(tj3, tn3, tohs3, tows3);
	tohs0 = ((tohs0 + ph) >> 1) - oph, tows0 = ((tows0 + pw) >> 1) - opw;
	tohs1 = ((tohs1 + ph) >> 1) - oph, tows1 = ((tows1 + pw) >> 1) - opw;
	tohs2 = ((tohs2 + ph) >> 1) - oph, tows2 = ((tows2 + pw) >> 1) - opw;
	tohs3 = ((tohs3 + ph) >> 1) - oph, tows3 = ((tows3 + pw) >> 1) - opw;
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

	deltaX += ic0 + (ihs*(IW_slice << 1) + iws)*IC;
	int alpha = IC << 2, beta = -IC << 1, X0 = alpha * j0;
	int X1 = X0 + alpha, X2 = X1 + alpha, X3 = X2 + alpha;
	int X4 = X3 + alpha, X5 = X4 + alpha, X6 = X5 + alpha, X7 = X6 + alpha;
	X0 += beta * ((j0) % IW_slice);
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
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_4R
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_4R

//LB = 4: Size = 1.53125, Time = 0.88 msec, Performace = 3736.74 GFlop/s
//LB = 4: Size = 1.125, Time = 0.72 msec, Performace = 3355.44 GFlop/s
//LB = 3: Size = 1.125, Time = 0.766 msec, Performace = 3153.94 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_4R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz >> 1, x = bz & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	CW += ic0 + ((ty >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 2;
	int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	Ims2_n_ih_iw(tj0, tn0, tohs0, tows0);
	Ims2_n_ih_iw(tj1, tn1, tohs1, tows1);
	tohs0 = ((tohs0 + ph) >> 1) - oph, tows0 = ((tows0 + pw) >> 1) - opw;
	tohs1 = ((tohs1 + ph) >> 1) - oph, tows1 = ((tows1 + pw) >> 1) - opw;
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
		float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
	}

	deltaX += ic0 + (ihs*(IW_slice << 1) + iws)*IC;
	int alpha = IC << 2, beta = -IC << 1, X0 = alpha * j0;
	int X1 = X0 + alpha, X2 = X1 + alpha, X3 = X2 + alpha;
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
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_8R
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_8R

//LB = 4: Size = 1.53125, Time = 0.83 msec, Performace = 3961.85 GFlop/s
//LB = 4: Size = 1.125, Time = 0.686 msec, Performace = 3521.75 GFlop/s
//LB = 3: Size = 1.125, Time = 0.856 msec, Performace = 2822.34 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_4_8R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//followed k8

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz >> 1, x = bz & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	Ims2_n_ih_iw(tj0, tn0, tohs0, tows0);
	Ims2_n_ih_iw(tj1, tn1, tohs1, tows1);
	Ims2_n_ih_iw(tj2, tn2, tohs2, tows2);
	Ims2_n_ih_iw(tj3, tn3, tohs3, tows3);
	tohs0 = ((tohs0 + ph) >> 1) - oph, tows0 = ((tows0 + pw) >> 1) - opw;
	tohs1 = ((tohs1 + ph) >> 1) - oph, tows1 = ((tows1 + pw) >> 1) - opw;
	tohs2 = ((tohs2 + ph) >> 1) - oph, tows2 = ((tows2 + pw) >> 1) - opw;
	tohs3 = ((tohs3 + ph) >> 1) - oph, tows3 = ((tows3 + pw) >> 1) - opw;
	int Y0 = ((tn0*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((tn1*OH + tohs1)*OW + tows1) * OC;
	int Y2 = ((tn2*OH + tohs2)*OW + tows2) * OC;
	int Y3 = ((tn3*OH + tohs3)*OW + tows3) * OC;

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int OW_OC = OW * OC;
	Ys[buf][tx][ty] = Ims_loadYs4(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
		Y0, tohs0, tows0,
		Y1, tohs1, tows1,
		Y2, tohs2, tows2,
		Y3, tohs3, tows3);

	//load 2 elem from W
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
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

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = Ims_loadYs4(deltaY, Y_k, OH, OW, OC, CFW_OC, OW_OC,
			Y0, tohs0, tows0,
			Y1, tohs1, tows1,
			Y2, tohs2, tows2,
			Y3, tohs3, tows3);

		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
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

	deltaX += ic0 + (ihs*(IW_slice << 1) + iws)*IC;
	int alpha = IC << 2, beta = -IC << 1, X0 = alpha * j0;
	int X1 = X0 + alpha, X2 = X1 + alpha, X3 = X2 + alpha;
	int X4 = X3 + alpha, X5 = X4 + alpha, X6 = X5 + alpha, X7 = X6 + alpha;
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
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_4R
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_4R

//LB = 4:Size = 1.53125, Time = 1.042 msec, Performace = 3155.79 GFlop/s
//LB = 4: Size = 1.125, Time = 0.848 msec, Performace = 2848.96 GFlop/s
//LB = 3: Size = 1.125, Time = 1.07 msec, Performace = 2257.87 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_4_4R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz >> 1, x = bz & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	Ims2_n_ih_iw(tj0, tn0, tohs0, tows0);
	Ims2_n_ih_iw(tj1, tn1, tohs1, tows1);
	tohs0 = ((tohs0 + ph) >> 1) - oph, tows0 = ((tows0 + pw) >> 1) - opw;
	tohs1 = ((tohs1 + ph) >> 1) - oph, tows1 = ((tows1 + pw) >> 1) - opw;
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

	deltaX += ic0 + (ihs*(IW_slice << 1) + iws)*IC;
	int alpha = IC << 2, beta = -IC << 1, X0 = alpha * j0;
	int X1 = X0 + alpha, X2 = X1 + alpha, X3 = X2 + alpha;
	X0 += beta * ((j0) % IW_slice);
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
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_2R
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_2R

//LB = 4: Size = 1.53125, Time = 1.458 msec, Performace = 2255.37 GFlop/s
//LB = 4: Size = 1.125, Time = 1.158 msec, Performace = 2086.29 GFlop/s
//LB = 3: Size = 1.125, Time = 1.51  msec, Performace = 1599.95 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_4_2R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz >> 1, x = bz & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index;
	int tj0 = j0 + (tx & 1);
	Ims2_n_ih_iw(tj0, tn0, tohs0, tows0);
	tohs0 = ((tohs0 + ph) >> 1) - oph, tows0 = ((tows0 + pw) >> 1) - opw;
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

	deltaX += ic0 + (ihs*(IW_slice << 1) + iws)*IC;
	int alpha = IC << 2, beta = -IC << 1;
	int X0 = alpha * j0, X1 = X0 + alpha;
	X0 += beta * ((j0) % IW_slice);
	X1 += beta * ((j0 + 1) % IW_slice);

	*(float4*)(deltaX + X0) = v0;
	*(float4*)(deltaX + X1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*4), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_2_4R
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_2_4R

//LB = 4:Size = 1.53125, Time = 1.472 msec, Performace = 2233.92 GFlop/s
//LB = 4: Size = 1.125, Time = 1.198 msec, Performace = 2016.63 GFlop/s
//LB = 3: Size = 1.125, Time = 1.668 msec, Performace = 1448.39 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_2_4R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz >> 1, x = bz & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	CW += ic0 + (ty & 1);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	Ims2_n_ih_iw(tj0, tn0, tohs0, tows0);
	Ims2_n_ih_iw(tj1, tn1, tohs1, tows1);
	tohs0 = ((tohs0 + ph) >> 1) - oph, tows0 = ((tows0 + pw) >> 1) - opw;
	tohs1 = ((tohs1 + ph) >> 1) - oph, tows1 = ((tows1 + pw) >> 1) - opw;
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

	deltaX += ic0 + (ihs*(IW_slice << 1) + iws)*IC;
	int alpha = IC << 2, beta = -IC << 1, X0 = alpha * j0;
	int X1 = X0 + alpha, X2 = X1 + alpha, X3 = X2 + alpha;
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

#endif