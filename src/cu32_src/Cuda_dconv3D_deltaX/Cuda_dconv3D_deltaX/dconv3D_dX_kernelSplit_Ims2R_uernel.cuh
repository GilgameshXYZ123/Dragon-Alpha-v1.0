#pragma once

#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2R_UERNEL_EX_H
#define DCONV3D_DX_KERNEL_SPLIT_IMS2R_UERNEL_EX_H

//IMS: INPUT_MOD_STEP
//R: remode: W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]
//S2: sh = sw = 2
//(IH, IW) % (sh, sw) == 0
//CWstride = CFH * CFW * OC * IC
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2R_EX_UERNEL_CALL
#define DCONV3D_DX_KERNEL_SPLIT_IMS2R_EX_UERNEL_CALL

//LB = log2(BLOCK_SIZE)

//======[OC is power of 2, FW = (2||3||4) -> CFW = (2||1)]===================
#define ksIms2_u88R8_oc_CFW2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM) \
	ksIms2_uernel_8_8R8_OC_CFW2pow<LB, (1<<LB>>1), (1 << LB)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, LOC, ph, pw,\
			ic_index, j_index)

#define ksIms2_u88R4_oc_CFW2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM) \
	ksIms2_uernel_8_8R4_OC_CFW2pow<LB, (1<<LB>>1), (1 << LB)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, LOC, ph, pw,\
			ic_index, j_index)

#define ksIms2_u88R_oc_CFW2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM) \
	ksIms2_uernel_8_8R_OC_CFW2pow<LB, (1<<LB>>1), (1 << LB)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, LOC, ph, pw,\
			ic_index, j_index)

#endif


//======[OC is power of 2, FW = (2||3||4) -> CFW = (2||1)]===================
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8),  OC % BLOCK_SIZE == 0, (IH_slice, IW_slice) % 8 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_UERNEL_8_8R8_OC_CFW_2POW
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_UERNEL_8_8R8_OC_CFW_2POW

//(IH, IW) = 32, N *= 2:
//LB = 4: Size = 1.125, Time = 0.469 msec, Performace = 5151.21 GFlop/s(1000)
//LB = 3: Size = 1.125, Time = 0.472 msec, Performace = 5118.47 GFlop/s(1000)
//(IH, IW) = 32, OC *= 2:
//LB = 4: Size = 1.125, Time = 0.448 msec, Performace = 5392.68 GFlop/s(1000)
//LB = 3: Size = 1.125, Time = 0.458 msec, Performace = 5274.93 GFlop/s(1000)
template<int LB, int STEP, int STEP2>
__global__ void ksIms2_uernel_8_8R8_OC_CFW2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int LOC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz >> 1, x = bz & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int LCFW_OC = (CFW >> 1) + LOC, CFW_OC_m1 = (1 << LCFW_OC) - 1;
	const int GK = CFH << LCFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = ((n0*(IH_IW_slice << 2)) + ih0 * (IW_slice << 1) + iw0)*IC + ic0;
	int tohs0 = ((ih0 + ph) >> 1) - oph;
	int tows0 = ((iw0 + ((tx >= STEP) << 3) + pw) >> 1) - opw;
	int tows1 = tows0 + 1, tows2 = tows0 + 2, tows3 = tows0 + 3;
	deltaY += ((n0*OH + tohs0)*OW + tows1) << LOC;//deltaY += Y1
	OH -= tohs0;//OH = OH - tohs0 = OH_m_tohs0

	//load 4 elem from deltaY
	int Y_k = tx - (((tx >= STEP) << LB >> 1)) << 1;
	Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	float2 y0 = (ly0 ? *(float2*)(deltaY + yoffset - (1 << LOC)) : F32_2_0);
	float2 y1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);
	float2 y2 = (ly2 ? *(float2*)(deltaY + yoffset + (1 << LOC)) : F32_2_0);
	float2 y3 = (ly3 ? *(float2*)(deltaY + yoffset + (2 << LOC)) : F32_2_0);
	Ys[buf][(tx << 1)][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };

	//load 4 elem from W
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_k * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
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
		buf ^= 1;

		//load 4 elem from Y
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
		bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH);
		bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		int yoffset = (Y_fhr << LOC) * OW + Y_k;
		float2 y0 = (ly0 ? *(float2*)(deltaY + yoffset - (1 << LOC)) : F32_2_0);
		float2 y1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);
		float2 y2 = (ly2 ? *(float2*)(deltaY + yoffset + (1 << LOC)) : F32_2_0);
		float2 y3 = (ly3 ? *(float2*)(deltaY + yoffset + (2 << LOC)) : F32_2_0);
		Ys[buf][(tx << 1)][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };

		//load 4 elem from W
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = W_k * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
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

	IC <<= 1;
	const int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	const int X4 = X3 + IC, X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, (IH_slice, IW_slice) % 4 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_UERNEL_8_8R4_OC_CFW_2POW
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_UERNEL_8_8R4_OC_CFW_2POW

//(IH, IW) = 32, N *= 2:
//LB = 4: Size = 1.125, Time = 0.473 msec, Performace = 5107.65 GFlop/s(1000)
//LB = 3: Size = 1.125, Time = 0.47  msec, Performace = 5140.25 GFlop/s(1000)
//(IH, IW) = 32, OC *= 2:
//LB = 4: Size = 1.125, Time = 0.45  msec, Performace = 5368.71 GFlop/s(1000)
//LB = 3: Size = 1.125, Time = 0.456 msec, Performace = 5298.07 GFlop/s(1000)
template<int LB, int STEP, int STEP2>
__global__ void ksIms2_uernel_8_8R4_OC_CFW2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int LOC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz >> 1, x = bz & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int LCFW_OC = (CFW >> 1) + LOC, CFW_OC_m1 = (1 << LCFW_OC) - 1;
	const int GK = CFH << LCFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, tic0]

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
	deltaY += ((n0*OH + tohs0)*OW + tows1) << LOC;//deltaY += Y1
	OH -= tohs0;//OH = OH - tohs0 = OH_m_tohs0

	//load 4 elem from deltaY
	int Y_k = tx - (((tx >= STEP) << LB >> 1)) << 1;
	Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	float2 y0 = (ly0 ? *(float2*)(deltaY + yoffset - (1 << LOC)) : F32_2_0);
	float2 y1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);
	float2 y2 = (ly2 ? *(float2*)(deltaY + yoffset + (1 << LOC)) : F32_2_0);
	float2 y3 = (ly3 ? *(float2*)(deltaY + yoffset + (2 << LOC)) : F32_2_0);
	Ys[buf][(tx << 1)][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };

	//load 4 elem from W
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_k * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
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
		buf ^= 1;

		//load 4 elem from Y
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
		bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH);
		bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		int yoffset = (Y_fhr << LOC) * OW + Y_k;
		float2 y0 = (ly0 ? *(float2*)(deltaY + yoffset - (1 << LOC)) : F32_2_0);
		float2 y1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);
		float2 y2 = (ly2 ? *(float2*)(deltaY + yoffset + (1 << LOC)) : F32_2_0);
		float2 y3 = (ly3 ? *(float2*)(deltaY + yoffset + (2 << LOC)) : F32_2_0);
		Ys[buf][(tx << 1)][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };

		//load 4 elem from W
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = W_k * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
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

	IC <<= 1;
	const int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	const int X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, 
//LB = 3: OC %  8 == 0 [LB = 3 Only]
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_UERNEL_8_8R_OC_CFW_2POW
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_UERNEL_8_8R_OC_CFW_2POW

//(IH, IW) = 32, N *= 2:
//LB = 3: Size = 1.125, Time = 0.483 msec, Performace = 5001.9 GFlop/s
//(IH, IW) = 32, OC *= 2:
//LB = 3: Size = 1.125, Time = 0.467 msec, Performace = 5173.27 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void ksIms2_uernel_8_8R_OC_CFW2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int LOC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz >> 1, x = bz & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int LCFW_OC = (CFW >> 1) + LOC, CFW_OC_m1 = (1 << LCFW_OC) - 1;
	const int GK = CFH << LCFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, tic0]

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
	const int Y0 = ((tn0*OH + tohs0)*OW + tows0) << LOC;
	const int Y1 = ((tn1*OH + tohs1)*OW + tows1) << LOC;
	const int Y2 = ((tn2*OH + tohs2)*OW + tows2) << LOC;
	const int Y3 = ((tn3*OH + tohs3)*OW + tows3) << LOC;
	deltaX += ic0 + (ihs*(IW_slice << 1) + iws)*IC;

	//load 4 elem from deltaY
	int Y_k = tx - (((tx >= STEP) << LB >> 1)) << 1;
	Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	bool ly0 = LOAD_Y(tohs0, tows0, Y_fhr, Y_fwr);
	bool ly1 = LOAD_Y(tohs1, tows1, Y_fhr, Y_fwr);
	bool ly2 = LOAD_Y(tohs2, tows2, Y_fhr, Y_fwr);
	bool ly3 = LOAD_Y(tohs3, tows3, Y_fhr, Y_fwr);
 	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	float2 y0 = (ly0 ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
	float2 y1 = (ly1 ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
	float2 y2 = (ly2 ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
	float2 y3 = (ly3 ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
	Ys[buf][(tx << 1)][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };

	//load 4 elem from W
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset = W_k * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
		bool ly0 = LOAD_Y(tohs0, tows0, Y_fhr, Y_fwr);
		bool ly1 = LOAD_Y(tohs1, tows1, Y_fhr, Y_fwr);
		bool ly2 = LOAD_Y(tohs2, tows2, Y_fhr, Y_fwr);
		bool ly3 = LOAD_Y(tohs3, tows3, Y_fhr, Y_fwr);
		int yoffset = (Y_fhr << LOC) * OW + Y_k;
		float2 y0 = (ly0 ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 y1 = (ly1 ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 y2 = (ly2 ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 y3 = (ly3 ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };

		//load 4 elem from W
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = W_k * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

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

#endif