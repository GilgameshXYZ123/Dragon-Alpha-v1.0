#pragma once

#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_UERNEL_H
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_UERNEL_H

//IMS: INPUT_MOD_STEP
//R: remode: W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]
//S2: sh = sw = 2
//(IH, IW) % (sh, sw) == 0
//CWstride = CFH * CFW * OC * IC
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_UERNEL_CALL
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_UERNEL_CALL

//======[Common]============================================================
#define ksV2_Ims2_u88R(stream, LB, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	ksV2_Ims2_uernel_8_8R<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, n_index)

//======[FW = 4 or 3 or 2] -> [CFW is power of 2]===========================
#define ksV2_Ims2_u88R_CFW_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, N) \
	ksV2_Ims2_uernel_8_8R_CFW_OC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, LOC, ph, pw,\
			ic_index, n_index)

//======[FW = 5] -> [CFW = 3 or 2], [OH, OW >= 2]===========================
#define ksV2_Ims2_u88R_W5_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, CW, deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, N) \
	ksV2_Ims2_uernel_8_8R_W5_OC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, LOC, ph, pw,\
			ic_index, n_index)

#endif


//======[Common]============================================================
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % BLOCK_SIZE == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R

//(OH, OW) = 2:
//LB = 4: Size = 1.125, Time = 0.362 msec, Performace = 6673.81 GFlop/s
//LB = 3: Size = 1.125, Time = 0.382 msec, Performace = 6324.4 GFlop/s
//[(OH, OW) = 4]:
//LB = 4: Size = 1.125, Time = 0.436 msec, Performace = 5541.8 GFlop/s
//LB = 3: Size = 1.125, Time = 0.438 msec, Performace = 5515.8 GFlop/s
//(OH, OW) = 8:
//LB = 4: Size = 1.125, Time = 0.706 msec, Performace = 3421.98 GFlop/s
//LB = 3: Size = 1.125, Time = 0.682 msec, Performace = 3542.4 GFlop/s
//(OH, OW) = 16:
//LB = 4: Size = 4.5, Time = 2.472 msec, Performace = 3909.25 GFlop/s
//LB = 3: Size = 4.5, Time = 2.488 msec, Performace = 3884.11 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void ksV2_Ims2_uernel_8_8R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GZ((y, x), ih, iw) = sh * sw * IH_slice * IW_slice 
	int bz = blockIdx.z;
	int yx = bz / IH_IW_slice; bz -= yx * IH_IW_slice;//bz %= IH_IW_slice
	int ih = bz / IW_slice, iw = bz - ih * IW_slice;//bz % IW_slice
	CW += yx * CWstride;//CW[y, x]

	int y = yx >> 1, x = yx & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));
	ih = (ih << 1) + ihs;
	iw = (iw << 1) + iws;

	//prepare for GK(bz(y, x)) = tCFW * tCFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	int ohs = ((ih + ph) >> 1) - oph;
	int ows = ((iw + pw) >> 1) - opw;

	int fhs = -IF_int((ohs < 0), ohs, 0);
	int fws = -IF_int((ows < 0), ows, 0);
	int fhe = OH - ohs; fhe = IF_int((fhe > CFH), CFH, fhe);
	int fwe = OW - ows; fwe = IF_int((fwe > CFW), CFW, fwe);
	const int tCFH = fhe - fhs;
	const int tCFW = fwe - fws;
	const int tCFW_OC = tCFW * OC, GK = tCFH * tCFW_OC;
	CW += (fhs*CFW + fws)*OC*IC;//CW[y, x, fhs, fws, 0, 0]
	deltaY += (fhs*OW + fws)*OC;//deltaY[0, fhs, fws, 0]

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, fhs, fws, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	IH_IW_slice <<= 2;//IH_IW_slice * sh * sw -> IH * IW
	IW_slice <<= 1;//IW_slice * sw -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + ohs)*OW + ows) * OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elem from deltaY
	int Y_k = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - tCFW)*OC;
	int fhr = Y_k / tCFW_OC;
	int yoffset = fhr * SY + Y_k;
	float2 dy0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 dy1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 dy2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 dy3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

	//load 4 elem from W
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	const int SW = (CFW - tCFW)*OC;
	int woffset = (fhr * SW + W_k)*IC;
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

		//load 4 elem from Y
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int fhr = Y_k / tCFW_OC;
		int yoffset = fhr * SY + Y_k;
		float2 dy0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 dy1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 dy2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 dy3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

		//load 4 elem from W
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = (fhr * SW + W_k)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
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

	int Xstride = IH_IW_slice * IC;//IH * IW * IC
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


//======[FW = 4 or 3 or 2] -> [CFW is power of 2]===========================
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, OC is power of 2
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R_CFW_OC2POW
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R_CFW_OC2POW

//(OH, OW) = 2:
//LB = 4: Size = 1.125, Time = 0.354 msec, Performace = 6824.63 GFlop/s
//LB = 3: Size = 1.125, Time = 0.368 msec, Performace = 6565    GFlop/s
//(OH, OW) = 4:
//LB = 4: Size = 1.125, Time = 0.426 msec, Performace = 5671.17 GFlop/s
//LB = 3: Size = 1.125, Time = 0.42  msec, Performace = 5752.19 GFlop/s
//(OH, OW) = 8:
//LB = 4: Size = 1.125, Time = 0.684 msec, Performace = 3532.05 GFlop/s
//LB = 3: Size = 1.125, Time = 0.67  msec, Performace = 3605.85 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void ksV2_Ims2_uernel_8_8R_CFW_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int LOC,
	int ph, int pw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GZ((y, x), ih, iw) = sh * sw * IH_slice * IW_slice 
	int bz = blockIdx.z;
	int yx = bz / IH_IW_slice; bz -= yx * IH_IW_slice;//bz %= IH_IW_slice
	int ih = bz / IW_slice, iw = bz - ih * IW_slice;//bz % IW_slice
	CW += yx * CWstride;//CW[y, x]

	int y = yx >> 1, x = yx & 1;//x = yx % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));
	ih = (ih << 1) + ihs;
	iw = (iw << 1) + iws;

	//prepare for GK(bz) = tCFW * tCFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	int ohs = ((ih + ph) >> 1) - oph;
	int ows = ((iw + pw) >> 1) - opw;

	int fhs = -IF_int((ohs < 0), ohs, 0);
	int fws = -IF_int((ows < 0), ows, 0);
	int fhe = OH - ohs; fhe = IF_int((fhe > CFH), CFH, fhe);
	int fwe = OW - ows; fwe = IF_int((fwe > CFW), CFW, fwe);
	const int tCFH = fhe - fhs;
	const int tCFW = fwe - fws;

	//tCFW = 2 or 1, so (tCFW >> 1) == log2(tCFW)
	const int LtCFW_OC = (tCFW >> 1) + LOC, GK = tCFH << LtCFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += (((fhs*CFW + fws)*IC) << LOC) + ic0 + ((ty >= STEP) << 2);//CW[y, x, fhs, fws, 0, tic0]
	
	//prepare_for_GM = N*IH_slice*IW_slice
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	IH_IW_slice <<= 2;//IH_IW_slice -> IH * IW
	IW_slice <<= 1;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + ohs + fhs)*OW + ows + fws) << LOC;//Y[tn0, fhs + ohs, fws + owss, 0]
	int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//load 4 elem from deltaY
	int Y_k = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - tCFW) << LOC;
	int fhr = Y_k >> LtCFW_OC;
	int yoffset = fhr * SY + Y_k;
	float2 dy0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 dy1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 dy2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 dy3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

	//load 4 elem from W
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	const int SW = (CFW - tCFW) << LOC;
	int woffset = (fhr * SW + W_k)*IC;
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

		//load 4 elem from Y
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int fhr = Y_k >> LtCFW_OC;
		int yoffset = fhr * SY + Y_k;
		float2 dy0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 dy1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 dy2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 dy3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

		//load 4 elem from W
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = (fhr * SW + W_k)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

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

	int Xstride = IH_IW_slice * IC;//IH * IW * IC
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


//======[FW = 5] -> [CFW = 3 or 2], [OH, OW >= 2]=============================
//LB = 4: OC % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R_W5_OC2POW
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R_W5_OC2POW

//(OH, OW) = 2:
//LB = 4: Size = 1.5625, Time = 0.378 msec, Performace = 8876.83 GFlop/s
//LB = 3: Size = 1.5625, Time = 0.374 msec, Performace = 8971.77 GFlop/s
//(OH, OW) = 4:
//LB = 4: Size = 1.5625, Time = 0.482 msec, Performace = 6961.5  GFlop/s
//LB = 3: Size = 1.5625, Time = 0.492 msec, Performace = 6820.01 GFlop/s
//(OH, OW) = 8:
//LB = 4: Size = 3.125, Time = 1.136 msec, Performace = 5907.47 GFlop/s
//LB = 3: Size = 1.5625, Time = 0.746 msec, Performace = 4497.91 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void ksV2_Ims2_uernel_8_8R_W5_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int CWstride,//FH = FW = 5
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int LOC,
	int ph, int pw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//prepare for GZ((y, x), ih, iw) = sh * sw * IH_slice * IW_slice 
	int bz = blockIdx.z;
	int yx = bz / IH_IW_slice; bz -= yx * IH_IW_slice;//bz %= IH_IW_slice
	int ih = bz / IW_slice, iw = bz - ih * IW_slice;//bz % IW_slice
	CW += yx * CWstride;//CW[y, x]

	int y = yx >> 1, x = yx & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));
	ih = (ih << 1) + ihs;
	iw = (iw << 1) + iws;

	//prepare for GK(bz(y, x)) = tCFW * tCFH * OC
	int CFH = (6 - y) >> 1, oph = CFH - 1;
	int CFW = (6 - x) >> 1, opw = CFW - 1;
	int ohs = ((ih + ph) >> 1) - oph;
	int ows = ((iw + pw) >> 1) - opw;

	int fhs = -IF_int((ohs < 0), ohs, 0);
	int fws = -IF_int((ows < 0), ows, 0);
	int fhe = OH - ohs; fhe = IF_int((fhe > CFH), CFH, fhe);
	int fwe = OW - ows; fwe = IF_int((fwe > CFW), CFW, fwe);
	const int tCFH = fhe - fhs;
	const int tCFW = fwe - fws;
	const int tCFW_OC = tCFW << LOC, GK = tCFH * tCFW_OC;
	CW += ((fhs*CFW + fws)*IC) << LOC;//CW[y, x, fhs, fws, 0, 0]
	deltaY += (fhs*OW + fws) << LOC;//deltaY[0, fhs, fws, 0]

	int fh_idx = (tCFH == 2) + ((tCFH == 3) << 1);
	int fw_idx = (tCFW == 2) + ((tCFW == 3) << 1);
	int fhw_offset = (fh_idx * 3 + fw_idx) * 9;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, fhs, fws, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	IH_IW_slice <<= 2;//IH_IW_slice * sh * sw -> IH * IW
	IW_slice <<= 1;//IW_slice * sw -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + ohs)*OW + ows) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//load 4 elem from deltaY
	int Y_k = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int SY = (OW - tCFW) << LOC;
	int Idx = Y_k >> LOC; char fhr = YIDX_V2_W3P2[fhw_offset + Idx] >> 2;
	int yoffset = fhr * SY + Y_k;
	float2 dy0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 dy1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 dy2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 dy3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

	//load 4 elem from W
	int W_k = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	const int SW = (CFW - tCFW) << LOC;
	int woffset = (fhr * SW + W_k)*IC;
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

		//load 4 elem from Y
		int Y_k = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		int Idx = Y_k >> LOC; char fhr = YIDX_V2_W3P2[fhw_offset + Idx] >> 2;
		int yoffset = fhr * SY + Y_k;
		float2 dy0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 dy1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 dy2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 dy3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

		//load 4 elem from W
		int W_k = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset = (fhr * SW + W_k)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
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

	const int Xstride = IH_IW_slice * IC;//IH * IW * IC
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