#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_S1_EX2_H
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_S1_EX2_H

//Unsparse Matrix Method
//We have:
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_S1_EX2_CALL
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_S1_EX2_CALL

//LB = log2(BLOCK_SIZE)

//=====[ph = pw = 1] , [FH = FW = 3], [OH, OW > 1] -> [oph = opw = 1]===========
#define kV2_88s1W3P1(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N) \
	zeroPaddingV2_kernel_8_8_s1_W3P1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,OC,\
			 ic_index,n_index)

#define kV2_88s1W3P1_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	zeroPaddingV2_kernel_8_8_s1_W3P1_OC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,LOC,\
			 ic_index,n_index)

//=====[ph = pw = 2] , [FH = FW = 5], [OH, OW > 2] -> [oph = opw = 2]===========
#define kV2_88s1W5P2(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N) \
	zeroPaddingV2_kernel_8_8_s1_W5P2<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,OC,\
			 ic_index,n_index)

#define kV2_88s1W5P2_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	zeroPaddingV2_kernel_8_8_s1_W5P2_OC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,LOC,\
			 ic_index,n_index)

#endif


//=====[ph = pw = 1], [FH = FW = 3], [OH, OW > 1] -> [oph = opw = 1]============
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4, OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_8_S1_W3_P1
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_8_S1_W3_P1

//(IH, IW) = 2:
//LB = 4: Size = 1.125, Time = 0.858 msec, Performace = 2815.76 GFlop/s
//LB = 3: Size = 1.125, Time = 1.01  msec, Performace = 2392    GFlop/s
//(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.142 msec, Performace = 2115.52 GFlop/s
//LB = 3: Size = 1.125, Time = 1.43  msec, Performace = 1689.45 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.125, Time = 1.398 msec, Performace = 1728.13 GFlop/s
//LB = 3: Size = 1.125, Time = 1.61  msec, Performace = 1500.57 GFlop/s
template<int LB, int STEP>
__global__ void zeroPaddingV2_kernel_8_8_s1_W3P1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__  W,//tFH = tFW = 2 or 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,//oph = opw = 1
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

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
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;//GK[fh, fw, oc]
	deltaY += (fhs*OW + fws)*OC;//deltaY[0, fhs, fws, 0]
	W += ((2 - fhs) * 3 + (2 - fws)) * IC;//W[0, -fhs, -fws, 0]

	int fh_idx = (tFH == 3), fw_idx = (tFW == 3);
	int fhw_offset = ((fh_idx << 1) + fw_idx) * 9;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	int FW = 3 * IC;//FW -> FW * IC = 3*IC
	int FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = (9 * OC + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(FW) = FH * FW * IC * FW * OC = 27*IC*OC
	FW = tFW_OC * FH + FW;//It's hard to out of bound of memory 

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = OC * (OW - tFW);
	int Idx = Y_k / OC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
	int fh = fhw >> 2, fw = fhw & 3;
	float4 x; int yoffset = fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int woffset = W_k * FH - fh * FW - fw * IC;
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
		int Idx = Y_k / OC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
		int fh = fhw >> 2, fw = fhw & 3;
		float4 x; int yoffset = fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = W_k * FH - fh * FW - fw * IC;
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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), N % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4, OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_8_S1_W3_P1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_8_S1_W3_P1_OC2POW

//(IH, IW) = 2:
//LB = 4: Size = 1.125, Time = 0.792 msec, Performace = 3050.4  GFlop/s
//LB = 3: Size = 1.125, Time = 0.894 msec, Performace = 2702.37 GFlop/s
//(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.198 msec, Performace = 2016.63 GFlop/s
//LB = 3: Size = 1.125, Time = 1.398 msec, Performace = 1728.13 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.125, Time = 1.318 msec, Performace = 1833.02 GFlop/s
//LB = 3: Size = 1.125, Time = 1.532 msec, Performace = 1576.97 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.125, Time = 1.452 msec, Performace = 1663.86 GFlop/s
//LB = 3: Size = 1.125, Time = 1.584 msec, Performace = 1525.2  GFlop/s
template<int LB, int STEP>
__global__ void zeroPaddingV2_kernel_8_8_s1_W3P1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__  W,//tFH = tFW = 2 or 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,//oph = opw = 1
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

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
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;//GK[fh, fw, oc]
	deltaY += (fhs*OW + fws) << LOC;//deltaY[0, fhs, fws, 0]
	W += ((2 - fhs) * 3 + (2 - fws)) * IC;//W[0, -fhs, -fws, 0]

	int fh_idx = (tFH == 3), fw_idx = (tFW == 3);
	int fhw_offset = ((fh_idx << 1) + fw_idx) * 9;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	int FW = 3 * IC;//FW -> FW * IC = 3*IC
	int FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = ((9 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 81*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 3640, It's hard to out of bound of memory 

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = (OW - tFW) << LOC;
	int Idx = Y_k >> LOC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
	int fh = fhw >> 2, fw = fhw & 3;
	float4 x; int yoffset = fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int woffset = W_k * FH - fh * FW - fw * IC;
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
		int Idx = Y_k >> LOC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
		int fh = fhw >> 2, fw = fhw & 3;
		float4 x; int yoffset = fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = W_k * FH - fh * FW - fw * IC;
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


//=====[ph = pw = 2], [FH = FW = 5], [OH, OW > 2] -> [oph = opw = 2]============
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4, OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_8_S1_W5_P2
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_8_S1_W5_P2

//(IH, IW) = 4:
//LB = 4: Size = 1.5625, Time = 1.22  msec, Performace = 2750.36 GFlop/s
//LB = 3: Size = 1.5625, Time = 1.414 msec, Performace = 2373.01 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.5625, Time = 1.85 msec, Performace = 1813.75 GFlop/s
//LB = 3: Size = 1.5625, Time = 1.87 msec, Performace = 1794.35 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.5625, Time = 1.852 msec, Performace = 1811.79 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.088 msec, Performace = 1607.01 GFlop/s
template<int LB, int STEP>
__global__ void zeroPaddingV2_kernel_8_8_s1_W5P2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__  W,//tFH = tFW = 5, 4, 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,//oph = opw = 2
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - 2, tiw = iw - 2;//tih = ih - oph

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 5), 5, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 5), 5, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;

	int fh_idx = (tFH == 4) + ((tFH == 5) << 1);
	int fw_idx = (tFW == 4) + ((tFW == 5) << 1);
	int fhw_offset = ((fh_idx * 3) + fw_idx) * 25;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((4 - fhs) * 5 + (4 - fws))*IC + ic0 + ((ty >= STEP) << 2);//W[0, -fhs, -fws, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih + fhs)*OW + tiw + fws)*OC;//deltaY[0, fhs, fws, 0]
	int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	int FW =  5 * IC;//FW -> FW * IC = 5*IC
	int FH = 25 * IC;//FH -> FH * FW * IC = 25*IC
	IC = (25 * OC + 1)*IC;//IC -> OC * FH * FW * IC + IC = (25*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) =FH * FH * FW * IC * FW * OC = 625*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 1310, It's hard to out of bound of memory 

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = OC * (OW - tFW);
	int Idx = Y_k / OC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
	int fh = fhw >> 3, fw = fhw & 7;
	float4 x; int yoffset = fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//Ws: with the same ty
	int woffset = W_k * FH - fh * FW - fw * IC;
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
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = Y_k / OC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
		int fh = fhw >> 3, fw = fhw & 7;
		float4 x; int yoffset = fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4, OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_8_S1_W5_P2_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_KERNEL_8_8_S1_W5_P2_OC2POW

//(IH, IW) = 4:
//LB = 4: Size = 1.5625, Time = 1.158 msec, Performace = 2897.62 GFlop/s
//LB = 3: Size = 1.5625, Time = 1.314 msec, Performace = 2553.61 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.5625, Time = 1.516 msec, Performace = 2213.35 GFlop/s
//LB = 3: Size = 1.5625, Time = 1.754 msec, Performace = 1913.02 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.5625, Time = 1.802 msec, Performace = 1862.07 GFlop/s
//LB = 3: Size = 1.5625, Time = 1.978 msec, Performace = 1696.38 GFlop/s
template<int LB, int STEP>
__global__ void zeroPaddingV2_kernel_8_8_s1_W5P2_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__  W,//tFH = tFW = 5, 4, 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,//oph = opw = 2
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

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

	int fh_idx = (tFH == 4) + ((tFH == 5) << 1);
	int fw_idx = (tFW == 4) + ((tFW == 5) << 1);
	int fhw_offset = ((fh_idx * 3) + fw_idx) * 25;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((4 - fhs) * 5 + (4 - fws))*IC + ic0 + ((ty >= STEP) << 2);//W[0, -fhs, -fws, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + fhs + tih)*OW + fws + tiw) << LOC;//deltaY[0, fhs, fws, 0]
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	int FW = 5 * IC;//FW -> FW * IC = 5*IC
	int FH = 25 * IC;//FH -> FH * FW * IC = 25*IC
	IC = ((25 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (25*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 625*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 1310, It's hard to out of bound of memory 

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = (OW - tFW) << LOC;
	int Idx = Y_k >> LOC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
	int fh = fhw >> 3, fw = fhw & 7;
	float4 x; int yoffset = fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int woffset = W_k * FH - fh * FW - fw * IC;
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

		//load 4 elem from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = Y_k >> LOC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
		int fh = fhw >> 3, fw = fhw & 7;
		float4 x; int yoffset = fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = W_k * FH - fh * FW - fw * IC;
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
