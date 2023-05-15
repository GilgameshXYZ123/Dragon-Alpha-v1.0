#pragma once

#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_KERNEL_H
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_KERNEL_H

//IMS: INPUT_MOD_STEP
//R: remode: W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]
//S2: sh = sw = 2
//(IH, IW) % (sh, sw) == 0
//CWstride = CFH * CFW * OC * IC
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_CALL
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]==============================================
#define ksV2_Ims2_88R(stream, LB, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	ksV2_Ims2_kernel_8_8R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, n_index)

#define ksV2_Ims2_84R(stream, LB, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	ksV2_Ims2_kernel_8_4R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>2), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, n_index)

#define ksV2_Ims2_48R(stream, LB, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	ksV2_Ims2_kernel_4_8R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, n_index)

#define ksV2_Ims2_44R(stream, LB, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	ksV2_Ims2_kernel_4_4R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (N>>LB>>2), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, n_index)

#endif


//======[Common]==============================================
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_KERNEL_8_8R
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_KERNEL_8_8R

//Target: Size = 1.125, Time = 0.484 msec, Performace = 4991.57 GFlop/s
//(OH, OW) = 2:
//LB = 4: Size = 1.125, Time = 0.372 msec, Performace = 6494.41 GFlop/s
//LB = 3: Size = 1.125, Time = 0.436 msec, Performace = 5541.1 GFlop/s
//[(OH, OW) = 4]:
//LB = 4: Size = 1.125, Time = 0.446 msec, Performace = 5416.86 GFlop/s
//LB = 3: Size = 1.125, Time = 0.482 msec, Performace = 5012.28 GFlop/s
//(OH, OW) = 8:
//k88R8_oc2pow<4>: Size = 1.125, Time = 0.724 msec, Performace = 3336.9 GFlop/s
//k88R8_oc2pow<3>: Size = 1.125, Time = 0.73  msec, Performace = 3309.48 GFlop/s
//LB = 4: Size = 1.125, Time = 0.706 msec, Performace = 3421.98 GFlop/s
//LB = 3: Size = 1.125, Time = 0.724 msec, Performace = 3336.9  GFlop/s
//(OH, OW) = 16:
//k88R8_oc2pow<4>: Size = 4.5, Time = 3    msec, Performace = 3221.23 GFlop/s
//k88R8_oc2pow<3>: Size = 4.5, Time = 3.14 msec, Performace = 3077.6 GFlop/s
//LB = 4: Size = 4.5, Time = 2.99 msec, Performace = 3232    GFlop/s
//LB = 3: Size = 4.5, Time = 3.07 msec, Performace = 3147.78 GFlop/s
template<int LB, int STEP>
__global__ void ksV2_Ims2_kernel_8_8R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

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
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = (OW - tCFW)*OC;
	int fhr = Y_k / tCFW_OC;
	float4 dy; int yoffset = fhr * SY + Y_k;
	dy.x = deltaY[Y0 + yoffset];
	dy.y = deltaY[Y1 + yoffset];
	dy.z = deltaY[Y2 + yoffset];
	dy.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = dy;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	const int SW = (CFW - tCFW)*OC;
	int woffset = (fhr * SW + W_k)*IC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int fhr = Y_k / tCFW_OC;
		float4 dy; int yoffset = fhr * SY + Y_k;
		dy.x = deltaY[Y0 + yoffset];
		dy.y = deltaY[Y1 + yoffset];
		dy.z = deltaY[Y2 + yoffset];
		dy.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = dy;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = (fhr * SW + W_k)*IC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*4), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_KERNEL_8_4R
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_KERNEL_8_4R

//Target: Size = 1.125, Time = 0.484 msec, Performace = 4991.57 GFlop/s
//(OH, OW) = 2:
//LB = 4: Size = 1.125, Time = 0.454 msec, Performace = 5321.41 GFlop/s
//LB = 3: Size = 1.125, Time = 0.518 msec, Performace = 4663.94 GFlop/s
//(OH, OW) = 4:
//LB = 4: Size = 1.125, Time = 0.548 msec, Performace = 4408.61 GFlop/s
//LB = 3: Size = 1.125, Time = 0.596 msec, Performace = 4053.56 GFlop/s
template<int LB, int STEP>
__global__ void ksV2_Ims2_kernel_8_4R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

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
	int n0 = (((blockIdx.y << LB) + ty) << 2) + n_index;
	IH_IW_slice <<= 2;//IH_IW_slice * sh * sw -> IH * IW
	IW_slice <<= 1;//IW_slice * sw -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;
	int tn0 = n0 + ((tx & 1) << 1);
	int Y0 = ((tn0*OH + ohs)*OW + ows) * OC;
	int Y1 = Y0 + OH * OW * OC;

	//load 2 elem from deltaY
	int Y_k = tx >> 1;
	const int SY = (OW - tCFW)*OC;
	int fhr = Y_k / tCFW_OC;
	float2 dy; int yoffset = fhr * SY + Y_k;
	dy.x = deltaY[Y0 + yoffset];
	dy.y = deltaY[Y1 + yoffset];
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = dy;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	const int SW = (CFW - tCFW)*OC;
	int woffset = (fhr * SW + W_k)*IC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		}
		buf ^= 1;

		//load 2 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		int fhr = Y_k / tCFW_OC;
		float2 dy; int yoffset = fhr * SY + Y_k;
		dy.x = deltaY[Y0 + yoffset];
		dy.y = deltaY[Y1 + yoffset];
		Ys[buf][Ys_x][Ys_y] = dy;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = (fhr * SW + W_k)*IC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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

	int Xstride = IH_IW_slice * IC;//IH * IW * IC
	int X1 = X0 + Xstride;
	int X2 = X1 + Xstride;
	int X3 = X2 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_KERNEL_4_8R
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_KERNEL_4_8R

//Target: Size = 1.125, Time = 0.484 msec, Performace = 4991.57 GFlop/s
//(OH, OW) = 2:
//LB = 4: Size = 1.125, Time = 0.396 msec, Performace = 6100.81 GFlop/s
//LB = 3: Size = 1.125, Time = 0.5   msec, Performace = 4831.84 GFlop/s
//(OH, OW) = 4:
//LB = 4: Size = 1.125, Time = 0.458 msec, Performace = 5274.93 GFlop/s
//LB = 3: Size = 1.125, Time = 0.554 msec, Performace = 4360.86 GFlop/s
template<int LB, int STEP>
__global__ void ksV2_Ims2_kernel_4_8R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

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
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);//CW[y, x, fhs, fws, 0, tic0]

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
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = (OW - tCFW)*OC;
	int fhr = Y_k / tCFW_OC;
	float4 dy; int yoffset = fhr * SY + Y_k;
	dy.x = deltaY[Y0 + yoffset];
	dy.y = deltaY[Y1 + yoffset];
	dy.z = deltaY[Y2 + yoffset];
	dy.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = dy;

	//load 2 elem from W
	int W_k = ty >> 1;
	const int SW = (CFW - tCFW)*OC;
	int woffset = (fhr * SW + W_k)*IC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + woffset);
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

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int fhr = Y_k / tCFW_OC;
		float4 dy; int yoffset = fhr * SY + Y_k;
		dy.x = deltaY[Y0 + yoffset];
		dy.y = deltaY[Y1 + yoffset];
		dy.z = deltaY[Y2 + yoffset];
		dy.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = dy;

		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		int woffset = (fhr * SW + W_k)*IC;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + woffset);
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

	int Xstride = IH_IW_slice * IC;//IH * IW * IC
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


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*4), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_KERNEL_4_4R
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_KERNEL_4_4R

//Target: Size = 1.125, Time = 0.484 msec, Performace = 4991.57 GFlop/s
//(OH, OW) = 2:
//LB = 4: Size = 1.125, Time = 0.51  msec, Performace = 4737.1 GFlop/s
//LB = 3: Size = 1.125, Time = 0.582 msec, Performace = 4151.06 GFlop/s
//(OH, OW) = 4:
//LB = 4: Size = 1.125, Time = 0.58  msec, Performace = 4165.38 GFlop/s
//LB = 3: Size = 1.125, Time = 0.662 msec, Performace = 3649.42 GFlop/s
//(OH, OW) = 8:
//k88R8_oc2pow<4>: Size = 1.125, Time = 0.724 msec, Performace = 3336.9 GFlop/s
//k88R8_oc2pow<3>: Size = 1.125, Time = 0.73  msec, Performace = 3309.48 GFlop/s
//LB = 4: Size = 1.125, Time = 0.718 msec, Performace = 3364.79 GFlop/s
//LB = 3: Size = 1.125, Time = 0.818 msec, Performace = 2953.45 GFlop/s
template<int LB, int STEP>
__global__ void ksV2_Ims2_kernel_4_4R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

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
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);//CW[y, x, fhs, fws, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int n0 = (((blockIdx.y << LB) + ty) << 2) + n_index;
	IH_IW_slice <<= 2;//IH_IW_slice * sh * sw -> IH * IW
	IW_slice <<= 1;//IW_slice * sw -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;
	int tn0 = n0 + ((tx & 1) << 1);
	int Y0 = ((tn0*OH + ohs)*OW + ows) * OC;
	int Y1 = Y0 + OH * OW * OC;

	//load 2 elem from deltaY
	int Y_k = tx >> 1;
	const int SY = (OW - tCFW)*OC;
	int fhr = Y_k / tCFW_OC;
	float2 dy; int yoffset = fhr * SY + Y_k;
	dy.x = deltaY[Y0 + yoffset];
	dy.y = deltaY[Y1 + yoffset];
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = dy;

	//load 2 elem from W
	int W_k = ty >> 1;
	const int SW = (CFW - tCFW)*OC;
	int woffset = (fhr * SW + W_k)*IC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + woffset);
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

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w);
			simdMM4(v2, y.z, w);
			simdMM4(v3, y.w, w);
		}
		buf ^= 1;

		//load 2 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		int fhr = Y_k / tCFW_OC;
		float2 dy; int yoffset = fhr * SY + Y_k;
		dy.x = deltaY[Y0 + yoffset];
		dy.y = deltaY[Y1 + yoffset];
		Ys[buf][Ys_x][Ys_y] = dy;

		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		int woffset = (fhr * SW + W_k)*IC;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + woffset);
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

	int Xstride = IH_IW_slice * IC;//IH * IW * IC
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