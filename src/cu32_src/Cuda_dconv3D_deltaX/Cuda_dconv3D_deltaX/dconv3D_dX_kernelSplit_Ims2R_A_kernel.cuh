#pragma once

#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2R_KERNEL_A_H
#define DCONV3D_DX_KERNEL_SPLIT_IMS2R_KERNEL_A_H

//IMS: INPUT_MOD_STEP
//R: remode: W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]
//S2: sh = sw = 2
//(IH, IW) % (sh, sw) == 0
//CWstride = CFH * CFW * OC * IC
//======[Improvement for Xoffset]===============================
//in kernel(8*8): j0 % 8 == 0
//ji = j0 + i = 8*x + i, i belongs to [0, 1,...7]
//when N % 8 == 0, N = 8*y
//<1> ni = ji % N = (8*x + i) % (8*y)
//So: ni = n0 + i
//<2> ihi = ji / (IW_slice * N) = (8*x + i) / (8*z) = x / z
//So: ni = nj
//<3> iwi = (ji %  (IW_slice * N)) / N = ((8*x + i) % 8*z) / (8*y)
//iwi = (8*u + i) / (8*y) = u / y
//So: iwi = iwj
//======[Improvement for Xoffset]===============================
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2R_A_CALL
#define DCONV3D_DX_KERNEL_SPLIT_IMS2R_A_CALL

//LB = log2(BLOCK_SIZE)

#define ksIms2A_88R(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM) \
	ksIms2_kernel_A_8_8R<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph, pw,\
			ic_index, j_index)

//======[FH = FW = (2||3||4) -> CFH = CFW = (2||1)]==========================
#define ksIms2A_88R_CW2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM) \
	ksIms2_kernel_A_8_8R_CW2pow<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph, pw,\
			ic_index, j_index)

#endif


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_A_8_8R
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_A_8_8R

//LB = 4: Size = 1.125, Time = 0.538 msec, Performace = 4490.56 GFlop/s
//LB = 3: Size = 1.125, Time = 0.604 msec, Performace = 3999.87 GFlop/s
//LB = 4: Size = 1.125, Time = 0.505 msec, Performace = 4784    GFlop/s(1000)
//LB = 3: Size = 1.125, Time = 0.564 msec, Performace = 4283.54 GFlop/s(1000)
template<int LB, int STEP>
__global__ void ksIms2_kernel_A_8_8R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_slice, int IW_slice,
	int N, int IC, int OC,
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
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int IW_slice_N = IW_slice * N;
	Ims2_ih_iw_n(j0, ih0, iw0, n0);
	IH_slice <<= 1;//IH_slice -> IH 
	IW_slice <<= 1;//IW_slice -> IW
	const int X0 = ((n0*IH_slice + ih0)*IW_slice + iw0)*IC + ic0;
	const int Xstride = IH_slice * IW_slice * IC;
	int tohs = ((ih0 + ph) >> 1) - oph, tows = ((iw0 + pw) >> 1) - opw;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tohs)*OW + tows)*OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	float4 yv; bool ly = LOAD_Y(tohs, tows, Y_fhr, Y_fwr);
	yv.x = (ly ? deltaY[Y0 + yoffset] : 0);
	yv.y = (ly ? deltaY[Y1 + yoffset] : 0);
	yv.z = (ly ? deltaY[Y2 + yoffset] : 0);
	yv.w = (ly ? deltaY[Y3 + yoffset] : 0);
	Ys[buf][tx][ty] = yv;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
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

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		float4 yv; bool ly = LOAD_Y(tohs, tows, Y_fhr, Y_fwr);
		yv.x = (ly ? deltaY[Y0 + yoffset] : 0);
		yv.y = (ly ? deltaY[Y1 + yoffset] : 0);
		yv.z = (ly ? deltaY[Y2 + yoffset] : 0);
		yv.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = yv;

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


//======[FH = FW = (2||3||4) -> CFH = CFW = (2||1)]==========================
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0, N % 8 == 0
//LB = 4: OC % 8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_A_8_8R_CW2POW
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_A_8_8R_CW2POW

//LB = 4: Size = 1.125, Time = 0.514 msec, Performace = 4700.23 GFlop/s
//LB = 3: Size = 1.125, Time = 0.604 msec, Performace = 3999.87 GFlop/s
template<int LB, int STEP>
__global__ void ksIms2_kernel_A_8_8R_CW2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_slice, int IW_slice,
	int N, int IC, int OC,
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
	const int LCFW = (CFW >> 1), LCFH_CFW = (CFH >> 1) + LCFW;
	const int CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int IW_slice_N = IW_slice * N;
	Ims2_ih_iw_n(j0, ih0, iw0, n0);
	IH_slice <<= 1;//IH_slice -> IH 
	IW_slice <<= 1;//IW_slice -> IW
	const int X0 = ((n0*IH_slice + ih0)*IW_slice + iw0)*IC + ic0;
	const int Xstride = IH_slice * IW_slice * IC;
	int tohs = ((ih0 + ph) >> 1) - oph, tows = ((iw0 + pw) >> 1) - opw;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tohs)*OW + tows)*OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr_CW2pow(Y_k, Y_oc, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr*OW + Y_fwr)*OC + Y_oc;
	float4 yv; bool ly = LOAD_Y(tohs, tows, Y_fhr, Y_fwr);
	yv.x = (ly ? deltaY[Y0 + yoffset] : 0);
	yv.y = (ly ? deltaY[Y1 + yoffset] : 0);
	yv.z = (ly ? deltaY[Y2 + yoffset] : 0);
	yv.w = (ly ? deltaY[Y3 + yoffset] : 0);
	Ys[buf][tx][ty] = yv;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int woffset = ((W_k & CFH_CFW_m1)*OC + (W_k >> LCFH_CFW))*IC;
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

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr_CW2pow(Y_k, Y_oc, Y_fhr, Y_fwr);
		int yoffset = (Y_fhr*OW + Y_fwr)*OC + Y_oc;
		float4 yv; bool ly = LOAD_Y(tohs, tows, Y_fhr, Y_fwr);
		yv.x = (ly ? deltaY[Y0 + yoffset] : 0);
		yv.y = (ly ? deltaY[Y1 + yoffset] : 0);
		yv.z = (ly ? deltaY[Y2 + yoffset] : 0);
		yv.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = yv;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = ((W_k & CFH_CFW_m1)*OC + (W_k >> LCFH_CFW))*IC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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
