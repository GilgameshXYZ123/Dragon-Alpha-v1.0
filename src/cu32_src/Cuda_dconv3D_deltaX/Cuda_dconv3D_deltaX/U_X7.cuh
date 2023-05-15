

//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef XKERNEL1
#define XKERNEL1

#define uxkernel1(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM) \
	UXkernel1<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH_slice, IW_slice, IC, OC, ph, pw,\
			ic_index, j_index)

//LB = 4: Size = 1.125, Time = 0.56 msec, Performace = 4314.14 GFlop/s
//LB = 3: Size = 1.125, Time = 0.642 msec, Performace = 3763.11 GFlop/s
template<int LB, int STEP>
__global__ void UXkernel1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_slice, int IW_slice,
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
	int IH_IW_slice = IH_slice * IW_slice;
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef XKERNEL2
#define XKERNEL2

#define uxkernel2(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	UXkernel2<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH_slice, IW_slice, IC, OC, ph, pw,\
			ic_index, j_index)

//LB = 4: Size = 1.125, Time = 0.542 msec, Performace = 4457.42 GFlop/s
//LB = 3: Size = 1.125, Time = 0.642 msec, Performace = 3763.11 GFlop/s
template<int LB, int STEP>
__global__ void UXkernel2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_slice, int IW_slice,
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
	int IH_IW_slice = IH_slice * IW_slice;
	int yx = bz / IH_IW_slice; bz %= IH_IW_slice;
	int ih = bz / IW_slice, iw = bz % IW_slice;
	CW += yx * CWstride;//CW[y, x]
	
	int y = yx >> 1, x = yx & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));
	ih = (ih << 1) + ihs;
	iw = (iw << 1) + iws;
	
	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	IH_IW_slice <<= 2;//IH_IW_slice -> IH * IW
	IW_slice <<= 1;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;
	int Xstride = IH_IW_slice * IC;//IH * IW * IC

	int tn0 = n0 + ((tx >= STEP) << 2);
	int ohs = ((ih + ph) >> 1) - oph;//the start Idx of oh
	int ows = ((iw + pw) >> 1) - opw;//the start Idx of ow
	int Y0 = ((tn0*OH + ohs)*OW + ows) * OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	
	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr); 
	bool ly = (ohs >= -Y_fhr) && (ohs < OH - Y_fhr) && (ows >= -Y_fwr) && (ows < OW - Y_fwr);
	float4 yv; int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	yv.x = (ly ? deltaY[Y0 + yoffset] : 0);
	yv.y = (ly ? deltaY[Y1 + yoffset] : 0);
	yv.z = (ly ? deltaY[Y2 + yoffset] : 0);
	yv.w = (ly ? deltaY[Y3 + yoffset] : 0);
	Ys[buf][tx][ty] = yv;
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
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		bool ly = (ohs >= -Y_fhr) && (ohs < OH - Y_fhr) && (ows >= -Y_fwr) && (ows < OW - Y_fwr);
		float4 yv; int yoffset = Y_fhr * OW_OC + Y_k;
		yv.x = (ly ? deltaY[Y0 + yoffset] : 0);
		yv.y = (ly ? deltaY[Y1 + yoffset] : 0);
		yv.z = (ly ? deltaY[Y2 + yoffset] : 0);
		yv.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = yv;
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef XKERNEL3
#define XKERNEL3

#define uxkernel3(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	UXkernel3<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, j_index)

//Target: Size = 1.125, Time = 0.484 msec, Performace = 4991.57 GFlop/s
//LB = 4: Size = 1.125, Time = 0.524 msec, Performace = 4610.53 GFlop/s
//LB = 3: Size = 1.125, Time = 0.642 msec, Performace = 3763.11 GFlop/s
template<int LB, int STEP>
__global__ void UXkernel3(
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

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	IH_IW_slice <<= 2;//IH_IW_slice -> IH * IW
	IW_slice <<= 1;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;

	int tn0 = n0 + ((tx >= STEP) << 2);
	int ohs = ((ih + ph) >> 1) - oph;//the start Idx of oh
	int ows = ((iw + pw) >> 1) - opw;//the start Idx of ow
	int Y0 = ((tn0*OH + ohs)*OW + ows) * OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	bool ly = (ohs >= -Y_fhr) && (ohs < OH - Y_fhr) && (ows >= -Y_fwr) && (ows < OW - Y_fwr);
	float4 yv; int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
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
		bool ly = (ohs >= -Y_fhr) && (ohs < OH - Y_fhr) && (ows >= -Y_fwr) && (ows < OW - Y_fwr);
		float4 yv; int yoffset = Y_fhr * OW_OC + Y_k;
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

	int Xstride = IH_IW_slice * IC;//IH * IW * IC
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef XKERNEL4
#define XKERNEL4

#define uxkernel4(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	UXkernel4<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, j_index)

//Target: Size = 1.125, Time = 0.484 msec, Performace = 4991.57 GFlop/s
//LB = 4: Size = 1.125, Time = 0.524 msec, Performace = 4610.53 GFlop/s
//LB = 3: Size = 1.125, Time = 0.642 msec, Performace = 3763.11 GFlop/s
template<int LB, int STEP>
__global__ void UXkernel4(
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

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	IH_IW_slice <<= 2;//IH_IW_slice -> IH * IW
	IW_slice <<= 1;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;

	int tn0 = n0 + ((tx >= STEP) << 2);
	int ohs = ((ih + ph) >> 1) - oph;//the start Idx of oh
	int ows = ((iw + pw) >> 1) - opw;//the start Idx of ow
	int Y0 = ((tn0*OH + ohs)*OW + ows) * OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;
	int woffset = ((W_fhr*CFW + W_fwr)*OC + W_oc)*IC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_fhr = Y_k / CFW_OC; Y_k -= Y_fhr * CFW_OC;
	int Y_fwr = Y_k / OC;
	bool ly = (ohs >= -Y_fhr) && (ohs < OH - Y_fhr) && (ows >= -Y_fwr) && (ows < OW - Y_fwr);
	float4 yv; int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	yv.x = (ly ? deltaY[Y0 + yoffset] : 0);
	yv.y = (ly ? deltaY[Y1 + yoffset] : 0);
	yv.z = (ly ? deltaY[Y2 + yoffset] : 0);
	yv.w = (ly ? deltaY[Y3 + yoffset] : 0);
	Ys[buf][tx][ty] = yv;
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
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;
		int woffset = ((W_fhr*CFW + W_fwr)*OC + W_oc)*IC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_fhr = Y_k / CFW_OC; Y_k -= Y_fhr * CFW_OC;
		int Y_fwr = Y_k / OC;
		bool ly = (ohs >= -Y_fhr) && (ohs < OH - Y_fhr) && (ows >= -Y_fwr) && (ows < OW - Y_fwr);
		float4 yv; int yoffset = Y_fhr * OW_OC + Y_k;
		yv.x = (ly ? deltaY[Y0 + yoffset] : 0);
		yv.y = (ly ? deltaY[Y1 + yoffset] : 0);
		yv.z = (ly ? deltaY[Y2 + yoffset] : 0);
		yv.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = yv;
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

	int Xstride = IH_IW_slice * IC;//IH * IW * IC
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef XKERNEL5
#define XKERNEL5

#define uxkernel5(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	UXkernel5<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, j_index)

//Target: Size = 1.125, Time = 0.484 msec, Performace = 4991.57 GFlop/s
//(OH, OW) = 2
//LB = 4: Size = 1.125, Time = 0.382 msec, Performace = 6324.4 GFlop/s
//LB = 3: Size = 1.125, Time = 0.448 msec, Performace = 5392.68 GFlop/s
//(OH, OW) = 4:
//LB = 4: Size = 1.125, Time = 0.472 msec, Performace = 5118.47 GFlop/s
//LB = 3: Size = 1.125, Time = 0.514 msec, Performace = 4700.23 GFlop/s
template<int LB, int STEP>
__global__ void UXkernel5(
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

	//(1) ohs + fhr >= 0
	//if ohs >= 0: fhr >= 0
	//if ohs <  0: fhr >= -ohs
	int fhs = -IF_int((ohs < 0), ohs, 0);
	int fws = -IF_int((ows < 0), ows, 0);
	
	//(2) ohs + fhr < OH
	//if ohs >= 0: fhr < OH - ohs
	//if ohs <  0: fhr < CFH
	//in all case: fhr < CFH
	int fhe = OH - ohs; fhe = IF_int((fhe > CFH), CFH, fhe);
	int fwe = OW - ows; fwe = IF_int((fwe > CFW), CFW, fwe);
	const int tCFH = fhe - fhs;
	const int tCFW = fwe - fws;
	const int tCFW_OC = tCFW * OC, GK = tCFH * tCFW_OC;
	CW += (fhs*CFW + fws)*OC*IC;//CW[y, x, fhs, fws, 0, 0]
	deltaY += (fhs*OW + fws)*OC;//deltaY[0, fhs, fws, 0]

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	IH_IW_slice <<= 2;//IH_IW_slice -> IH * IW
	IW_slice <<= 1;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + ohs)*OW + ows) * OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_fhr = W_k / tCFW_OC; W_k -= W_fhr * tCFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;
	int woffset = ((W_fhr*CFW + W_fwr)*OC + W_oc)*IC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_fhr = Y_k / tCFW_OC; Y_k -= Y_fhr * tCFW_OC;
	int Y_fwr = Y_k / OC;
	float4 dy; int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	dy.x = deltaY[Y0 + yoffset];
	dy.y = deltaY[Y1 + yoffset];
	dy.z = deltaY[Y2 + yoffset];
	dy.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = dy;
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
		int W_fhr = W_k / tCFW_OC; W_k -= W_fhr * tCFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;
		int woffset = ((W_fhr*CFW + W_fwr)*OC + W_oc)*IC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_fhr = Y_k / tCFW_OC; Y_k -= Y_fhr * tCFW_OC;
		int Y_fwr = Y_k / OC;
		float4 dy; int yoffset = Y_fhr * OW_OC + Y_k;
		dy.x = deltaY[Y0 + yoffset];
		dy.y = deltaY[Y1 + yoffset];
		dy.z = deltaY[Y2 + yoffset];
		dy.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = dy;
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

	int Xstride = IH_IW_slice * IC;//IH * IW * IC
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef XKERNEL6
#define XKERNEL6

#define uxkernel6(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	UXkernel6<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, j_index)

//Target: Size = 1.125, Time = 0.484 msec, Performace = 4991.57 GFlop/s
//(OH, OW) = 4:
//LB = 4: Size = 1.125, Time = 0.476 msec, Performace = 5075.46 GFlop/s
//LB = 3: Size = 1.125, Time = 0.514 msec, Performace = 4700.23 GFlop/s
template<int LB, int STEP>
__global__ void UXkernel6(
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
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1)); ih = (ih << 1) + ihs;
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1)); iw = (iw << 1) + iws;

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
	IH_IW_slice <<= 2;//IH_IW_slice -> IH * IW
	IW_slice <<= 1;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + ohs)*OW + ows) * OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_fhr = W_k / tCFW_OC; W_k -= W_fhr * tCFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;
	int woffset = (W_fhr*CFW*OC + W_k)*IC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int OW_OC = OW * OC;
	int Y_fhr = Y_k / tCFW_OC; Y_k -= Y_fhr * tCFW_OC;
	float4 dy; int yoffset = Y_fhr * OW_OC + Y_k;
	dy.x = deltaY[Y0 + yoffset];
	dy.y = deltaY[Y1 + yoffset];
	dy.z = deltaY[Y2 + yoffset];
	dy.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = dy;
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

		//woffset = ((fhr*CFW + fwr)*OC + oc)*IC = U * IC
		//U= fhr*OCFW*C + (fwr*OC + oc)
		//(fwr*OC + oc) = W_k - W_fhr * tCFW_OC;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fhr = W_k / tCFW_OC; W_k -= W_fhr * tCFW_OC;
		int woffset = (W_fhr*CFW*OC + W_k)*IC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_fhr = Y_k / tCFW_OC; Y_k -= Y_fhr * tCFW_OC;
		float4 dy; int yoffset = Y_fhr * OW_OC + Y_k;
		dy.x = deltaY[Y0 + yoffset];
		dy.y = deltaY[Y1 + yoffset];
		dy.z = deltaY[Y2 + yoffset];
		dy.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = dy;
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

	int Xstride = IH_IW_slice * IC;//IH * IW * IC
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef XKERNEL7
#define XKERNEL7

#define uxkernel7(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	UXkernel7<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, j_index)

//Target: Size = 1.125, Time = 0.484 msec, Performace = 4991.57 GFlop/s
//(OH, OW) = 4:
//LB = 4: Size = 1.125, Time = 0.472 msec, Performace = 5118.47 GFlop/s
//LB = 3: Size = 1.125, Time = 0.514 msec, Performace = 4700.23 GFlop/s
template<int LB, int STEP>
__global__ void UXkernel7(
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
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1)); ih = (ih << 1) + ihs;
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1)); iw = (iw << 1) + iws;

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
	IH_IW_slice <<= 2;//IH_IW_slice -> IH * IW
	IW_slice <<= 1;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + ohs)*OW + ows) * OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;


	//woffset = (W_fhr*CFW*OC + W_k - W_fhr*tCFW*OC)*IC = U * IC
	//U = W_fhr*CFW*OC + W_k - W_fhr*tCFW*OC;
	//U = W_fhr * (CFW - tCFW) * OC + W_k
	//let: SW = (CFW - tCFW) * OC
	//woffset = (W_fhr * SW + W_k)*IC

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	const int SW = (CFW - tCFW) * OC;
	int W_fhr = W_k / tCFW_OC; 
	int woffset = (W_fhr * SW + W_k)*IC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);

	//yoffset = Y_fhr * OW_OC + Y_k
	//= Y_fhr * OW * OC + Y_k - Y_fhr * tCFW * OC;
	//= Y_k + Y_fhr * OC * (OW - tCFW)
	//let: SY = (OW - tCFW)*OC
	//yoffset = Y_k + Y_fhr * SY

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = (OW - tCFW)*OC;
	int Y_fhr = Y_k / tCFW_OC;
	float4 dy; int yoffset = Y_fhr * SY + Y_k;
	dy.x = deltaY[Y0 + yoffset];
	dy.y = deltaY[Y1 + yoffset];
	dy.z = deltaY[Y2 + yoffset];
	dy.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = dy;
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
		int W_fhr = W_k / tCFW_OC; 
		int woffset = (W_fhr * SW + W_k)*IC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_fhr = Y_k / tCFW_OC; 
		float4 dy; int yoffset = Y_fhr * SY + Y_k;
		dy.x = deltaY[Y0 + yoffset];
		dy.y = deltaY[Y1 + yoffset];
		dy.z = deltaY[Y2 + yoffset];
		dy.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = dy;
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

	int Xstride = IH_IW_slice * IC;//IH * IW * IC
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef XKERNEL8
#define XKERNEL8

#define uxkernel8(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	UXkernel8<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, j_index)

//Target: Size = 1.125, Time = 0.484 msec, Performace = 4991.57 GFlop/s
//(OH, OW) = 4:
//LB = 4: Size = 1.125, Time = 0.446 msec, Performace = 5416.86 GFlop/s
//LB = 3: Size = 1.125, Time = 0.482 msec, Performace = 5012.28 GFlop/s
template<int LB, int STEP>
__global__ void UXkernel8(
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
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1)); ih = (ih << 1) + ihs;
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1)); iw = (iw << 1) + iws;

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
	IH_IW_slice <<= 2;//IH_IW_slice -> IH * IW
	IW_slice <<= 1;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + ohs)*OW + ows) * OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//W_k = ok*STEP + Uy
	//Y_k = ok*STEP + Ux
	//Uy, Ux belongs to [0, STEP - 1]
	//when: OC % STEP == 0, weh have tCFW_OC % STEP == 0
	//W_fhr = W_k / tCFW_OC = (ok*STEP + Uy) / STEP*z
	//Y_fhr = Y_k / tCFW_OC = (ok*STEP + Ux) / STEP*z
	//we have: W_fhr = Y_fhr = fhr

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
	const int SW = (CFW - tCFW) * OC;
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


//======when FH = FW = 3=============
//CFH = CFW = 2 or 1
//oph = opw = 1 or 0
//when CFH = 2, oph = 1 
//we have: CFH -> tCFH = 2, 1 
//so: tCFH, tCFW is power of 2
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef XKERNEL9
#define XKERNEL9

#define uxkernel9(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, N) \
	UXkernel9<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, LOC, ph, pw,\
			ic_index, j_index)

//Target: Size = 1.125, Time = 0.484 msec, Performace = 4991.57 GFlop/s
//(OH, OW) = 2:
//LB = 4: Size = 1.125, Time = 0.354 msec, Performace = 6824.63 GFlop/s
//LB = 3: Size = 1.125, Time = 0.396 msec, Performace = 6100.81 GFlop/s
//(OH, OW) = 4:
//LB = 4: Size = 1.125, Time = 0.434 msec, Performace = 5566.63 GFlop/s
//LB = 3: Size = 1.125, Time = 0.46  msec, Performace = 5252 GFlop/s
template<int LB, int STEP>
__global__ void UXkernel9(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int LOC,
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

	int y = yx >> 1, x = yx & 1;//x = yx % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1)); ih = (ih << 1) + ihs;
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1)); iw = (iw << 1) + iws;

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

	CW += ((fhs*CFW + fws)*IC) << LOC;//CW[y, x, fhs, fws, 0, 0]
	deltaY += (fhs*OW + fws) << LOC;//deltaY[0, fhs, fws, 0]
	
	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, fhs, fws, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	IH_IW_slice <<= 2;//IH_IW_slice -> IH * IW
	IW_slice <<= 1;//IW_slice -> IW
	int X0 = ((n0*IH_IW_slice) + ih * IW_slice + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + ohs)*OW + ows) << LOC;
	int Ystride = (OH * OW) << LOC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = (OW - tCFW) << LOC;
	int fhr = Y_k >> LtCFW_OC;
	float4 dy; int yoffset = fhr * SY + Y_k;
	dy.x = deltaY[Y0 + yoffset];
	dy.y = deltaY[Y1 + yoffset];
	dy.z = deltaY[Y2 + yoffset];
	dy.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = dy;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	const int SW = (CFW - tCFW) << LOC;
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
		int fhr = Y_k >> LtCFW_OC;
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