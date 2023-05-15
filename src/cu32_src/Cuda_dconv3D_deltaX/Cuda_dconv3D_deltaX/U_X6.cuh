
//k88s1_tex<4>: Size = 1.125, Time = 1.86 msec, Performace = 1298.88 GFlop/s
//k88s1W3x4_tex<4>: Size = 1.125, Time = 1.778 msec, Performace = 1358.78 GFlop/s
//k88s1W3x4_oc2pow_tex<4>: Size = 1.125, Time = 1.65 msec, Performace = 1464.19 GFlop/s


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef XKERNEL1
#define XKERNEL1

//int GN = IC;
//int GM = N * IH*IW;
//int GK = OC * FH*FW;

#define xkernel1(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N) \
	Xkernel1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,n_index)

//LB = 4: Size = 1.125, Time = 1.898 msec, Performace = 1272.88 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1), dY_fh, dY_fw, dY_oc;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	bool ly = (tih >= -dY_fh) && (tih < OH - dY_fh) && (tiw >= -dY_fw) && (tiw < OW - dY_fw);
	float4 x;
	x.x = (ly ? deltaY[Y0 + yoffset] : 0);
	x.y = (ly ? deltaY[Y1 + yoffset] : 0);
	x.z = (ly ? deltaY[Y2 + yoffset] : 0);
	x.w = (ly ? deltaY[Y3 + yoffset] : 0);
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1), W_oc, Wr_fh_fw; //Ws: with the same ty
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	int woffset = (W_oc*FH_FW + Wr_fh_fw)*IC;
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
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx, dY_fh, dY_fw, dY_oc;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		bool ly = (tih >= -dY_fh) && (tih < OH - dY_fh) && (tiw >= -dY_fw) && (tiw < OW - dY_fw);
		float4 x;
		x.x = (ly ? deltaY[Y0 + yoffset] : 0);
		x.y = (ly ? deltaY[Y1 + yoffset] : 0);
		x.z = (ly ? deltaY[Y2 + yoffset] : 0);
		x.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		int woffset = (W_oc*FH_FW + Wr_fh_fw)*IC;
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

	int Xstride = IH * IW * IC;
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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef XKERNEL2
#define XKERNEL2

//int GN = IC;
//int GM = N * IH*IW;
//int GK = OC * FH*FW;

#define xkernel2(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N) \
	Xkernel2<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

//LB = 4: Size = 1.125, Time = 1.898 msec, Performace = 1272.88 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_oc = Y_k / FH_FW; Y_k -= Y_oc * FH_FW; 
	int Y_fh = Y_k / FW, Y_fw = Y_k - Y_fh * FW;
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	bool ly = (tih >= -Y_fh) && (tih < OH - Y_fh) && (tiw >= -Y_fw) && (tiw < OW - Y_fw);
	float4 x;
	x.x = (ly ? deltaY[Y0 + yoffset] : 0);
	x.y = (ly ? deltaY[Y1 + yoffset] : 0);
	x.z = (ly ? deltaY[Y2 + yoffset] : 0);
	x.w = (ly ? deltaY[Y3 + yoffset] : 0);
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int W_oc = W_k / FH_FW; W_k -= W_oc * FH_FW;
	int Wr_fh_fw = FH_FW - 1 - W_k;
	int woffset = (W_oc*FH_FW + Wr_fh_fw)*IC;
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
		int Y_oc = Y_k / FH_FW; Y_k -= Y_oc * FH_FW;
		int Y_fh = Y_k / FW, Y_fw = Y_k - Y_fh * FW;
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		bool ly = (tih >= -Y_fh) && (tih < OH - Y_fh) && (tiw >= -Y_fw) && (tiw < OW - Y_fw);
		float4 x;
		x.x = (ly ? deltaY[Y0 + yoffset] : 0);
		x.y = (ly ? deltaY[Y1 + yoffset] : 0);
		x.z = (ly ? deltaY[Y2 + yoffset] : 0);
		x.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k / FH_FW; W_k -= W_oc * FH_FW;
		int Wr_fh_fw = FH_FW - 1 - W_k;
		int woffset = (W_oc*FH_FW + Wr_fh_fw)*IC;
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

	int Xstride = IH * IW * IC;
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


#ifndef XKERNEL3
#define XKERNEL3

//int GN = IC;
//int GM = N * IH*IW;
//int GK = OC * FH*FW;

#define xkernel3(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N) \
	Xkernel3<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

//LB = 4: Size = 1.125, Time = 1.898 msec, Performace = 1272.88 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_oc = Y_k / FH_FW; Y_k -= Y_oc * FH_FW;
	int Y_fh = Y_k / FW, Y_fw = Y_k - Y_fh * FW;
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	bool ly = (tih >= -Y_fh) && (tih < OH - Y_fh) && (tiw >= -Y_fw) && (tiw < OW - Y_fw);
	float4 x;
	x.x = (ly ? deltaY[Y0 + yoffset] : 0);
	x.y = (ly ? deltaY[Y1 + yoffset] : 0);
	x.z = (ly ? deltaY[Y2 + yoffset] : 0);
	x.w = (ly ? deltaY[Y3 + yoffset] : 0);
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int W_oc = W_k / FH_FW; W_k -= W_oc * FH_FW;
	int W_fh = W_k / FW, W_fw = W_k - W_fh * FW;
	W_fh = FH - 1 - W_fh;
	W_fw = FW - 1 - W_fw;
	int woffset = ((W_oc*FH + W_fh)*FW + W_fw)*IC;
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
		int Y_oc = Y_k / FH_FW; Y_k -= Y_oc * FH_FW;
		int Y_fh = Y_k / FW, Y_fw = Y_k - Y_fh * FW;
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		bool ly = (tih >= -Y_fh) && (tih < OH - Y_fh) && (tiw >= -Y_fw) && (tiw < OW - Y_fw);
		float4 x;
		x.x = (ly ? deltaY[Y0 + yoffset] : 0);
		x.y = (ly ? deltaY[Y1 + yoffset] : 0);
		x.z = (ly ? deltaY[Y2 + yoffset] : 0);
		x.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k / FH_FW; W_k -= W_oc * FH_FW;
		int W_fh = W_k / FW, W_fw = W_k - W_fh * FW;
		W_fh = FH - 1 - W_fh;
		W_fw = FW - 1 - W_fw;
		int woffset = ((W_oc*FH + W_fh)*FW + W_fw)*IC;
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

	int Xstride = IH * IW * IC;
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


#ifndef XKERNEL4
#define XKERNEL4

//int GN = IC;
//int GM = N * IH*IW;
//int GK = OC * FH*FW;

#define xkernel4(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N) \
	Xkernel4<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

//LB = 4: Size = 1.125, Time = 1.898 msec, Performace = 1272.88 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GK = FH * FW * OC
	//(1) tih + fh >= 0
	//if: tih <  0: fh >= -tih -> fhs = -tih
	//if: tih >= 0: fh >= 0    -> fhs = 0
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);

	//(2) tih + fh < OH
	//if: tih <  0: fh <  OH       
	//if: tih >= 0: fh <= OH - tih
	//And: fh < FH
	int fhe = OH - tih; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > FW), FW, fwe);

	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFH_FW = tFH * tFW, GK = tFH_FW * OC;
	deltaY += (fhs*OW + fws)*OC;//deltaY[0, fhs, fws, 0]

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_oc = Y_k / tFH_FW; Y_k -= Y_oc * tFH_FW;
	int Y_fh = Y_k / tFW, Y_fw = Y_k - Y_fh * tFW;
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	float4 x;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int W_oc = W_k / tFH_FW; W_k -= W_oc * tFH_FW;
	int W_fh = W_k / tFW, W_fw = W_k - W_fh * tFW;
	W_fh = FH - 1 - W_fh - fhs;
	W_fw = FW - 1 - W_fw - fws;
	int woffset = ((W_oc*FH + W_fh)*FW + W_fw)*IC;
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
		int Y_oc = Y_k / tFH_FW; Y_k -= Y_oc * tFH_FW;
		int Y_fh = Y_k / tFW, Y_fw = Y_k - Y_fh * tFW;
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		float4 x;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k / tFH_FW; W_k -= W_oc * tFH_FW;
		int W_fh = W_k / tFW, W_fw = W_k - W_fh * tFW;
		W_fh = FH - 1 - W_fh - fhs;
		W_fw = FW - 1 - W_fw - fws;
		int woffset = ((W_oc*FH + W_fh)*FW + W_fw)*IC;
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

	int Xstride = IH * IW * IC;
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


#ifndef XKERNEL5
#define XKERNEL5

//int GN = IC;
//int GM = N * IH*IW;
//int GK = OC * FH*FW;

#define xkernel5(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N) \
	Xkernel5<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

//LB = 4: Size = 1.125, Time = 1.648 msec, Performace = 1465.97 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel5(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFH_FW = tFH * tFW, GK = tFH_FW * OC;
	deltaY += (fhs*OW + fws)*OC;//deltaY[0, fhs, fws, 0]
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws)) *OC;//W[0, -fhs, -fws, 0]

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_oc = Y_k / tFH_FW; Y_k -= Y_oc * tFH_FW;
	int Y_fh = Y_k / tFW, Y_fw = Y_k - Y_fh * tFW;
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	float4 x;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int W_oc = W_k / tFH_FW; W_k -= W_oc * tFH_FW;
	int W_fh = W_k / tFW, W_fw = W_k - W_fh * tFW;
	int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
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
		int Y_oc = Y_k / tFH_FW; Y_k -= Y_oc * tFH_FW;
		int Y_fh = Y_k / tFW, Y_fw = Y_k - Y_fh * tFW;
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		float4 x;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k / tFH_FW; W_k -= W_oc * tFH_FW;
		int W_fh = W_k / tFW, W_fw = W_k - W_fh * tFW;
		int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
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

	int Xstride = IH * IW * IC;
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


#ifndef XKERNEL6
#define XKERNEL6

//int GN = IC;
//int GM = N * IH*IW;
//int GK = OC * FH*FW;

#define xkernel6(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N) \
	Xkernel6<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

//LB = 4: Size = 1.125, Time = 1.26 msec, Performace = 1917.4 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel6(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;
	deltaY += (fhs*OW + fws)*OC;//deltaY[0, fhs, fws, 0]
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws)) *OC;//W[0, -fhs, -fws, 0]

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_fh = Y_k / tFW_OC; Y_k -= Y_fh * tFW_OC;
	int Y_fw = Y_k / OC, Y_oc = Y_k - Y_fw * OC;
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	float4 x;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int W_fh = W_k / tFW_OC; W_k -= W_fh * tFW_OC;
	int W_fw = W_k / OC, W_oc = W_k - W_fw * OC;
	int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
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
		int Y_fh = Y_k / tFW_OC; Y_k -= Y_fh * tFW_OC;
		int Y_fw = Y_k / OC, Y_oc = Y_k - Y_fw * OC;
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		float4 x;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh = W_k / tFW_OC; W_k -= W_fh * tFW_OC;
		int W_fw = W_k / OC, W_oc = W_k - W_fw * OC;
		int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
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

	int Xstride = IH * IW * IC;
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


#ifndef XKERNEL7
#define XKERNEL7

//int GN = IC;
//int GM = N * IH*IW;
//int GK = OC * FH*FW;

#define xkernel7(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N) \
	Xkernel7<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

//LB = 4: Size = 1.125, Time = 1.23 msec, Performace = 1964.16 GFlop/s
//LB = 3: Size = 1.125, Time = 1.51 msec, Performace = 1599.95 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel7(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;
	deltaY += (fhs*OW + fws)*OC;//deltaY[0, fhs, fws, 0]
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws)) *OC;//W[0, -fhs, -fws, 0]

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = OC * (OW - tFW);
	int Y_fh = Y_k / tFW_OC;
	float4 x; int yoffset = Y_fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int W_fh = W_k / tFW_OC; W_k -= W_fh * tFW_OC;
	int W_fw = W_k / OC, W_oc = W_k - W_fw * OC;
	int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
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
		int Y_fh = Y_k / tFW_OC;
		float4 x; int yoffset = Y_fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
		//(W_oc*FH - W_fh)*FW - W_fw
		//W_oc*FH*FW - W_fh*FW - W_fw
		//(W_k - W_fw * OC)*FH*FW - W_fh*FW - W_fw
		//W_k*FH*FW - W_fw*FH*FW*OC - W_fh*FW - Wfw
		//W_k*FH*FW - W_fh*FW - W_fw*(FH*FW*OC + 1) 
		//(W_k*FH - Wfh)*FW - W_fw*(FH*FW*OC + 1)

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh = W_k / tFW_OC; W_k -= W_fh * tFW_OC;
		int W_fw = W_k / OC, W_oc = W_k - W_fw * OC;
		int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
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

	int Xstride = IH * IW * IC;
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


#ifndef XKERNEL8
#define XKERNEL8

//int GN = IC;
//int GM = N * IH*IW;
//int GK = OC * FH*FW;

#define xkernel8(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N) \
	Xkernel8<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

//LB = 4: Size = 1.125, Time = 1.192 msec, Performace = 2026.78 GFlop/s
//LB = 3: Size = 1.125, Time = 1.51 msec, Performace = 1599.95 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel8(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;
	deltaY += (fhs*OW + fws)*OC;//deltaY[0, fhs, fws, 0]
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws)) *OC;//W[0, -fhs, -fws, 0]

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//Y_k = ok*STEP + tx - ((tx >= STEP) << LB >> 1)
	//W_k = ok*STEP + ty - ((ty >= STEP) << LB >> 1)
	//let: Ux = tx - ((tx >= STEP) << LB >> 1)
	//let: Uy = ty - ((ty >= STEP) << LB >> 1)
	//So: Y_k = ok*STEP + Ux
	//So: W_k = ok*STEP + Uy
	//As: Ux, Uy belongs to [0, STEP - 1]
	//(1) when: LB = 4, STEP = 8; OC % 8 == 0, tFW_OC % 8 == 0
	//we have:
	//Y_fh = Y_k / tFW_OC = (ok*STEP + Ux) / tFW_OC = (8*ok + Ux) / 8*y
	//W_fh = W_k / tFW_OC = (ok*STEP + Uy) / tFW_OC = (8*ok + Uy) / 8*y
	//So: Y_fh = W_fh
	//(2) when: LB = 3, STEP = 4; OC % 4 == 0, tFW_OC % 4 == 0
	//we also have: W_fh = Y_fh


	//we also have:
	//(1) when: LB = 4, STEP = 8; OC % 8 == 0, tFW_OC % 8 == 0
	//Y_fw = (Y_k % tFW_OC) / OC = ((8*ok + Ux) % 8*y)/8*z
	//Y_fw = (8x + Ux) / 8z 
	//W_fw = (8x + Uy) / 8z
	//So: Y_fw = W_fw

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = OC * (OW - tFW);
	int fh = Y_k / tFW_OC;
	float4 x; int yoffset = fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int W_fw = (W_k -= fh * tFW_OC) / OC, W_oc = W_k - W_fw * OC;
	int woffset = ((W_oc*FH - fh)*FW - W_fw)*IC;
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
		int fh = Y_k / tFW_OC;
		float4 x; int yoffset = fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fw = (W_k -= fh * tFW_OC) / OC, W_oc = W_k - W_fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - W_fw)*IC;
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

	int Xstride = IH * IW * IC;
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


#ifndef XKERNEL9
#define XKERNEL9

#define xkernel9(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N) \
	Xkernel9<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

//LB = 4: Size = 1.125, Time = 1.174 msec, Performace = 2057.85 GFlop/s
//LB = 3: Size = 1.125, Time = 1.406 msec, Performace = 1718.29 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel9(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;
	deltaY += (fhs*OW + fws)*OC;//deltaY[0, fhs, fws, 0]
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws)) * IC;//W[0, -fhs, -fws, 0]

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

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = OC * (OW - tFW);
	int fh = Y_k / tFW_OC;
	float4 x; int yoffset = fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//woffset = U * IC
	//U = (W_oc*FH - fh)*FW - W_fw
	//W_oc*FH*FW - fh*FW - W_fw
	//(W_k - W_fw * OC)*FH*FW - fh*FW - W_fw
	//W_k*FH*FW - fh*FW - W_fw*OC*FH*FW  - W_fw
	//xoffset = W_k*FH*FW*IC - fh*FW*IC - W_fw*(OC*FH*FW + 1)*IC
	//let:
	//FH0 -> FH*FW*IC
	//FW0 -> FW*IC
	//IC0 -> (OC*FH*FW + 1)*IC
	//xoffset = W_k*FH0 - fh*FW0 - W_fw*IC0
	
	FW = FW * IC;//FW -> FW*IC
	FH = FH * FW;//FH -> FH*FW*IC
	IC = OC * FH + IC;//IC -> OC*FH*FW*IC + IC

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int W_fw = (W_k -= fh * tFW_OC) / OC;
	int woffset = W_k * FH - fh * FW - W_fw * IC;
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
		int fh = Y_k / tFW_OC;
		float4 x; int yoffset = fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fw = (W_k -= fh * tFW_OC) / OC;
		int woffset = W_k * FH - fh * FW - W_fw * IC;
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


//OC is power of 2
#ifndef XKERNEL10
#define XKERNEL10

#define xkernel10(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOC, ph, pw, GN, N) \
	Xkernel10<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,LOC, (FH - ph - 1), (FW - pw - 1), ic_index,n_index)

//(IH, IW) = 2:
//LB = 4: Size = 1.125, Time = 0.814 msec, Performace = 2967.96 GFlop/s
//LB = 3: Size = 1.125, Time = 0.928 msec, Performace = 2603.36 GFlop/s
//(IH, IW) = 4:
//LB = 4: Size = 1.125, Time = 1.124 msec, Performace = 2149.39 GFlop/s
//LB = 3: Size = 1.125, Time = 1.338 msec, Performace = 1805.62 GFlop/s
//(IH, IW) = 8:
//LB = 4: Size = 1.125, Time = 1.318 msec, Performace = 1833.02 GFlop/s
//LB = 3: Size = 1.125, Time = 1.494 msec, Performace = 1617.08 GFlop/s
//(IH, IW) = 16:
//LB = 4: Size = 1.125 , Time = 1.5 msec, Performace = 1610.61 GFlop/s
//LB = 3: Size = 0.5625, Time = 0.848 msec, Performace = 1424.48 GFlop/s
//(IH, IW) = 32:
//k88W3x4_oc2pow<4>: Size = 2.25, Time = 3.266 msec, Performace = 1479.44 GFlop/s
//LB = 4: Size = 2.25, Time = 3.168 msec, Performace = 1525.2 GFlop/s
//LB = 3: Size = 2.25, Time = 3.408 msec, Performace = 1417.79 GFlop/s
//(IH, IW) = 64:
//k88W3x4_oc2pow<4>: Size = 9, Time = 13.96 msec, Performace = 1384.48 GFlop/s
//LB = 4: Size = 9, Time = 13.8 msec, Performace = 1400.53 GFlop/s
//(IH, IW) = 128:
//k88W3x4_oc2pow<4>: Size = 36, Time = 52.58 msec, Performace = 1470.32 GFlop/s
//LB = 4: Size = 36, Time = 51.2 msec, Performace = 1509.95 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel10(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - oph, tiw = iw - opw;

	//prepare for GK = FH * FW * OC
	int fhs = -IF_int((tih < 0), tih, 0);
	int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;
	deltaY += (fhs*OW + fws) << LOC;//deltaY[0, fhs, fws, 0]
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws)) * IC;//W[0, -fhs, -fws, 0]

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0, Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	int Ystride = (OH * OW) << LOC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	FW = FW * IC;//FW -> FW*IC
	FH = FH * FW;//FH -> FH*FW*IC
	IC = (FH << LOC) + IC;//IC -> OC*FH*FW*IC + IC

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	const int SY = (OW - tFW) << LOC;
	int fh = Y_k / tFW_OC;
	float4 x; int yoffset = fh * SY + Y_k;
	x.x = deltaY[Y0 + yoffset];
	x.y = deltaY[Y1 + yoffset];
	x.z = deltaY[Y2 + yoffset];
	x.w = deltaY[Y3 + yoffset];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_fw = (W_k -= fh * tFW_OC) >> LOC;
	int woffset = W_k * FH - fh * FW - W_fw * IC;
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
		int fh = Y_k / tFW_OC;
		float4 x; int yoffset = fh * SY + Y_k;
		x.x = deltaY[Y0 + yoffset];
		x.y = deltaY[Y1 + yoffset];
		x.z = deltaY[Y2 + yoffset];
		x.w = deltaY[Y3 + yoffset];
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fw = (W_k -= fh * tFW_OC) >> LOC;
		int woffset = W_k * FH - fh * FW - W_fw * IC;
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