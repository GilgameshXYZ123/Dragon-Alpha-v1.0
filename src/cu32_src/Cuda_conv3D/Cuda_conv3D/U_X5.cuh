

//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL1
#define XKERNEL1

#define xkernel1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	Xkernel1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//for(IH, IW) =  4, LB = 4: Size = 1.125, Time = 2.14014 msec, Performace = 1128.86 GFlop/s
//                          Size = 2.25 , Time = 3.4686  msec, Performace = 1393.02 GFlop/s
//-> Size = 2.25, Time = 3.26523 msec, Performace = 1479.78 GFlop/s
//for(IH, IW) =  8, LB = 4: Size = 1.125, Time = 1.70936 msec, Performace = 1413.35 GFlop/s
//for(IH, IW) = 16, LB = 4: Size = 1.125, Time = 1.70837 msec, Performace = 1414.16 GFlop/s
//LB = 4: Size = 0.5625, Time = 0.951464 msec, Performace = 1269.58 GFlop/s
//LB = 3: Size = 0.5625, Time = 1.0617   msec, Performace = 1137.76 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
	Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
		Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif



//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL2
#define XKERNEL2

#define xkernel2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	Xkernel2<LB, (1<<LB>>1)>\
		<<< dim3(OC>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.67909 msec, Performace = 1438.82 GFlop/s

//bz -> (oh, ow)
//(bx, tx) -> n
//(by, ty) -> oc, As: GN = OC
template<int LB, int STEP>
__global__ void Xkernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz % OW;
	int toh = oh * sh - ph, tow = ow * sw - pw;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for M = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//prepare for GK = FH * FW * IC
	const int GK = 9 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
	bool lx = (toh >= -X_fh) && (toh < IH - X_fh) &&
		      (tow >= -X_fw) && (tow < IW - X_fw);
	float4 x;
	x.x = (lx ? X[X0 + xoffset] : 0);
	x.y = (lx ? X[X1 + xoffset] : 0);
	x.z = (lx ? X[X2 + xoffset] : 0);
	x.w = (lx ? X[X3 + xoffset] : 0);
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
		bool lx = (toh >= -X_fh) && (toh < IH - X_fh) && 
			      (tow >= -X_fw) && (tow < IW - X_fw);
		float4 x;
		x.x = (lx ? X[X0 + xoffset] : 0);
		x.y = (lx ? X[X1 + xoffset] : 0);
		x.z = (lx ? X[X2 + xoffset] : 0);
		x.w = (lx ? X[X3 + xoffset] : 0);
		Xs[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;
	
	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL3
#define XKERNEL3

#define xkernel3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	Xkernel3<LB, (1<<LB>>1)>\
		<<< dim3(OC>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.69424 msec, Performace = 1425.96 GFlop/s
//bz -> (oh, ow)
//(bx, tx) -> n
//(by, ty) -> oc, As: GN = OC
template<int LB, int STEP>
__global__ void Xkernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - ph, tow = ow * sw - pw;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for M = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//prepare for GK = FH * FW * IC
	const int GK = 9 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
	bool lx = (toh >= -X_fh) && (toh < IH - X_fh) && (tow >= -X_fw) && (tow < IW - X_fw);
	float4 x;
	x.x = (lx ? X[X0 + xoffset] : 0);
	x.y = (lx ? X[X1 + xoffset] : 0);
	x.z = (lx ? X[X2 + xoffset] : 0);
	x.w = (lx ? X[X3 + xoffset] : 0);
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
		bool lx = (toh >= -X_fh) && (toh < IH - X_fh) && (tow >= -X_fw) && (tow < IW - X_fw);
		float4 x;
		x.x = (lx ? X[X0 + xoffset] : 0);
		x.y = (lx ? X[X1 + xoffset] : 0);
		x.z = (lx ? X[X2 + xoffset] : 0);
		x.w = (lx ? X[X3 + xoffset] : 0);
		Xs[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL4
#define XKERNEL4

#define xkernel4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	Xkernel4<LB, (1<<LB>>1)>\
		<<< dim3(OC>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.69424 msec, Performace = 1425.96 GFlop/s
//bz -> (oh, ow)
//(bx, tx) -> n
//(by, ty) -> oc, As: GN = OC
template<int LB, int STEP>
__global__ void Xkernel4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - ph, tow = ow * sw - pw;
	
	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for M = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//prepare for GK = FH * FW * IC
	//if toh < 0: FH = FH + toh
	const int tFH = 3 + IF_int((toh < 0), toh, 0);
	const int tFW = 3 + IF_int((tow < 0), tow, 0);
	const int GK = IC * tFH * tFW;
	int fhs = -IF_int((toh < 0), toh, 0);
	int fws = -IF_int((tow < 0), tow, 0);
	int ws = (fhs*3 + fws)*IC*OC; CW += ws;
	int xs = (fhs*IW + fws)*IC; X += xs;
	toh = IF_int((toh < 0), 0, toh);
	tow = IF_int((tow < 0), 0, tow);

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh = X_k / (tFW * IC); X_k -= X_fh * (tFW * IC);
	int X_fw = X_k / IC; 
	int xoffset = X_fh * IW * IC + X_k;
	float4 x; bool lx = (toh < IH - X_fh) && (tow < IW - X_fw);
	x.x = (lx ? X[X0 + xoffset] : 0);
	x.y = (lx ? X[X1 + xoffset] : 0);
	x.z = (lx ? X[X2 + xoffset] : 0);
	x.w = (lx ? X[X3 + xoffset] : 0);
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh = X_k / (tFW * IC); X_k -= X_fh * (tFW * IC);
		int X_fw = X_k / IC; 
		int xoffset = X_fh * IW * IC + X_k;
		float4 x; bool lx = (toh < IH - X_fh) && (tow < IW - X_fw);
		x.x = (lx ? X[X0 + xoffset] : 0);
		x.y = (lx ? X[X1 + xoffset] : 0);
		x.z = (lx ? X[X2 + xoffset] : 0);
		x.w = (lx ? X[X3 + xoffset] : 0);
		Xs[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL5
#define XKERNEL5

#define xkernel5(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	Xkernel5<LB, (1<<LB>>1)>\
		<<< dim3(OC>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.69424 msec, Performace = 1425.96 GFlop/s
//LB = 4: Size = 1.125, Time = 1.36087 msec, Performace = 1775.28 GFlop/s
//bz -> (oh, ow)
//(bx, tx) -> n
//(by, ty) -> oc, As: GN = OC
template<int LB, int STEP>
__global__ void Xkernel5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - ph, tow = ow * sw - pw;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for M = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//prepare for GK = FH * FW * IC
	const int fhs = -IF_int((toh < 0), toh, 0), tFH = FH - fhs;
	const int fws = -IF_int((tow < 0), tow, 0), tFW = FW - fws;
	const int tFW_IC = tFW * IC, GK = tFH * tFW_IC;
	CW += (fhs * FW + fws)*IC*OC;//CW[fhs, fws, 0, oc0]

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh = X_k / tFW_IC; X_k -= X_fh * tFW_IC;
	int X_fw = X_k / IC; 
	int IW_IC = IW * IC;
	int xoffset = (X_fh + fhs)* IW_IC + X_k + fws * IC;
	
	float4 x; 
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	int W_fh = W_k / tFW_IC; W_k -= W_fh * tFW_IC;
	int W_fw = W_k / IC, W_ic = W_k % IC;
	int woffset = ((W_fh*FW + W_fw)*IC + W_ic)*OC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh = X_k / tFW_IC; X_k -= X_fh * tFW_IC;
		int X_fw = X_k / IC;
		int IW_IC = IW * IC;
		int xoffset = (X_fh + fhs)* IW_IC + X_k + fws * IC;

		float4 x;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh = W_k / tFW_IC; W_k -= W_fh * tFW_IC;
		int W_fw = W_k / IC, W_ic = W_k % IC;
		int woffset = ((W_fh*FW + W_fw)*IC + W_ic)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL6
#define XKERNEL6

#define xkernel6(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	Xkernel6<LB, (1<<LB>>1)>\
		<<< dim3(OC>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//k88W3_ic2pow: Size = 2.25, Time = 3.347 msec, Performace = 1443.63 GFlop/s
//LB = 4: Size = 2.25, Time = 2.63742 msec, Performace = 1832.03 GFlop/s
//LB = 4: Size = 1.125, Time = 1.69424 msec, Performace = 1425.96 GFlop/s
//LB = 4: Size = 1.125, Time = 1.36087 msec, Performace = 1775.28 GFlop/s
//bz -> (oh, ow)
//(bx, tx) -> n
//(by, ty) -> oc, As: GN = OC
template<int LB, int STEP>
__global__ void Xkernel6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - ph, tow = ow * sw - pw;

	//prepare for GK = FH * FW * IC
	const int fhs = -IF_int((toh < 0), toh, 0), tFH = FH - fhs;
	const int fws = -IF_int((tow < 0), tow, 0), tFW = FW - fws;
	const int tFW_IC = tFW * IC, GK = tFH * tFW_IC;
	CW += (fhs * FW + fws)*IC*OC;//CW[fhs, fws, 0, oc0]

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]
	X += (fhs*IW + fws)*IC;//X[0, fhs, fws, 0]

	//prepare for M = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh = X_k / tFW_IC; X_k -= X_fh * tFW_IC;
	int X_fw = X_k / IC; int IW_IC = IW * IC;
	float4 x; int xoffset = X_fh * IW_IC + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	int W_fh = W_k / tFW_IC; W_k -= W_fh * tFW_IC; int FW_IC = FW * IC;
	int woffset = (W_fh*FW_IC + W_k)*OC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh = X_k / tFW_IC; X_k -= X_fh * tFW_IC;
		int X_fw = X_k / IC; 
		float4 x; int xoffset = X_fh * IW_IC + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh = W_k / tFW_IC; W_k -= W_fh * tFW_IC;
		int woffset = (W_fh*FW_IC + W_k)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL7
#define XKERNEL7

#define xkernel7(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	Xkernel7<LB, (1<<LB>>1)>\
		<<< dim3(OC>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//k88W3_ic2pow: Size = 2.25, Time = 3.347 msec, Performace = 1443.63 GFlop/s
//LB = 4: Size = 2.25, Time = 2.63742 msec, Performace = 1832.03 GFlop/s
//LB = 4: Size = 1.125, Time = 1.69424 msec, Performace = 1425.96 GFlop/s
//LB = 4: Size = 1.125, Time = 1.36087 msec, Performace = 1775.28 GFlop/s
//bz -> (oh, ow)
//(bx, tx) -> n
//(by, ty) -> oc, As: GN = OC
template<int LB, int STEP>
__global__ void Xkernel7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - ph, tow = ow * sw - pw;

	//prepare for GK = FH * FW * IC
	const int fhs = -IF_int((toh < 0), toh, 0); 
	const int fws = -IF_int((tow < 0), tow, 0); 
	int fhe = IH - toh; fhe = IF_int((fhe > FH), FH, fhe);//tow0 + fw < IW -> fw < IW - tow0
	int fwe = IW - tow; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;

	const int tFW_IC = tFW * IC, GK = tFH * tFW_IC;
	CW += (fhs * FW + fws)*IC*OC;//CW[fhs, fws, 0, oc0]

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]
	X += (fhs*IW + fws)*IC;//X[0, fhs, fws, 0]

	//prepare for M = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh = X_k / tFW_IC; X_k -= X_fh * tFW_IC;
	int X_fw = X_k / IC; int IW_IC = IW * IC;
	float4 x; int xoffset = X_fh * IW_IC + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	int W_fh = W_k / tFW_IC; W_k -= W_fh * tFW_IC; int FW_IC = FW * IC;
	int woffset = (W_fh*FW_IC + W_k)*OC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh = X_k / tFW_IC; X_k -= X_fh * tFW_IC;
		int X_fw = X_k / IC;
		float4 x; int xoffset = X_fh * IW_IC + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh = W_k / tFW_IC; W_k -= W_fh * tFW_IC;
		int woffset = (W_fh*FW_IC + W_k)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif



//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL8
#define XKERNEL8

#define xkernel8(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	Xkernel8<LB, (1<<LB>>1)>\
		<<< dim3(OC>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//k88W3_ic2pow: Size = 2.25, Time = 3.347 msec, Performace = 1443.63 GFlop/s
//LB = 4: Size = 2.25, Time = 2.63742 msec, Performace = 1832.03 GFlop/s
//LB = 4: Size = 1.125, Time = 1.69424 msec, Performace = 1425.96 GFlop/s
//LB = 4: Size = 1.125, Time = 1.36087 msec, Performace = 1775.28 GFlop/s
//bz -> (oh, ow)
//(bx, tx) -> n
//(by, ty) -> oc, As: GN = OC
template<int LB, int STEP>
__global__ void Xkernel8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - ph, tow = ow * sw - pw;
	
	//prepare for GK = FH * FW * IC
	int fhs = -IF_int((toh < 0), toh, 0);
	int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_IC = tFW * IC, GK = tFH * tFW_IC;
	CW += (fhs * FW + fws)*IC*OC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws)*IC;//X[0, fhs, fws, 0]

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for M = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int Xstride = IH * IW * IC; 
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	int W_fh = W_k / tFW_IC; W_k -= W_fh * tFW_IC; int FW_IC = FW * IC;
	int woffset = (W_fh*FW_IC + W_k)*OC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh = X_k / tFW_IC; X_k -= X_fh * tFW_IC;
	int X_fw = X_k / IC; int IW_IC = IW * IC;
	float4 x; int xoffset = X_fh * IW_IC + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh = W_k / tFW_IC; W_k -= W_fh * tFW_IC;
		int woffset = (W_fh*FW_IC + W_k)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
		__syncthreads();

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh = X_k / tFW_IC; X_k -= X_fh * tFW_IC;
		int X_fw = X_k / IC;
		float4 x; int xoffset = X_fh * IW_IC + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL9
#define XKERNEL9

#define xkernel9(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	Xkernel9<LB, (1<<LB>>1)>\
		<<< dim3(OC>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//k88W3_ic2pow: Size = 2.25, Time = 3.347 msec, Performace = 1443.63 GFlop/s
//LB = 4: Size = 2.25, Time = 2.63678 msec, Performace = 1832.48 GFlop/s
//LB = 4: Size = 1.125, Time = 1.69424 msec, Performace = 1425.96 GFlop/s
//LB = 4: Size = 1.125, Time = 1.36087 msec, Performace = 1775.28 GFlop/s
//bz -> (oh, ow)
//(bx, tx) -> n
//(by, ty) -> oc, As: GN = OC
template<int LB, int STEP>
__global__ void Xkernel9(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - ph, tow = ow * sw - pw;

	//prepare for GK = FH * FW * IC
	int fhs = -IF_int((toh < 0), toh, 0);
	int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_IC = tFW * IC, GK = tFH * tFW_IC;
	CW += (fhs * FW + fws)*IC*OC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws)*IC;//X[0, fhs, fws, 0]

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for M = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//xoffset = X_fh * IW_IC + X_k - X_fh * tFW_IC;
	//= X_fh * (IW_IC - tFW_IC) + X_k
	
	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh = X_k / tFW_IC; const int SX = (IW - tFW)*IC;
	float4 x; int xoffset = X_fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	int W_fh = W_k / tFW_IC; const int SW = (FW - tFW)*IC;
	int woffset = (W_fh*SW + W_k)*OC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh = X_k / tFW_IC;
		float4 x; int xoffset = X_fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh = W_k / tFW_IC;
		int woffset = (W_fh*SW + W_k)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC % 8 == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL10
#define XKERNEL10

#define xkernel10(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	Xkernel10<LB, (1<<LB>>1)>\
		<<< dim3(OC>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//k88W3_ic2pow: Size = 2.25, Time = 3.347 msec, Performace = 1443.63 GFlop/s
//LB = 4: Size = 2.25, Time = 2.55994 msec, Performace = 1887.48 GFlop/ss
//LB = 4: Size = 1.125, Time = 1.69424 msec, Performace = 1425.96 GFlop/s
//LB = 4: Size = 1.125, Time = 1.36087 msec, Performace = 1775.28 GFlop/s
//bz -> (oh, ow)
//(bx, tx) -> n
//(by, ty) -> oc, As: GN = OC
template<int LB, int STEP>
__global__ void Xkernel10(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - ph, tow = ow * sw - pw;

	//prepare for GK = FH * FW * IC
	int fhs = -IF_int((toh < 0), toh, 0);
	int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_IC = tFW * IC, GK = tFH * tFW_IC;
	CW += (fhs * FW + fws)*IC*OC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws)*IC;//X[0, fhs, fws, 0]

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for M = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow)*IC;
	int Xstride = IH * IW * IC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
	//W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
	//X_k = ok*STEP + tx - ((tx >= STEP) << LB >> 1)
	//W_k = ok*STEP + ty - ((ty >= STEP) << LB >> 1)
	//Let: Ux = tx - ((tx >= STEP) << LB >> 1)
	//Let: Uy = ty - ((ty >= STEP) << LB >> 1)
	//X_k = ok*STEP + Ux
	//W_k = ok*STEP + Uy
	//As: STEP % 8 == 0,
	//we have:
	//X_k = ok*8*x + Ux
	//W_k = ok*8*x + Uy, 
	//when IC % 8 == 0, we have: (tFH*IC) % 8 == 0
	//X_fh = (ok*8*x + Ux) / tFW_IC = (ok*8*x + Ux)/8y
	//W_fh = (ok*8*x + Uy) / tFW_IC = (ok*8*x + Ux)/8y
	//As: LB = 4 or 3, Ux, Uy belongs to [0, 7]
	//So: X_fh = W_fh, when IC % 8 == 0, LB = 4
	//when LB = 3, Ux, Uy belongs to [0, 3], STEP = 4
	//X_fh = (ok*4*x + Ux) / tFW_IC = (ok*4*x + Ux)/4y
	//W_fh = (ok*4*x + Uy) / tFW_IC = (ok*4*x + Ux)/4y

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh = X_k / tFW_IC; const int SX = (IW - tFW)*IC;
	float4 x; int xoffset = X_fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	const int SW = (FW - tFW)*IC;
	int woffset = (X_fh*SW + W_k)*OC;
	Ws[buf][ty][tx] = *(float4*)(CW + woffset);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh = X_k / tFW_IC;
		float4 x; int xoffset = X_fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = (X_fh*SW + W_k)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif
