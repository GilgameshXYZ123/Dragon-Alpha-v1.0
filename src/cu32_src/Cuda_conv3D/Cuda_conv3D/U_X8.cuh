



//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef PXKERNEL1
#define PXKERNEL1

#define pxkernel1(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	PXkernel1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 3.3556 msec, Performace = 719.967 GFlop/s
template<int LB, int STEP>
__global__ void PXkernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FH_FW = FH * FW, GK = FH_FW * IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
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

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh, X_fw, X_ic; get_ic_fh_fw(X_k, X_ic, X_fh, X_fw);
	int xoffset = (X_fh*IW + X_fw)*IC + X_ic;
	Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);

	//load 4 elements from CW[FH, FW, IC, OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	int W_fh, W_fw, W_ic; get_ic_fh_fw(W_k, W_ic, W_fh, W_fw);
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
		int X_fh, X_fw, X_ic; get_ic_fh_fw(X_k, X_ic, X_fh, X_fw);
		int xoffset = (X_fh*IW + X_fw)*IC + X_ic;
		Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh, W_fw, W_ic; get_ic_fh_fw(W_k, W_ic, W_fh, W_fw);
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

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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
#ifndef PXKERNEL2
#define PXKERNEL2

#define pxkernel2(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	PXkernel2<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 3.32371 msec, Performace = 726.875 GFlop/s
template<int LB, int STEP>
__global__ void PXkernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FH_FW = FH * FW, GK = FH_FW * IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;
	const int sw_IC = sw * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh, X_fw, X_ic; get_ic_fh_fw(X_k, X_ic, X_fh, X_fw);
	int xoffset = (X_fh*IW + X_fw)*IC + X_ic;
	Xs[buf][tx][ty] = SaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
		toh0, tow0, tow1, tow2, tow3);

	//load 4 elements from CW[FH, FW, IC, OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	int W_fh, W_fw, W_ic; get_ic_fh_fw(W_k, W_ic, W_fh, W_fw);
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
		int X_fh, X_fw, X_ic; get_ic_fh_fw(X_k, X_ic, X_fh, X_fw);
		int xoffset = (X_fh*IW + X_fw)*IC + X_ic;
		Xs[buf][tx][ty] = SaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
			toh0, tow0, tow1, tow2, tow3);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh, W_fw, W_ic; get_ic_fh_fw(W_k, W_ic, W_fh, W_fw);
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

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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
#ifndef PXKERNEL3
#define PXKERNEL3

#define pxkernel3(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	PXkernel3<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 3.32371 msec, Performace = 726.875 GFlop/s
template<int LB, int STEP>
__global__ void PXkernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FH_FW = FH * FW, GK = FH_FW * IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;
	const int sw_IC = sw * IC;

	int count = 0;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh, X_fw, X_ic; get_ic_fh_fw(X_k, X_ic, X_fh, X_fw);
	int xoffset = (X_fh*IW + X_fw)*IC + X_ic;
	float4 xv = SaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
		toh0, tow0, tow1, tow2, tow3);
	Xs[buf][tx][ty] = xv; count++;

	if (tx == 0 && blockIdx.x == 0 && ty == 0 && blockIdx.y == 0) {
		printf("%d : %f, %f, %f, %f\n", count, xv.x, xv.y, xv.z, xv.w);
	}

	//load 4 elements from CW[FH, FW, IC, OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	int W_fh, W_fw, W_ic; get_ic_fh_fw(W_k, W_ic, W_fh, W_fw);
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
		int X_fh, X_fw, X_ic; get_ic_fh_fw(X_k, X_ic, X_fh, X_fw);
		int xoffset = (X_fh*IW + X_fw)*IC + X_ic;
		float4 xv = SaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
			toh0, tow0, tow1, tow2, tow3);
		Xs[buf][tx][ty] = xv; count++;

		if (tx == 0 && blockIdx.x == 0 && ty == 0 && blockIdx.y == 0) {
			printf("%d : %f, %f, %f, %f\n", count, xv.x, xv.y, xv.z, xv.w);
		}


		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh, W_fw, W_ic; get_ic_fh_fw(W_k, W_ic, W_fh, W_fw);
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

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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
#ifndef PXKERNEL4
#define PXKERNEL4

#define pxkernel4(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	PXkernel4<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 3.32371 msec, Performace = 726.875 GFlop/s
template<int LB, int STEP>
__global__ void PXkernel4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
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

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 0, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW * IC + X_k;
		Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
		__syncthreads();

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
		__syncthreads();
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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
#ifndef PXKERNEL5
#define PXKERNEL5

#define pxkernel5(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	PXkernel5<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 3.32371 msec, Performace = 726.875 GFlop/s
template<int LB, int STEP>
__global__ void PXkernel5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
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

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	const int OIC = (IC << 1 >> LB);
	for (int fh = 0; fh < FH; fh++) {
		for (int fw = 0; fw < FW; fw++) {
			for (int oic = 0; oic < OIC; oic++)
			{
				//load 4 elements from X[N, IH, IW, IC]
				int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
				int xoffset = (fh*IW + fw)*IC + xic;
				Xs[buf][tx][ty] = SaveX4(X, fh, fw, IH, IW, xoffset,
					X0, toh0, tow0,
					X1, toh1, tow1,
					X2, toh2, tow2,
					X3, toh3, tow3);

				//load 4 elements from CW[FH, FW, IC, OC]
				int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
				int woffset = ((fh*FW + fw)*IC + wic)*OC;
				Ws[buf][ty][tx] = *(float4*)(CW + woffset);
				__syncthreads();

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
				__syncthreads();
			}
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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
#ifndef PXKERNEL6
#define PXKERNEL6

#define pxkernel6(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	PXkernel6<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 3.32371 msec, Performace = 726.875 GFlop/s
template<int LB, int STEP>
__global__ void PXkernel6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
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

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	const int OIC = (IC << 1 >> LB);
	for (int oic = 0; oic < OIC; oic++) 
	{
		int count = 0;
		for (int fh = 0; fh < FH; fh++) {//[fh, fw, OC] sub element from CW
			for (int fw = 0; fw < FW; fw++)
			{
				//load 4 elements from X[N, IH, IW, IC]
				int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
				int xoffset = (fh*IW + fw)*IC + xic;
				float4 xv = SaveX4(X, fh, fw, IH, IW, xoffset,
					X0, toh0, tow0,
					X1, toh1, tow1,
					X2, toh2, tow2,
					X3, toh3, tow3);
				Xs[buf][tx][ty] = xv; count++;

				if (tx == 0 && ty == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
					printf("%d: %f, %f, %f, %f\n", count, xv.x, xv.y, xv.z, xv.w);
				}

				//load 4 elements from CW[FH, FW, IC, OC]
				int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
				int woffset = ((fh*FW + fw)*IC + wic)*OC;
				Ws[buf][ty][tx] = *(float4*)(CW + woffset);
				__syncthreads();

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
				__syncthreads();
			}
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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
#ifndef PXKERNEL7
#define PXKERNEL7

#define pxkernel7(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	PXkernel7<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 3.32371 msec, Performace = 726.875 GFlop/s
template<int LB, int STEP>
__global__ void PXkernel7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	toh1 = toh1 - ph, tow1 = tow1 - pw;
	toh2 = toh2 - ph, tow2 = tow2 - pw;
	toh3 = toh3 - ph, tow3 = tow3 - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	const int OIC = (IC << 1 >> LB);
	for (int oic = 0; oic < OIC; oic++) 
	{
		for (int fh = 0; fh < FH; fh++)//[fh, fw, STEP] sub from CW
		{
			//load 4 elements from X[N, IH, IW, IC]
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int xoffset = (fh*IW*IC) + xic;
			Xs[buf][tx][ty] = SaveX4(X, fh, 0, IH, IW, xoffset,
				X0, toh0, tow0,
				X1, toh1, tow1,
				X2, toh2, tow2,
				X3, toh3, tow3);

			//load 4 elements from CW[FH, FW, IC, OC]
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int woffset = ((fh*FW*IC) + wic)*OC;
			Ws[buf][ty][tx] = *(float4*)(CW + woffset);
			__syncthreads();

			for (int fw = 1; fw < FW; fw++)
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
				int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
				int xoffset = (fh*IW + fw)*IC + xic;
				Xs[buf][tx][ty] = SaveX4(X, fh, fw, IH, IW, xoffset,
					X0, toh0, tow0,
					X1, toh1, tow1,
					X2, toh2, tow2,
					X3, toh3, tow3);

				//load 4 elements from CW[FH, FW, IC, OC]
				int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
				int woffset = ((fh*FW + fw)*IC + wic)*OC;
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
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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


//(OH, OW) % 4 == 0
//sh = sw = 1;
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef PXKERNEL8
#define PXKERNEL8

#define pxkernel8(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	PXkernel8<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 3.32371 msec, Performace = 726.875 GFlop/s
template<int LB, int STEP>
__global__ void PXkernel8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
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
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	int tow1 = tow0 + 1, tow2 = tow0 + 2, tow3 = tow0 + 3;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	const int OIC = (IC << 1 >> LB);
	for (int oic = 0; oic < OIC; oic++)
	{
		for (int fh = 0; fh < FH; fh++)//[fh, fw, STEP] sub from CW
		{
			//load 4 elements from X[N, IH, IW, IC]
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int xoffset = (fh*IW*IC) + xic;
			Xs[buf][tx][ty] = SaveX4x(X, fh, 0, IH, IW, xoffset, IC,
				toh0, tow0, tow1, tow2, tow3);

			//load 4 elements from CW[FH, FW, IC, OC]
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int woffset = ((fh*FW*IC) + wic)*OC;
			Ws[buf][ty][tx] = *(float4*)(CW + woffset);
			__syncthreads();

			for (int fw = 1; fw < FW; fw++)
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
				int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
				int xoffset = (fh*IW + fw)*IC + xic;
				Xs[buf][tx][ty] = SaveX4x(X, fh, fw, IH, IW, xoffset, IC,
					toh0, tow0, tow1, tow2, tow3);

				//load 4 elements from CW[FH, FW, IC, OC]
				int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
				int woffset = ((fh*FW + fw)*IC + wic)*OC;
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
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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


//(OH, OW) % 4 == 0
//sh = sw = 1;
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef PXKERNEL9
#define PXKERNEL9

#define pxkernel9(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	PXkernel9<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 3.32371 msec, Performace = 726.875 GFlop/s
template<int LB, int STEP>
__global__ void PXkernel9(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
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
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	int tow1 = tow0 + 1, tow2 = tow0 + 2, tow3 = tow0 + 3;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	const int OIC = (IC << 1 >> LB);
	for (int oic = 0; oic < OIC; oic++)
	{
		for (int fh = 0; fh < FH; fh++)//[fh, fw, STEP] sub from CW
		{
			//load 4 elements from X[N, IH, IW, IC]
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int xoffset = (fh*IW*IC) + xic;
			float4 xv1 = SaveX4x(X, fh, 0, IH, IW, xoffset, IC,
				toh0, tow0, tow1, tow2, tow3);
			Xs[buf][tx][ty] = xv1; 

			if (tx == 0 && ty == 0 && blockIdx.x == 0 && blockIdx.y == 0)
					printf(" A: %f, %f, %f, %f\n",
						xv1.x, xv1.y, xv1.z, xv1.w);
					
			//load 4 elements from CW[FH, FW, IC, OC]
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int woffset = ((fh*FW*IC) + wic)*OC;
			Ws[buf][ty][tx] = *(float4*)(CW + woffset);
			__syncthreads();

			for (int fw = 1; fw < FW; fw++)
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
				int xoffset = (fh*IW + fw)*IC + xic;
				float4 xv1 = SaveX4x(X, fh, fw, IH, IW, xoffset, IC,
					toh0, tow0, tow1, tow2, tow3);
				Xs[buf][tx][ty] = xv1;

				if (tx == 0 && ty == 0 && blockIdx.x == 0 && blockIdx.y == 0)
					printf(" A: %f, %f, %f, %f\n",
						xv1.x, xv1.y, xv1.z, xv1.w);


				//load 4 elements from CW[FH, FW, IC, OC]
				int woffset = ((fh*FW + fw)*IC + wic)*OC;
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
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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


//(OH, OW) % 4 == 0
//sh = sw = 1;
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef PXKERNEL10
#define PXKERNEL10

#define pxkernel10(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	PXkernel10<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 3.32371 msec, Performace = 726.875 GFlop/s
template<int LB, int STEP>
__global__ void PXkernel10(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
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
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	int tow1 = tow0 + 1, tow2 = tow0 + 2, tow3 = tow0 + 3;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	const int OIC = (IC << 1 >> LB);
	for (int oic = 0; oic < OIC; oic++)
	{
		for (int fh = 0; fh < FH; fh++)//[fh, fw, STEP] sub from CW
		{
			//load 4 elements from X[N, IH, IW, IC]
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int xoffset = (fh*IW*IC) + xic;
			Xs[buf][tx][ty] = SaveX4x(X, fh, 0, IH, IW, xoffset, IC,
				toh0, tow0, tow1, tow2, tow3);

			//load 4 elements from CW[FH, FW, IC, OC]
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int woffset = ((fh*FW*IC) + wic)*OC;
			Ws[buf][ty][tx] = *(float4*)(CW + woffset);
			__syncthreads();

			for (int fw = 1; fw < FW; fw++)
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
				float4 xv0 = Xs[buf][tx][ty];
				buf ^= 1;

				//load 4 elements from X[N, IH, IW, IC]
				int xoffset = (fh*IW + fw)*IC + xic;
				bool lx3 = (toh0 >= -fh) && (toh0 < IH - fh) && (tow3 >= -fw) && (tow3 < IW - fw);
				float x3 = (lx3 ? X[xoffset + (IC << 1)] : 0);
				Xs[buf][tx][ty] = { xv0.y, xv0.z, xv0.w, x3 };

				//load 4 elements from CW[FH, FW, IC, OC]
				int woffset = ((fh*FW + fw)*IC + wic)*OC;
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
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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



//(OH, OW) % 4 == 0
//sh = sw = 1;
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef PXKERNEL11
#define PXKERNEL11

#define pxkernel11(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	PXkernel11<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.71014 msec, Performace = 1412.7 GFlop/s
template<int LB, int STEP>
__global__ void PXkernel11(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
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
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	int tow1 = tow0 + 1, tow2 = tow0 + 2, tow3 = tow0 + 3;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++) {
		for (int fh = 0; fh < FH; fh++)//[fh, fw, STEP] sub from CW
		{
			//load 4 elements from X[N, IH, IW, IC]
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
			bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
			bool lx1 = lx && (tow1 >= 0) && (tow1 < IW);
			bool lx2 = lx && (tow2 >= 0) && (tow2 < IW);
			bool lx3 = lx && (tow3 >= 0) && (tow3 < IW);
			float4 xv; int xoffset = (fh*IW*IC) + xic;
			xv.x = (lx0 ? X[xoffset - IC] : 0);
			xv.y = (lx1 ? X[xoffset] : 0);
			xv.z = (lx2 ? X[xoffset + IC] : 0);
			xv.w = (lx3 ? X[xoffset + (IC << 1)] : 0);
			Xs[buf][tx][ty] = xv;

			//load 4 elements from CW[FH, FW, IC, OC]
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int woffset = ((fh*FW*IC) + wic)*OC;
			Ws[buf][ty][tx] = *(float4*)(CW + woffset);
			__syncthreads();

			for (int fw = 1; fw < FW; fw++)
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
				bool lx3 = lx && (tow3 >= -fw) && (tow3 < IW - fw);
				int X3 = (fh*IW + fw + 2)*IC + xic;
				float x3 = (lx3 ? X[X3] : 0);
				float4 xv0 = Xs[buf ^ 1][tx][ty];
				Xs[buf][tx][ty] = { xv0.y, xv0.z, xv0.w, x3 };

				//load 4 elements from CW[FH, FW, IC, OC]
				int woffset = ((fh*FW + fw)*IC + wic)*OC;
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
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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



//(OH, OW) % 4 == 0
//sh = sw = 1;
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef PXKERNEL12
#define PXKERNEL12

#define pxkernel12(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	PXkernel12<LB, (1<<LB>>1), FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.57016 msec, Performace = 1538.64 GFlop/s
template<int LB, int STEP, int FH, int FW>
__global__ void PXkernel12(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, 
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
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
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	int tow1 = tow0 + 1, tow2 = tow0 + 2, tow3 = tow0 + 3;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++) {
		for (int fh = 0; fh < FH; fh++)//[fh, fw, STEP] sub from CW
		{
			//load 4 elements from CW[FH, FW, IC, OC]
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int woffset = ((fh*FW*IC) + wic)*OC;
			Ws[buf][ty][tx] = *(float4*)(CW + woffset);

			//load 4 elements from X[N, IH, IW, IC]
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
			bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
			bool lx1 = lx && (tow1 >= 0) && (tow1 < IW);
			bool lx2 = lx && (tow2 >= 0) && (tow2 < IW);
			bool lx3 = lx && (tow3 >= 0) && (tow3 < IW);
			float4 xv; int xoffset = (fh*IW*IC) + xic;
			xv.x = (lx0 ? X[xoffset - IC] : 0);
			xv.y = (lx1 ? X[xoffset] : 0);
			xv.z = (lx2 ? X[xoffset + IC] : 0);
			xv.w = (lx3 ? X[xoffset + (IC << 1)] : 0);
			Xs[buf][tx][ty] = xv;
			__syncthreads();

#pragma unroll
			for (int fw = 1; fw < FW; fw++) {
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
				bool lx3 = lx && (tow3 >= -fw) && (tow3 < IW - fw);
				int xoffset3 = (fh*IW + fw + 2)*IC + xic;
				float x3 = (lx3 ? X[xoffset3] : 0);
				float4 xv0 = Xs[buf ^ 1][tx][ty];
				Xs[buf][tx][ty] = { xv0.y, xv0.z, xv0.w,  x3};

				//load 4 elements from CW[FH, FW, IC, OC]
				int woffset = ((fh*FW + fw)*IC + wic)*OC;
				Ws[buf][ty][tx] = *(float4*)(CW + woffset);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

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
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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



//(OH, OW) % 4 == 0
//sh = sw = 1;
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef PXKERNEL13
#define PXKERNEL13

#define pxkernel13(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	PXkernel13<LB, (1<<LB>>1), FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.57016 msec, Performace = 1538.64 GFlop/s
template<int LB, int STEP, int FH, int FW>
__global__ void PXkernel13(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
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
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	int tow1 = tow0 + 1, tow2 = tow0 + 2, tow3 = tow0 + 3;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++) {
		for (int fh = 0; fh < FH; fh++)//[fh, fw, STEP] sub from CW
		{
			//load 4 elements from X[N, IH, IW, IC]
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
			bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
			bool lx1 = lx && (tow1 >= 0) && (tow1 < IW);
			bool lx2 = lx && (tow2 >= 0) && (tow2 < IW);
			bool lx3 = lx && (tow3 >= 0) && (tow3 < IW);
			float4 xv; int xoffset = (fh*IW*IC) + xic;
			xv.x = (lx0 ? X[xoffset - IC] : 0);
			xv.y = (lx1 ? X[xoffset] : 0);
			xv.z = (lx2 ? X[xoffset + IC] : 0);
			xv.w = (lx3 ? X[xoffset + (IC << 1)] : 0);
			Xs[buf][tx][ty] = xv;

			//load 4 elements from CW[FH, FW, IC, OC]
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int woffset = ((wic*FH + fh)*FW)*OC;
			Ws[buf][ty][tx] = *(float4*)(CW + woffset);
			__syncthreads();

#pragma unroll
			for (int fw = 1; fw < FW; fw++) {
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
				bool lx3 = lx && (tow3 >= -fw) && (tow3 < IW - fw);
				int xoffset3 = (fh*IW + fw + 2)*IC + xic;
				float x3 = (lx3 ? X[xoffset3] : 0);
				float4 xv0 = Xs[buf ^ 1][tx][ty];
				Xs[buf][tx][ty] = { xv0.y, xv0.z, xv0.w,  x3 };

				//load 4 elements from CW[FH, FW, IC, OC]
				int woffset = ((wic*FH + fh)*FW + fw)*OC;
				Ws[buf][ty][tx] = *(float4*)(CW + woffset);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

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
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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



//(OH, OW) % 4 == 0
//sh = sw = 1;
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 16 == 0 
#ifndef PXKERNEL15
#define PXKERNEL15

#define pxkernel15(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	PXkernel15<LB, (1<<LB>>1), (1<<LB), FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.52079 msec, Performace = 1588.6 GFlop/s(1000)
template<int LB, int STEP, int STEP2, int FH, int FW>
__global__ void PXkernel15(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	int tow1 = tow0 + 1, tow2 = tow0 + 2, tow3 = tow0 + 3;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int oic = 0, OIC = (IC >> LB); oic < OIC; oic++) {
		for (int fh = 0; fh < FH; fh++)//[fh, fw, STEP2] sub from CW
		{
			//load 4 elements from X[N, IH, IW, IC]
			int xic = (((oic - (tx >= STEP)) << LB >> 1) + tx) << 1;
			bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
			bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
			bool lx1 = lx && (tow1 >= 0) && (tow1 < IW);
			bool lx2 = lx && (tow2 >= 0) && (tow2 < IW);
			bool lx3 = lx && (tow3 >= 0) && (tow3 < IW);
			int xoffset0 = (fh*IW*IC) + xic;
			float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
			float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
			float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
			float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
			Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
			Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

			//load 4 elements from CW[FH, FW, IC, OC]
			int wic = (((oic - (ty >= STEP)) << LB >> 1) + ty) << 1;
			int woffset0 = ((fh*FW*IC) + wic)*OC;
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);
			__syncthreads();

#pragma unroll
			for (int fw = 1; fw < FW; fw++) {
//#pragma unroll
				for (int ik = 0; ik < STEP2; ik++)
				{
					float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
					float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

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
				bool lx3 = lx && (tow3 >= -fw) && (tow3 < IW - fw);
				int xoffset3 = xoffset0 + (fw + 2)*IC;
				float2 x3 = (lx3 ? *(float2*)(X + xoffset3) : F32_2_0);
				float4 xv0 = Xs[buf^1][(tx << 1)][ty];
				float4 xv1 = Xs[buf^1][(tx << 1) + 1][ty];
				Xs[buf][(tx << 1)    ][ty] = { xv0.y, xv0.z, xv0.w, x3.x };
				Xs[buf][(tx << 1) + 1][ty] = { xv1.y, xv1.z, xv1.w, x3.y };

				//load 4 elements from CW[FH, FW, IC, OC]
				int woffset = woffset0 + (fw*IC*OC);
				Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
				Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP2; ik++)
			{
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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



//(OH, OW) % 4 == 0
//sh = sw = 1;
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef PXKERNEL16
#define PXKERNEL16

#define pxkernel16(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	PXkernel16<LB, (1<<LB>>1), FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.52079 msec, Performace = 1588.6 GFlop/s(1000)
template<int LB, int STEP, int FH, int FW>
__global__ void PXkernel16(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
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
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	int tow1 = tow0 + 1, tow2 = tow0 + 2, tow3 = tow0 + 3;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++) {
		for (int fh = 0; fh < FH; fh++)//[fh, fw, STEP] sub from CW
		{
			//load 4 elements from CW[FH, FW, IC, OC]
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int woffset0 = ((fh*FW*IC) + wic)*OC;
			Ws[buf][ty][tx] = *(float4*)(CW + woffset0);

			//load 4 elements from X[N, IH, IW, IC]
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
			bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
			bool lx1 = lx && (tow1 >= 0) && (tow1 < IW);
			bool lx2 = lx && (tow2 >= 0) && (tow2 < IW);
			bool lx3 = lx && (tow3 >= 0) && (tow3 < IW);
			float4 xv; int xoffset0 = (fh*IW*IC) + xic;
			xv.x = (lx0 ? X[xoffset0 - IC] : 0);
			xv.y = (lx1 ? X[xoffset0] : 0);
			xv.z = (lx2 ? X[xoffset0 + IC] : 0);
			xv.w = (lx3 ? X[xoffset0 + (IC << 1)] : 0);
			Xs[buf][tx][ty] = xv;
			__syncthreads();

#pragma unroll
			for (int fw = 1; fw < FW; fw++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
					float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

					simdMM4(v0, b0.x, a0);  simdMM4(v1, b0.x, a1);
					simdMM4(v2, b0.y, a0);  simdMM4(v3, b0.y, a1);
					simdMM4(v4, b0.z, a0);  simdMM4(v5, b0.z, a1);
					simdMM4(v6, b0.w, a0);  simdMM4(v7, b0.w, a1);
					simdMM4(v8, b1.x, a0);  simdMM4(v9, b1.x, a1);
					simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
					simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
					simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
				}
				float4 xv0 = Xs[buf][tx][ty];
				buf ^= 1;

				//load 4 elements from X[N, IH, IW, IC]
				bool lx3 = lx && (tow3 >= -fw) && (tow3 < IW - fw);
				int xoffset3 = xoffset0 + (fw + 2)*IC;
				float x3 = (lx3 ? X[xoffset3] : 0);
				Xs[buf][tx][ty] = { xv0.y, xv0.z, xv0.w,  x3 };

				//load 4 elements from CW[FH, FW, IC, OC]
				int woffset = woffset0 + (fw*IC*OC);
				Ws[buf][ty][tx] = *(float4*)(CW + woffset);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

				simdMM4(v0, b0.x, a0);  simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0);  simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0);  simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0);  simdMM4(v7, b0.w, a1);
				simdMM4(v8, b1.x, a0);  simdMM4(v9, b1.x, a1);
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

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