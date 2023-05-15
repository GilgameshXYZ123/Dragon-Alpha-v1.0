



//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL1
#define XKERNEL1

#define Xkernel1(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	pxkernel1<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1, Time = 1.52632 msec, Performace = 1406.97 GFlop/s
//LB = 3: Size = 1, Time = 1.87689 msec, Performace = 1144.17 GFlop/s
//LB = 4: Size = 1.125, Time = 1.73012 msec, Performace = 1396.39 GFlop/s
//LB = 3: Size = 1.125, Time = 2.05591 msec, Performace = 1175.11 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void pxkernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

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

	//when IC % STEP2 == 0
	//load 4 elements from X[N, IH, IW, IC]
	const int IW_IC = IW * IC;

	int X_k1 = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int X_fh1, X_fw1; get_X_fh_fw(X_k1, X_fh1, X_fw1);
	int xoffset1 = X_fh1 * IW_IC + X_k1;
	Xs[buf][(tx << 1)][ty] = SaveX4(X, X_fh1, X_fw1, IH, IW, xoffset1,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);

	int X_k2 = ((tx - ((tx >= STEP) << LB >> 1)) << 1) + 1;
	int X_fh2, X_fw2; get_X_fh_fw(X_k2, X_fh2, X_fw2);
	int xoffset2 = X_fh2 * IW_IC + X_k2;
	Xs[buf][(tx << 1) + 1][ty] = SaveX4(X, X_fh2, X_fw2, IH, IW, xoffset2,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k1 = (ty - ((ty >= STEP) << LB >> 1) << 1);
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + W_k1 * OC);

	int W_k2 = (ty - ((ty >= STEP) << LB >> 1) << 1) + 1;
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + W_k2 * OC);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
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
		int X_k1 = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int X_fh1, X_fw1; get_X_fh_fw(X_k1, X_fh1, X_fw1);
		int xoffset1 = X_fh1 * IW_IC + X_k1;
		Xs[buf][(tx << 1)][ty] = SaveX4(X, X_fh1, X_fw1, IH, IW, xoffset1,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		int X_k2 = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1) + 1;
		int X_fh2, X_fw2; get_X_fh_fw(X_k2, X_fh2, X_fw2);
		int xoffset2 = X_fh2 * IW_IC + X_k2;
		Xs[buf][(tx << 1) + 1][ty] = SaveX4(X, X_fh2, X_fw2, IH, IW, xoffset2,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k1 = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + W_k1 * OC);

		int W_k2 = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1) + 1;
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + W_k2 * OC);
		__syncthreads();
	}
#pragma unroll
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
#ifndef XKERNEL2
#define XKERNEL2

#define Xkernel2(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	pxkernel2<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1, Time = 1.52632 msec, Performace = 1406.97 GFlop/s
//LB = 3: Size = 1, Time = 1.87689 msec, Performace = 1144.17 GFlop/s
//LB = 4: Size = 1.125, Time = 1.73012 msec, Performace = 1396.39 GFlop/s
//LB = 3: Size = 1.125, Time = 2.05591 msec, Performace = 1175.11 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void pxkernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

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

	//k0 = STEP2*ok + Ux*2 
	//k1 = STEP2*ok + Ux*2 + 1
	//Ux belongs to [0, STEP)
	//(1) ici = ki % IC, as IC is power of 4
	//As: STEP2 is power of 4
	//[1] ic0 = (STEP2*ok + Ux*2    ) % IC = (4*x + Ux*2    ) % (4*y)
	//[2] ic1 = (STEP2*ok + Ux*2 + 1) % IC = (4*x + Ux*2 + 1) % (4*y)
	//so: ic0 = ic1 - 1
	//(2) fh = ki / (FW_IC), 
	//when: IC % STEP2 == 0
	//[1] fh0 = (STEP2*ok + Ux*2    ) / (FW*IC£©= (STEP2*ok + Ux*2    ) / (STEP2*v) = ok / v
	//[2] fh1 = (STEP2*ok + Ux*2 + 1) / (FW*IC£©= (STEP2*ok + Ux*2 + 1) / (STEP2*v) = ok / v
	//So: fh0 = fh1
	//(3) fw = (ki % FW_IC) / IC
	//when: IC % STEP2 == 0
	//we have: fw0 = fh0

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0, x1, x2, x3;
	x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : make_float2(0, 0));
	x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : make_float2(0, 0));
	x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : make_float2(0, 0));
	x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : make_float2(0, 0));
	Xs[buf][(tx << 1)    ][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k1 = (ty - ((ty >= STEP) << LB >> 1) << 1);
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + W_k1 * OC);

	int W_k2 = (ty - ((ty >= STEP) << LB >> 1) << 1) + 1;
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + W_k2 * OC);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k1 = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + W_k1 * OC);

		int W_k2 = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1) + 1;
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + W_k2 * OC);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0, x1, x2, x3;
		x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : make_float2(0, 0));
		x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : make_float2(0, 0));
		x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : make_float2(0, 0));
		x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : make_float2(0, 0));
		Xs[buf][(tx << 1)    ][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

		
		__syncthreads();
	}
#pragma unroll
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % BLOCK_SIZE == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL3
#define XKERNEL3

#define Xkernel3(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	pxkernel3<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1, Time = 1.52632 msec, Performace = 1406.97 GFlop/s
//LB = 3: Size = 1, Time = 1.87689 msec, Performace = 1144.17 GFlop/s
//LB = 4: Size = 1.125, Time = 1.73012 msec, Performace = 1396.39 GFlop/s
//LB = 3: Size = 1.125, Time = 2.05591 msec, Performace = 1175.11 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void pxkernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

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

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0, x1, x2, x3;
	x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : make_float2(0, 0));
	x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : make_float2(0, 0));
	x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : make_float2(0, 0));
	x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : make_float2(0, 0));
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0, x1, x2, x3;
		x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : make_float2(0, 0));
		x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : make_float2(0, 0));
		x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : make_float2(0, 0));
		x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : make_float2(0, 0));
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
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


//IC is power of 2
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % BLOCK_SIZE == 0
//LB = 4, GK % 8 == 0 
#ifndef XKERNEL4
#define XKERNEL4

#define Xkernel4(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	pxkernel4<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.61774 msec, Performace = 1493.39 GFlop/s
//LB = 3: Size = 1.125, Time = 1.76776 msec, Performace = 1366.66 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void pxkernel4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

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
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3) << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	int xoffset = (X_fh << LIC)*IW + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0, x1, x2, x3;
	x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : make_float2(0, 0));
	x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : make_float2(0, 0));
	x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : make_float2(0, 0));
	x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : make_float2(0, 0));
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
		int xoffset = (X_fh << LIC)*IW + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0, x1, x2, x3;
		x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : make_float2(0, 0));
		x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : make_float2(0, 0));
		x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : make_float2(0, 0));
		x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : make_float2(0, 0));
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
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


//FH = FW = 3
//IC is power of 2
#ifndef XKERNEL5
#define XKERNEL5

#define Xkernel5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	pxkernel5<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.59539 msec, Performace = 1514.31 GFlop/s
//LB = 3: Size = 1.125, Time = 1.76776 msec, Performace = 1366.66 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void pxkernel5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
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
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3) << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0, x1, x2, x3;
	x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : make_float2(0, 0));
	x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : make_float2(0, 0));
	x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : make_float2(0, 0));
	x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : make_float2(0, 0));
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0, x1, x2, x3;
		x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : make_float2(0, 0));
		x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : make_float2(0, 0));
		x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : make_float2(0, 0));
		x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : make_float2(0, 0));
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
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


//FH = FW = 3
//IC is power of 2
//(OH, OW) % 4 == 0
#ifndef XKERNEL6
#define XKERNEL6

#define Xkernel6(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	pxkernel6<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.56717 msec, Performace = 1541.58 GFlop/s
//LB = 3: Size = 1.125, Time = 1.76776 msec, Performace = 1366.66 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void pxkernel6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
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
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X += X1;
	const int sw_IC = sw << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0, x1, x2, x3;
	x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : make_float2(0, 0));
	x1 = (lx1 ? *(float2*)(X + xoffset) : make_float2(0, 0));
	x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : make_float2(0, 0));
	x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : make_float2(0, 0));
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0, x1, x2, x3;
		x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : make_float2(0, 0));
		x1 = (lx1 ? *(float2*)(X + xoffset) : make_float2(0, 0));
		x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : make_float2(0, 0));
		x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : make_float2(0, 0));
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
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


#ifndef XKERNEL7
#define XKERNEL7

#define Xkernel7(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	pxkernel7<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.56111 msec, Performace = 1547.57 GFlop/s
//LB = 3: Size = 1.125, Time = 1.76776 msec, Performace = 1366.66 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void pxkernel7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
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
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X += X1;
	const int sw_IC = sw << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0, x1, x2, x3;
	x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : make_float2(0, 0));
	x1 = (lx1 ? *(float2*)(X + xoffset) : make_float2(0, 0));
	x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : make_float2(0, 0));
	x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : make_float2(0, 0));
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int woffset = (ty - ((ty >= STEP) << LB >> 1) << 1) * OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
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

		//load 4 elements from W[OC, FH, FW, IC]
		int woffset = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1) * OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0, x1, x2, x3;
		x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : make_float2(0, 0));
		x1 = (lx1 ? *(float2*)(X + xoffset) : make_float2(0, 0));
		x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : make_float2(0, 0));
		x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : make_float2(0, 0));
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
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



#ifndef FSAVE_X4
#define FSAVE_X4

struct pfloat4 { float4 v1, v2; };

__device__ __forceinline__ pfloat4 FSaveX4x(const float* __restrict__ X,
	int X_fh, int X_fw, int IH, int IW, int xoffset, int sw_IC,
	int toh0, int tow0, int tow1, int tow2, int tow3)
{
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh); IW -= X_fw;
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW);
	float2 x0, x1, x2, x3;
	x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : make_float2(0, 0));
	x1 = (lx1 ? *(float2*)(X + xoffset) : make_float2(0, 0));
	x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : make_float2(0, 0));
	x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : make_float2(0, 0));

	float4 v1; v1.x = x0.x; v1.y = x1.x; v1.z = x2.x; v1.w = x3.x;
	float4 v2; v2.x = x0.y; v2.y = x1.y; v2.z = x2.y; v2.w = x3.y;
	return pfloat4{ v1, v2 };
}

#endif

#ifndef XKERNEL8
#define XKERNEL8

#define Xkernel8(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	pxkernel8<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.56111 msec, Performace = 1547.57 GFlop/s
//LB = 3: Size = 1.125, Time = 1.76776 msec, Performace = 1366.66 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void pxkernel8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
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
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X += X1;
	const int sw_IC = sw << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	pfloat4 pf = FSaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
		toh0, tow0, tow1, tow2, tow3);
	Xs[buf][(tx << 1)][ty] = pf.v1;
	Xs[buf][(tx << 1) + 1][ty] = pf.v2;

	//load 4 elements from W[OC, FH, FW, IC]
	int woffset = (ty - ((ty >= STEP) << LB >> 1) << 1) * OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
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

		//load 4 elements from W[OC, FH, FW, IC]
		int woffset = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1) * OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		pfloat4 pf = FSaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
			toh0, tow0, tow1, tow2, tow3);
		Xs[buf][(tx << 1)][ty] = pf.v1;
		Xs[buf][(tx << 1) + 1][ty] = pf.v2;
		__syncthreads();
	}
#pragma unroll
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



#ifndef XKERNEL9
#define XKERNEL9

#define Xkernel9(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	pxkernel9<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.55801 msec, Performace = 1550.65 GFlop/s
//LB = 3: Size = 1.125, Time = 1.68134 msec, Performace = 1436.9 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void pxkernel9(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
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
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X += X1;
	const int sw_IC = sw << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : float2{ 0, 0 });
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : float2{ 0, 0 });
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : float2{ 0, 0 });
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : float2{ 0, 0 });
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int woffset = (ty - ((ty >= STEP) << LB >> 1) << 1) * OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
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

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
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

		//load 4 elements from W[OC, FH, FW, IC]
		int woffset = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1) * OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : float2{ 0, 0 });
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : float2{ 0, 0 });
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : float2{ 0, 0 });
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : float2{ 0, 0 });
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
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
