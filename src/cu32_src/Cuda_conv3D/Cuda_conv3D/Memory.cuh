


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_4R_IC_2POW
#define CONV_3D_GEMM_KERNEL_8_4R_IC_2POW

#define conv3dGemm_k84R_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_4R_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.96958 msec, Performace = 1226.62 GFlop/s
//LB = 3: Size = 1.125, Time = 2.34096 msec, Performace = 1032.02 GFlop/s
//LB = 4: Size = 1, Time = 1.76776 msec, Performace = 1214.81 GFlop/s
//LB = 3: Size = 1, Time = 2.09386 msec, Performace = 1025.61 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_4R_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	CW += oc0 + ((tx >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = LoadX2_ic2pow(X, X_k, IH, IW, LIC, FW_IC,
		X0, toh0, tow0,
		X1, toh1, tow1);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);
	Ws[buf][tx][ty] = *(float4*)(CW + W_k * OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		}
		buf ^= 1;


		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = LoadX2_ic2pow(X, X_k, IH, IW, LIC, FW_IC,
			X0, toh0, tow0,
			X1, toh1, tow1);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ws[buf][tx][ty] = *(float4*)(CW + W_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_8R_IC_2POW
#define CONV_3D_GEMM_KERNEL_4_8R_IC_2POW
	
#define conv3dGemm_k48R_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_8R_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.1456 msec, Performace = 1125.99 GFlop/s
//LB = 3: Size = 1.125, Time = 3.06069 msec, Performace = 789.338 GFlop/s
//LB = 4: Size = 1, Time = 1.92035 msec, Performace = 1118.28  GFlop/s
//LB = 3: Size = 1, Time = 2.48418 msec, Performace =  864.462 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_8R_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += ((tx & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
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
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3) << LIC;

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	Xs[buf][ty][tx] = LoadX4_ic2pow(X, X_k, IH, IW, LIC, FW_IC, 
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, b0.x, a0);
			simdMM4(v2, b0.y, a0);
			simdMM4(v4, b0.z, a0);
			simdMM4(v6, b0.w, a0);
			simdMM4(v8, b1.x, a0);
			simdMM4(v10, b1.y, a0);
			simdMM4(v12, b1.z, a0);
			simdMM4(v14, b1.w, a0);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Xs[buf][ty][tx] = LoadX4_ic2pow(X, X_k, IH, IW, LIC, FW_IC,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);

		simdMM4(v0, b0.x, a0);
		simdMM4(v2, b0.y, a0);
		simdMM4(v4, b0.z, a0);
		simdMM4(v6, b0.w, a0);
		simdMM4(v8, b1.x, a0);
		simdMM4(v10, b1.y, a0);
		simdMM4(v12, b1.z, a0);
		simdMM4(v14, b1.w, a0);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v2;
	*(float4*)(Y + j2) = v4;
	*(float4*)(Y + j3) = v6;
	*(float4*)(Y + j4) = v8;
	*(float4*)(Y + j5) = v10;
	*(float4*)(Y + j6) = v12;
	*(float4*)(Y + j7) = v14;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_4R
#define CONV_3D_GEMM_KERNEL_8_4R

#define conv3dGemm_k84R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_4R<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 1.94379 msec, Performace = 1104.79  GFlop/s
//LB = 3: Size = 1, Time = 2.28713 msec, Performace =  938.941 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_4R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	CW += oc0 + ((tx >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	int IW_IC = IW * IC, Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = LoadX2(X, X_k, IH, IW, IC, FW_IC, IW_IC,
		X0, toh0, tow0,
		X1, toh1, tow1);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);
	Ws[buf][tx][ty] = *(float4*)(CW + W_k * OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		}
		buf ^= 1;


		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = LoadX2(X, X_k, IH, IW, IC, FW_IC, IW_IC,
			X0, toh0, tow0,
			X1, toh1, tow1);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ws[buf][tx][ty] = *(float4*)(CW + W_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_8R
#define CONV_3D_GEMM_KERNEL_4_8R

#define conv3dGemm_k48R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_8R<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 2.16836 msec, Performace = 990.374 GFlop/s
//LB = 3: Size = 1, Time = 2.83054 msec, Performace = 758.682 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_8R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += ((tx & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
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
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	const int IW_IC = IW * IC;
	Xs[buf][ty][tx] = LoadX4(X, X_k, IH, IW, IC, FW_IC, IW_IC,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, b0.x, a0);
			simdMM4(v2, b0.y, a0);
			simdMM4(v4, b0.z, a0);
			simdMM4(v6, b0.w, a0);
			simdMM4(v8, b1.x, a0);
			simdMM4(v10, b1.y, a0);
			simdMM4(v12, b1.z, a0);
			simdMM4(v14, b1.w, a0);
		}
		buf ^= 1;
		
		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Xs[buf][ty][tx] = LoadX4(X, X_k, IH, IW, IC, FW_IC, IW_IC,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);

		simdMM4(v0, b0.x, a0);
		simdMM4(v2, b0.y, a0);
		simdMM4(v4, b0.z, a0);
		simdMM4(v6, b0.w, a0);
		simdMM4(v8, b1.x, a0);
		simdMM4(v10, b1.y, a0);
		simdMM4(v12, b1.z, a0);
		simdMM4(v14, b1.w, a0);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v2;
	*(float4*)(Y + j2) = v4;
	*(float4*)(Y + j3) = v6;
	*(float4*)(Y + j4) = v8;
	*(float4*)(Y + j5) = v10;
	*(float4*)(Y + j6) = v12;
	*(float4*)(Y + j7) = v14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, FW, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_8R_FW_IC_2POW
#define CONV_3D_GEMM_KERNEL_4_8R_FW_IC_2POW

#define conv3dGemm_k48R_fw_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_8R_FW_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 1.89312 msec, Performace = 1134.36  GFlop/s
//LB = 3: Size = 1, Time = 2.30257 msec, Performace =  932.648 GFlop/s
//[OC = 64]
//LB = 4: Size = 1, Time = 1.84486 msec, Performace = 1164.04 GFlop/s
//LB = 3: Size = 1, Time = 2.38942 msec, Performace = 898.748 GFlop/s

template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_8R_FW_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int LFW,
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += ((tx & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
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
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3) << LIC;

	//prepare for GK = FH * FW * IC
	const int LFW_IC = LFW + LIC, FW_IC_m1 = (1 << LFW_IC) - 1;
	const int GK = FH << LFW_IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	Xs[buf][ty][tx] = LoadX4_FW_ic2pow(X, X_k, IH, IW, LFW_IC, FW_IC_m1, LIC,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, b0.x, a0);
			simdMM4(v2, b0.y, a0);
			simdMM4(v4, b0.z, a0);
			simdMM4(v6, b0.w, a0);
			simdMM4(v8, b1.x, a0);
			simdMM4(v10, b1.y, a0);
			simdMM4(v12, b1.z, a0);
			simdMM4(v14, b1.w, a0);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Xs[buf][ty][tx] = LoadX4_FW_ic2pow(X, X_k, IH, IW, LFW_IC, FW_IC_m1, LIC,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);

		simdMM4(v0, b0.x, a0);
		simdMM4(v2, b0.y, a0);
		simdMM4(v4, b0.z, a0);
		simdMM4(v6, b0.w, a0);
		simdMM4(v8, b1.x, a0);
		simdMM4(v10, b1.y, a0);
		simdMM4(v12, b1.z, a0);
		simdMM4(v14, b1.w, a0);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v2;
	*(float4*)(Y + j2) = v4;
	*(float4*)(Y + j3) = v6;
	*(float4*)(Y + j4) = v8;
	*(float4*)(Y + j5) = v10;
	*(float4*)(Y + j6) = v12;
	*(float4*)(Y + j7) = v14;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_2_8R
#define CONV_3D_PURE_KERNEL_2_8R

#define conv3dPure_k28R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_2_8R<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>1, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//LB = 4: Size = 1.125, Time = 3.01598 msec, Performace = 801.039 GFlop/s
//LB = 3: Size = 1.125, Time = 3.54186 msec, Performace = 682.105 GFlop/s
//[OC = 64, N*2]:
//LB = 4: Size = 1.125, Time = 2.9807 msec, Performace = 810.521 GFlop/s
//LB = 3: Size = 1.125, Time = 2.69313 msec, Performace = 897.069 GFlop/s
//[OC = 32, N*4]:
//LB = 4: Size = 1.125, Time = 3.20552 msec, Performace = 753.675 GFlop/s
//LB = 3: Size = 1.125, Time = 3.82278 msec, Performace = 631.979 GFlop/s
//[OC = 16, N*8]
//LB = 3: Size = 1.125, Time = 4.5541 msec, Performace = 530.494 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_2_8R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 1) + oc_index;
	CW += oc0 + (ty & 1);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int tihs2 = toh2 * sh - ph, tiws2 = tow2 * sw - pw;
	const int tihs3 = toh3 * sh - ph, tiws3 = tow3 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;
	const int X2 = (((tn2 *IH) + tihs2) * IW + tiws2) * IC;
	const int X3 = (((tn3 *IH) + tihs3) * IW + tiws3) * IC;

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	float2 v4 = make_float2(0, 0);
	float2 v5 = make_float2(0, 0);
	float2 v6 = make_float2(0, 0);
	float2 v7 = make_float2(0, 0);

	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC, SCW = IC * OC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, CW += SCW)
		{
			//load 4 elem from X
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			bool lx2 = LOAD_X(tihs2, tiws2, fh, fw);
			bool lx3 = LOAD_X(tihs3, tiws3, fh, fw);
			float4 x; int Xic = tx - ((tx >= STEP) << LB >> 1);
			x.x = (lx0 ? X[X0 + Xic] : 0);
			x.y = (lx1 ? X[X1 + Xic] : 0);
			x.z = (lx2 ? X[X2 + Xic] : 0);
			x.w = (lx3 ? X[X3 + Xic] : 0);
			Xs[buf][tx][ty] = x;

			//load 1 elem from W
			int Wic = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = CW[Wic * OC];
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
					float2 a0 = *(float2*)(&Ws[buf][ik][tx << 1]);

					simdMM2(v0, b0.x, a0);
					simdMM2(v1, b0.y, a0);
					simdMM2(v2, b0.z, a0);
					simdMM2(v3, b0.w, a0);
					simdMM2(v4, b1.x, a0);
					simdMM2(v5, b1.y, a0);
					simdMM2(v6, b1.z, a0);
					simdMM2(v7, b1.w, a0);
				}
				buf ^= 1;

				//load 2 elem from X
				float4 x; int Xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
				x.x = (lx0 ? X[X0 + Xic] : 0);
				x.y = (lx1 ? X[X1 + Xic] : 0);
				x.z = (lx2 ? X[X2 + Xic] : 0);
				x.w = (lx3 ? X[X3 + Xic] : 0);
				Xs[buf][tx][ty] = x;

				//load 1 elem from W
				int Wic = ((oic << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = CW[Wic * OC];
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
				float2 a0 = *(float2*)(&Ws[buf][ik][tx << 1]);

				simdMM2(v0, b0.x, a0);
				simdMM2(v1, b0.y, a0);
				simdMM2(v2, b0.z, a0);
				simdMM2(v3, b0.w, a0);
				simdMM2(v4, b1.x, a0);
				simdMM2(v5, b1.y, a0);
				simdMM2(v6, b1.z, a0);
				simdMM2(v7, b1.w, a0);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
	*(float2*)(Y + j2) = v2;
	*(float2*)(Y + j3) = v3;
	*(float2*)(Y + j4) = v4;
	*(float2*)(Y + j5) = v5;
	*(float2*)(Y + j6) = v6;
	*(float2*)(Y + j7) = v7;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_2_8R_TEXTURE
#define CONV_3D_PURE_KERNEL_2_8R_TEXTURE

#define conv3dPure_k28R_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_2_8R_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>1, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.45624 msec, Performace = 983.583 GFlop/s
//LB = 3: Size = 1.125, Time = 3.37415 msec, Performace = 716.009 GFlop/s
//[OC = 64, N*2]:
//LB = 4: Size = 1.125, Time = 2.4549 msec, Performace = 984.122 GFlop/s
//LB = 3: Size = 1.125, Time = 3.3909 msec, Performace = 712.471 GFlop/s
//[OC = 32, N*4]:
//LB = 4: Size = 1.125, Time = 2.48972 msec, Performace = 970.357 GFlop/s
//LB = 3: Size = 1.125, Time = 3.54042 msec, Performace = 682.383 GFlop/s
//[OC = 16, N*8]
//LB = 3: Size = 1.125, Time = 4.61472 msec, Performace = 523.524 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_2_8R_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 1) + oc_index;
	CW += oc0 + (ty & 1);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int tihs2 = toh2 * sh - ph, tiws2 = tow2 * sw - pw;
	const int tihs3 = toh3 * sh - ph, tiws3 = tow3 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;
	const int X2 = (((tn2 *IH) + tihs2) * IW + tiws2) * IC;
	const int X3 = (((tn3 *IH) + tihs3) * IW + tiws3) * IC;

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	float2 v4 = make_float2(0, 0);
	float2 v5 = make_float2(0, 0);
	float2 v6 = make_float2(0, 0);
	float2 v7 = make_float2(0, 0);

	int xoffset = 0;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC, SCW = IC * OC;
	for (int fh = 0; fh < FH; fh++, xoffset += SX) {
		for (int fw = 0; fw < FW; fw++, xoffset += IC, CW += SCW)
		{
			//load 4 elem from X
			float4 x; int Xic = xoffset + tx - ((tx >= STEP) << LB >> 1);
			x.x = tex1Dfetch<float>(X, X0 + Xic);
			x.y = tex1Dfetch<float>(X, X1 + Xic);
			x.z = tex1Dfetch<float>(X, X2 + Xic);
			x.w = tex1Dfetch<float>(X, X3 + Xic);
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw); zero_float(x.x, lx0, x.x);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw); zero_float(x.y, lx1, x.y);
			bool lx2 = LOAD_X(tihs2, tiws2, fh, fw); zero_float(x.z, lx2, x.z);
			bool lx3 = LOAD_X(tihs3, tiws3, fh, fw); zero_float(x.w, lx3, x.w);
			Xs[buf][tx][ty] = x;

			//load 1 elem from W
			int Wic = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = CW[Wic * OC];
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
					float2 a0 = *(float2*)(&Ws[buf][ik][tx << 1]);

					simdMM2(v0, b0.x, a0);
					simdMM2(v1, b0.y, a0);
					simdMM2(v2, b0.z, a0);
					simdMM2(v3, b0.w, a0);
					simdMM2(v4, b1.x, a0);
					simdMM2(v5, b1.y, a0);
					simdMM2(v6, b1.z, a0);
					simdMM2(v7, b1.w, a0);
				}
				buf ^= 1;

				//load 4 elem from X
				float4 x; int Xic = xoffset + ((oic - (tx >= STEP)) << LB >> 1) + tx;
				x.x = tex1Dfetch<float>(X, X0 + Xic);
				x.y = tex1Dfetch<float>(X, X1 + Xic);
				x.z = tex1Dfetch<float>(X, X2 + Xic);
				x.w = tex1Dfetch<float>(X, X3 + Xic);
				zero_float(x.x, lx0, x.x);
				zero_float(x.y, lx1, x.y);
				zero_float(x.z, lx2, x.z);
				zero_float(x.w, lx3, x.w);
				Xs[buf][tx][ty] = x;

				//load 1 elem from W
				int Wic = ((oic << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = CW[Wic * OC];
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
				float2 a0 = *(float2*)(&Ws[buf][ik][tx << 1]);

				simdMM2(v0, b0.x, a0);
				simdMM2(v1, b0.y, a0);
				simdMM2(v2, b0.z, a0);
				simdMM2(v3, b0.w, a0);
				simdMM2(v4, b1.x, a0);
				simdMM2(v5, b1.y, a0);
				simdMM2(v6, b1.z, a0);
				simdMM2(v7, b1.w, a0);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
	*(float2*)(Y + j2) = v2;
	*(float2*)(Y + j3) = v3;
	*(float2*)(Y + j4) = v4;
	*(float2*)(Y + j5) = v5;
	*(float2*)(Y + j6) = v6;
	*(float2*)(Y + j7) = v7;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_4_IC_2POW
#define CONV_3D_GEMM_KERNEL_4_4_IC_2POW

#define conv3dGemm_k44_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_4_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 2.36026 msec, Performace = 909.851 GFlop/s
//LB = 3: Size = 1, Time = 3.53646 msec, Performace = 607.24 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_4_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//[1 << LB >> 1][2 << LB]
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//[1 << LB >> 1][2 << LB]

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	int toc0 = (((tx & 1) << 1) + oc0)*GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
	Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = LoadX2_ic2pow(X, X_k, IH, IW, LIC, FW_IC,
		X0, toh0, tow0,
		X1, toh1, tow1);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, b.x, a);
			simdMM4(v1, b.y, a);
			simdMM4(v2, b.z, a);
			simdMM4(v3, b.w, a);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = LoadX2_ic2pow(X, X_k, IH, IW, LIC, FW_IC,
			X0, toh0, tow0,
			X1, toh1, tow1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

		simdMM4(v0, b.x, a);
		simdMM4(v1, b.y, a);
		simdMM4(v2, b.z, a);
		simdMM4(v3, b.w, a);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0, FW, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_4_FW_IC_2POW
#define CONV_3D_GEMM_KERNEL_4_4_FW_IC_2POW

#define conv3dGemm_k44_fw_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, W, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_4_FW_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 2.22807 msec, Performace = 963.832 GFlop/s
//LB = 3: Size = 1, Time = 3.46019 msec, Performace = 620.627 GFlop/s
	template<int LB, int STEP>
	__global__ void conv3dGemm_kernel_4_4_FW_IC2pow(
		const float* __restrict__ X, int IH, int IW,
		const float* __restrict__ W, int FH, int LFW,
		float* __restrict__ Y, int OH, int OW,
		int LIC, int OC,
		int sh, int sw, int ph, int pw,
		int oc_index, int j_index)
	{
		int ty = threadIdx.y, tx = threadIdx.x;

		bool buf = 0;
		__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//[1 << LB >> 1][2 << LB]
		__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//[1 << LB >> 1][2 << LB]

		//prepare for GK = FH * FW * IC
		const int LFW_IC = LFW + LIC, LFW_IC_m1 = (1 << LFW_IC) - 1;
		const int GK = FH << LFW_IC;

		//prepare for GN = OC
		const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
		int toc0 = (((tx & 1) << 1) + oc0)*GK, toc1 = toc0 + GK;

		//prepare for GM = N * OH * OW
		int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
		int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
		const int OH_OW = OH * OW;
		get_n_oh_ow(tj0, tn0, toh0, tow0);
		get_n_oh_ow(tj1, tn1, toh1, tow1);
		toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
		toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
		const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
		const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = tx >> 1;
		const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ty >> 1;
		const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
		Xs[buf][Xs_y][Xs_x] = LoadX2_FW_ic2pow(X, X_k, IH, IW, LFW_IC, LFW_IC_m1, LIC,
			X0, toh0, tow0,
			X1, toh1, tow1);
		__syncthreads();

		//compute area----------------------------------------------------
		float4 v0 = make_float4(0, 0, 0, 0);
		float4 v1 = make_float4(0, 0, 0, 0);
		float4 v2 = make_float4(0, 0, 0, 0);
		float4 v3 = make_float4(0, 0, 0, 0);
		for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
				float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

				simdMM4(v0, b.x, a);
				simdMM4(v1, b.y, a);
				simdMM4(v2, b.z, a);
				simdMM4(v3, b.w, a);
			}
			buf ^= 1;

			//load 2 elements from W[OC, FH, FW, IC]
			int W_k = ((ok << LB) + tx) >> 1;
			Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
			Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

			//load 2 elements from X[N, IH, IW, IC]
			int X_k = ((ok << LB) + ty) >> 1;
			Xs[buf][Xs_y][Xs_x] = LoadX2_FW_ic2pow(X, X_k, IH, IW, LFW_IC, LFW_IC_m1, LIC,
				X0, toh0, tow0,
				X1, toh1, tow1);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, b.x, a);
			simdMM4(v1, b.y, a);
			simdMM4(v2, b.z, a);
			simdMM4(v3, b.w, a);
		}

		j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
		int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

		*(float4*)(Y + j0) = v0;
		*(float4*)(Y + j1) = v1;
		*(float4*)(Y + j2) = v2;
		*(float4*)(Y + j3) = v3;
	}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_8_FW_IC_2POW
#define CONV_3D_GEMM_KERNEL_4_8_FW_IC_2POW

#define conv3dGemm_k48_fw_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, W, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_8_FW_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 2.08944 msec, Performace = 1027.78  GFlop/s
//LB = 3: Size = 1, Time = 2.63061 msec, Performace =  816.343 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_8_FW_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int LFW,
	      float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GK = FH * FW * IC
	const int LFW_IC = LFW + LIC, LFW_IC_m1 = (1 << LFW_IC) - 1;
	const int GK = FH << LFW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	int toc0 = (((tx & 1) << 1) + oc0)*GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
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
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3) << LIC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
	Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	Xs[buf][ty][tx] = LoadX4_FW_ic2pow(X, X_k, IH, IW, LFW_IC, LFW_IC_m1, LIC,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4( v0, b0.x, a0);
			simdMM4( v2, b0.y, a0);
			simdMM4( v4, b0.z, a0);
			simdMM4( v6, b0.w, a0);
			simdMM4( v8, b1.x, a0);
			simdMM4(v10, b1.y, a0);
			simdMM4(v12, b1.z, a0);
			simdMM4(v14, b1.w, a0);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Xs[buf][ty][tx] = LoadX4_FW_ic2pow(X, X_k, IH, IW, LFW_IC, LFW_IC_m1, LIC,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4( v0, b0.x, a0);
		simdMM4( v2, b0.y, a0);
		simdMM4( v4, b0.z, a0);
		simdMM4( v6, b0.w, a0);
		simdMM4( v8, b1.x, a0);
		simdMM4(v10, b1.y, a0);
		simdMM4(v12, b1.z, a0);
		simdMM4(v14, b1.w, a0);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v2;
	*(float4*)(Y + j2) = v4;
	*(float4*)(Y + j3) = v6;
	*(float4*)(Y + j4) = v8;
	*(float4*)(Y + j5) = v10;
	*(float4*)(Y + j6) = v12;
	*(float4*)(Y + j7) = v14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_8_IC_2POW
#define CONV_3D_GEMM_KERNEL_4_8_IC_2POW

#define conv3dGemm_k48_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_8_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 2.1827 msec, Performace = 983.865 GFlop/s
//LB = 3: Size = 1, Time = 2.77649 msec, Performace = 773.452 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_8_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	int toc0 = (((tx & 1) << 1) + oc0)*GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
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
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3) << LIC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
	Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	Xs[buf][ty][tx] = LoadX4_ic2pow(X, X_k, IH, IW, LIC, FW_IC,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0);
			simdMM4(v2, b0.y, a0);
			simdMM4(v4, b0.z, a0);
			simdMM4(v6, b0.w, a0);
			simdMM4(v8, b1.x, a0);
			simdMM4(v10, b1.y, a0);
			simdMM4(v12, b1.z, a0);
			simdMM4(v14, b1.w, a0);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Xs[buf][ty][tx] = LoadX4_ic2pow(X, X_k, IH, IW, LIC, FW_IC,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0);
		simdMM4(v2, b0.y, a0);
		simdMM4(v4, b0.z, a0);
		simdMM4(v6, b0.w, a0);
		simdMM4(v8, b1.x, a0);
		simdMM4(v10, b1.y, a0);
		simdMM4(v12, b1.z, a0);
		simdMM4(v14, b1.w, a0);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v2;
	*(float4*)(Y + j2) = v4;
	*(float4*)(Y + j3) = v6;
	*(float4*)(Y + j4) = v8;
	*(float4*)(Y + j5) = v10;
	*(float4*)(Y + j6) = v12;
	*(float4*)(Y + j7) = v14;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_4
#define CONV_3D_GEMM_KERNEL_8_4

#define conv3dGemm_k84(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 2.26531 msec, Performace = 947.988 GFlop/s
//LB = 3: Size = 1, Time = 2.96239 msec, Performace = 724.916 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	int toc0 = (oc0 + ((tx >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];
	Ws[buf][tx][ty].z = W[toc2 + W_k];
	Ws[buf][tx][ty].w = W[toc3 + W_k];

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	const int IW_IC = IW * IC, Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = LoadX2(X, X_k, IH, IW, IC, FW_IC, IW_IC,
		X0, toh0, tow0,
		X1, toh1, tow1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];
		Ws[buf][tx][ty].z = W[toc2 + W_k];
		Ws[buf][tx][ty].w = W[toc3 + W_k];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = LoadX2(X, X_k, IH, IW, IC, FW_IC, IW_IC,
			X0, toh0, tow0,
			X1, toh1, tow1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_8
#define CONV_3D_GEMM_KERNEL_4_8

#define conv3dGemm_k48(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_8<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 2.30123 msec, Performace = 933.19  GFlop/s
//LB = 3: Size = 1, Time = 2.90783 msec, Performace = 738.517 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	int toc0 = (((tx & 1) << 1) + oc0)*GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
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

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
	Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	const int IW_IC = IW * IC;
	Xs[buf][ty][tx] = LoadX4(X, X_k, IH, IW, IC, FW_IC, IW_IC,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, b0.x, a0);
			simdMM4(v2, b0.y, a0);
			simdMM4(v4, b0.z, a0);
			simdMM4(v6, b0.w, a0);
			simdMM4(v8, b1.x, a0);
			simdMM4(v10, b1.y, a0);
			simdMM4(v12, b1.z, a0);
			simdMM4(v14, b1.w, a0);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Xs[buf][ty][tx] = LoadX4(X, X_k, IH, IW, IC, FW_IC, IW_IC,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, b0.x, a0);
		simdMM4(v2, b0.y, a0);
		simdMM4(v4, b0.z, a0);
		simdMM4(v6, b0.w, a0);
		simdMM4(v8, b1.x, a0);
		simdMM4(v10, b1.y, a0);
		simdMM4(v12, b1.z, a0);
		simdMM4(v14, b1.w, a0);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v2;
	*(float4*)(Y + j2) = v4;
	*(float4*)(Y + j3) = v6;
	*(float4*)(Y + j4) = v8;
	*(float4*)(Y + j5) = v10;
	*(float4*)(Y + j6) = v12;
	*(float4*)(Y + j7) = v14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_2
#define CONV_3D_GEMM_KERNEL_4_2

#define conv3dGemm_k42(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 3.474 msec, Performace = 618.159 GFlop/s
//LB = 3: Size = 1, Time = 5.53  msec, Performace = 388.333 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	int toc0 = (((tx & 1) << 1) + oc0)*GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int tj0 = j0 + (ty & 1);
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	const int Xoffset0 = ((tn0*IH + toh0)*IW + tow0)*IC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
	Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

	//load 1 element from X[N, IH, IW, IC]
	int X_k = ty >> 1, X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][Xs_y][Xs_x] = (lx0 ? X[Xoffset0 + xoffset] : 0);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);
			simdMM4(v0, b.x, a);
			simdMM4(v1, b.y, a);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

		//load 1 element1 from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][Xs_y][Xs_x] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);
		simdMM4(v0, b.x, a);
		simdMM4(v1, b.y, a);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_2_4
#define CONV_3D_GEMM_KERNEL_2_4

#define conv3dGemm_k24(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_2_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1  , Time = 3.954 msec, Performace = 543.117 GFlop/s
//LB = 3: Size = 0.5, Time = 3.4 msec, Performace = 315.806 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_2_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	int toc0 = ((tx & 1) + oc0)*GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	const int Xoffset0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int Xoffset1 = ((tn1*IH + toh1)*IW + tow1)*IC;

	//load 1 element from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = W[toc0 + W_k];

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1, X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][Xs_y][Xs_x].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
	Xs[buf][Xs_y][Xs_x].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			float2 a = *(float2*)(&Ws[buf][ik][ty << 1]);

			simdMM2(v0, b.x, a);
			simdMM2(v1, b.y, a);
			simdMM2(v2, b.z, a);
			simdMM2(v3, b.w, a);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y] = W[toc0 + W_k];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][Xs_y][Xs_x].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		Xs[buf][Xs_y][Xs_x].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		float2 a = *(float2*)(&Ws[buf][ik][ty << 1]);

		simdMM2(v0, b.x, a);
		simdMM2(v1, b.y, a);
		simdMM2(v2, b.z, a);
		simdMM2(v3, b.w, a);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
	*(float2*)(Y + j2) = v2;
	*(float2*)(Y + j3) = v3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_4_IC_2POW
#define CONV_3D_GEMM_KERNEL_8_4_IC_2POW

#define conv3dGemm_k84_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_4_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 1.96448 msec, Performace = 1093.16 GFlop/s
//LB = 3: Size = 1, Time = 2.54084 msec, Performace = 845.185 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_4_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	      float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	int toc0 = (oc0 + ((tx >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];
	Ws[buf][tx][ty].z = W[toc2 + W_k];
	Ws[buf][tx][ty].w = W[toc3 + W_k];

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = LoadX2_ic2pow(X, X_k, IH, IW, LIC, FW_IC,
		X0, toh0, tow0,
		X1, toh1, tow1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];
		Ws[buf][tx][ty].z = W[toc2 + W_k];
		Ws[buf][tx][ty].w = W[toc3 + W_k];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = LoadX2_ic2pow(X, X_k, IH, IW, LIC, FW_IC,
			X0, toh0, tow0,
			X1, toh1, tow1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_4_FW_IC_2POW
#define CONV_3D_GEMM_KERNEL_8_4_FW_IC_2POW

#define conv3dGemm_k84_fw_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, W, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_4_FW_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 1.85643 msec, Performace = 1156.78  GFlop/s
//LB = 3: Size = 1, Time = 2.50735 msec, Performace =  856.474 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_4_FW_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int LFW,
	      float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//prepare for GK = FH * FW * IC
	const int LFW_IC = LFW + LIC, LFW_IC_m1 = (1 << LFW_IC) - 1;
	const int GK = FH << LFW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	int toc0 = (oc0 + ((tx >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];
	Ws[buf][tx][ty].z = W[toc2 + W_k];
	Ws[buf][tx][ty].w = W[toc3 + W_k];

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = LoadX2_FW_ic2pow(X, X_k, IH, IW, LFW_IC, LFW_IC_m1, LIC,
		X0, toh0, tow0,
		X1, toh1, tow1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];
		Ws[buf][tx][ty].z = W[toc2 + W_k];
		Ws[buf][tx][ty].w = W[toc3 + W_k];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = LoadX2_FW_ic2pow(X, X_k, IH, IW, LFW_IC, LFW_IC_m1, LIC,
			X0, toh0, tow0,
			X1, toh1, tow1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_4R_IC_2POW
#define CONV_3D_GEMM_KERNEL_4_4R_IC_2POW

#define conv3dGemm_k44R_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_4R_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 2.28265 msec, Performace = 940.786 GFlop/s
//LB = 3: Size = 1, Time = 3.09519 msec, Performace = 693.812 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_4R_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += ((tx & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = LoadX2_ic2pow(X, X_k, IH, IW, LIC, FW_IC,
		X0, toh0, tow0,
		X1, toh1, tow1);

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, b.x, a);
			simdMM4(v1, b.y, a);
			simdMM4(v2, b.z, a);
			simdMM4(v3, b.w, a);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = LoadX2_ic2pow(X, X_k, IH, IW, LIC, FW_IC,
			X0, toh0, tow0,
			X1, toh1, tow1);

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

		simdMM4(v0, b.x, a);
		simdMM4(v1, b.y, a);
		simdMM4(v2, b.z, a);
		simdMM4(v3, b.w, a);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0, FW, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_4R_FW_IC_2POW
#define CONV_3D_GEMM_KERNEL_4_4R_FW_IC_2POW

#define conv3dGemm_k44R_fw_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_4R_FW_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 2.15283 msec, Performace = 997.517 GFlop/s
//LB = 3: Size = 1, Time = 2.99275 msec, Performace = 717.562 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_4R_FW_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int LFW,
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += ((tx & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;

	//prepare for GK = FH * FW * IC
	const int LFW_IC = LFW + LIC, FW_IC_m1 = (1 << LFW_IC) - 1;
	const int GK = FH << LFW_IC;

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = LoadX2_FW_ic2pow(X, X_k, IH, IW, LFW_IC, FW_IC_m1, LIC,
		X0, toh0, tow0,
		X1, toh1, tow1);

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, b.x, a);
			simdMM4(v1, b.y, a);
			simdMM4(v2, b.z, a);
			simdMM4(v3, b.w, a);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = LoadX2_FW_ic2pow(X, X_k, IH, IW, LFW_IC, FW_IC_m1, LIC,
			X0, toh0, tow0,
			X1, toh1, tow1);

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

		simdMM4(v0, b.x, a);
		simdMM4(v1, b.y, a);
		simdMM4(v2, b.z, a);
		simdMM4(v3, b.w, a);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
}

#endif
