


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), OH, OW, OC, IC is power of 2, IC % 4 == 0 
//LB = 4: GK % 8 == 0
#ifndef X_BGEMM1
#define X_BGEMM1

#define xbgemm1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, N, LIC, LOC, sh, sw, ph, pw, GN, GM)\
	XBGemm1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 N, LIC, LOC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 1.474 msec, Performace = 1456.91 GFlop/s
//LB = 3: Size = 1, Time = 1.678 msec, Performace = 1279.79 GFlop/s
template<int LB, int STEP>
__global__ void XBGemm1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int LIC, int LOC,
	int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK_slice = (N_end - N_start) << LOH_OW;
	X += (N_start * IH*IW) << LIC; //X[N_start,...]
	deltaY += (N_start << LOH_OW) << LOC; //deltaY[N_start,...]

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW << LIC, FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + ((bz - 1) << LOC) *FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	const int IC_m1 = (1 << LIC) - 1;
	get_fh_fw_ic_IC2pow(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += ((tfh0*IW + tfw0) << LIC) + tic0;

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow_s2(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow) << LIC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + (Y_k << LOC));
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow_s2(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow) << LIC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + (Y_k << LOC));
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  *(float4*)(dst + oc0 + 4) = v1;
	*(float4*)(dst + oc1) = v2;  *(float4*)(dst + oc1 + 4) = v3;
	*(float4*)(dst + oc2) = v4;  *(float4*)(dst + oc2 + 4) = v5;
	*(float4*)(dst + oc3) = v6;  *(float4*)(dst + oc3 + 4) = v7;
	*(float4*)(dst + oc4) = v8;  *(float4*)(dst + oc4 + 4) = v9;
	*(float4*)(dst + oc5) = v10; *(float4*)(dst + oc5 + 4) = v11;
	*(float4*)(dst + oc6) = v12; *(float4*)(dst + oc6 + 4) = v13;
	*(float4*)(dst + oc7) = v14; *(float4*)(dst + oc7 + 4) = v15;
}

#endif


//IH = OH * sh, IW = OW * sw
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), OH, OW, OC, IC is power of 2, IC % 4 == 0 
//LB = 4: GK % 8 == 0
#ifndef X_BGEMM2
#define X_BGEMM2

#define xbgemm2(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, N, LIC, LOC, sh, sw, ph, pw, GN, GM)\
	XBGemm2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 N, LIC, LOC, ph, pw, oc_index, j_index)

#define Xget_X_oh_ow_OHW2pow_s2(k, oh, ow) int oh, ow; {\
	int kr = k & OH_OW_m1;\
	oh = kr >> LOW; ow = kr & OW_m1;\
	oh <<= 1; ow <<= 1;}


//LB = 4: Size = 1, Time = 1.466 msec, Performace = 1464.86 GFlop/s
//LB = 3: Size = 1, Time = 1.666 msec, Performace = 1289.01 GFlop/s
template<int LB, int STEP>
__global__ void XBGemm2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int LIC, int LOC,
	int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OW_m1 = (1 << LOW) - 1;
	const int OH_OW_m1 = (1 << LOH << LOW) - 1;
	const int GK_slice = (N_end - N_start) << LOH << LOW;
	X += (N_start * IH*IW) << LIC; //X[N_start,...]
	deltaY += (N_start << LOH << LOW) << LOC; //deltaY[N_start,...]

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW << LIC, FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + ((bz - 1) << LOC) * FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	const int IC_m1 = (1 << LIC) - 1;
	get_fh_fw_ic_IC2pow(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += ((tfh0*IW + tfw0) << LIC) + tic0;

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	Xget_X_oh_ow_OHW2pow_s2(X_k, X_oh, X_ow);
	int xoffset = ((X_k << 2) - X_ow) << LIC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + (Y_k << LOC));
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;
	
		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Xget_X_oh_ow_OHW2pow_s2(X_k, X_oh, X_ow);
		int xoffset = ((X_k << 2) - X_ow) << LIC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + (Y_k << LOC));
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  *(float4*)(dst + oc0 + 4) = v1;
	*(float4*)(dst + oc1) = v2;  *(float4*)(dst + oc1 + 4) = v3;
	*(float4*)(dst + oc2) = v4;  *(float4*)(dst + oc2 + 4) = v5;
	*(float4*)(dst + oc3) = v6;  *(float4*)(dst + oc3 + 4) = v7;
	*(float4*)(dst + oc4) = v8;  *(float4*)(dst + oc4 + 4) = v9;
	*(float4*)(dst + oc5) = v10; *(float4*)(dst + oc5 + 4) = v11;
	*(float4*)(dst + oc6) = v12; *(float4*)(dst + oc6 + 4) = v13;
	*(float4*)(dst + oc7) = v14; *(float4*)(dst + oc7 + 4) = v15;
}

#endif


//IH = OH * sh, IW = OW * sw
//sh = sw = 2
//IH = OH << 1, IW = OW << 1
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), OH, OW, OC, IC is power of 2, IC % 4 == 0 
//LB = 4: GK % 8 == 0
#ifndef X_BGEMM3
#define X_BGEMM3

#define xbgemm3(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, N, LIC, LOC, sh, sw, ph, pw, GN, GM)\
	XBGemm3<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 N, LIC, LOC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 1.466 msec, Performace = 1464.86 GFlop/s
//LB = 3: Size = 1, Time = 1.666 msec, Performace = 1289.01 GFlop/s
template<int LB, int STEP>
__global__ void XBGemm3(
	const float* __restrict__ X, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int LIC, int LOC,
	int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = IF_int((bz != (gridDim.z - 1)), N_start + N_slice, N);//min(N_start + Nslice, N)
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH << LOW) - 1;
	const int GK_slice = (N_end - N_start) << LOH << LOW;
	X += N_start << LOH_OW << 2 << LIC; //X[N_start,...]
	deltaY += N_start << LOH_OW << LOC; //deltaY[N_start,...]

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW << LIC, FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + ((bz - 1) << LOC) * FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	const int IC_m1 = (1 << LIC) - 1;
	get_fh_fw_ic_IC2pow(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (((tfh0 << LOW << 1) + tfw0) << LIC) + tic0;
	LOH = (2 << LOH) - tfh0;
	IW -= tfw0;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + (Y_k << LOC));

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	Xget_X_oh_ow_OHW2pow_s2(X_k, X_oh, X_ow);
	int xoffset = ((X_k << 2) - X_ow) << LIC;
	bool lx = (X_oh >= -tfh0) && (X_oh < LOH) && (X_ow >= -tfw0) && (X_ow < IW);
	Xs[buf][ty][tx] = (lx ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + (Y_k << LOC));

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Xget_X_oh_ow_OHW2pow_s2(X_k, X_oh, X_ow);
		int xoffset = ((X_k << 2) - X_ow) << LIC;
		bool lx = (X_oh >= -tfh0) && (X_oh < LOH) && (X_ow >= -tfw0) && (X_ow < IW);
		Xs[buf][ty][tx] = (lx ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  *(float4*)(dst + oc0 + 4) = v1;
	*(float4*)(dst + oc1) = v2;  *(float4*)(dst + oc1 + 4) = v3;
	*(float4*)(dst + oc2) = v4;  *(float4*)(dst + oc2 + 4) = v5;
	*(float4*)(dst + oc3) = v6;  *(float4*)(dst + oc3 + 4) = v7;
	*(float4*)(dst + oc4) = v8;  *(float4*)(dst + oc4 + 4) = v9;
	*(float4*)(dst + oc5) = v10; *(float4*)(dst + oc5 + 4) = v11;
	*(float4*)(dst + oc6) = v12; *(float4*)(dst + oc6 + 4) = v13;
	*(float4*)(dst + oc7) = v14; *(float4*)(dst + oc7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), OH, OW, OC, IC is power of 2
//LB = 4: GK % 8 == 0
#ifndef X_BGEMM4
#define X_BGEMM4

#define xbgemm4(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, N, LIC, LOC, sh, sw, ph, pw, GN, GM)\
	XBGemm4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 N, LIC, LOC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 1, Time = 1.508 msec, Performace = 1424.06 GFlop/s
//LB = 3: Size = 1, Time = 1.748 msec, Performace = 1228.54 GFlop/s
template<int LB, int STEP>
__global__ void XBGemm4(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int LIC, int LOC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK_slice = (N_end - N_start) << LOH_OW;
	int XNoffset = (N_start * IH*IW) << LIC; //X[N_start,...]
	deltaY += (N_start << LOH_OW) << LOC; //deltaY[N_start,...]

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW << LIC, FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + ((bz - 1) << LOC) *FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	const int IC_m1 = (1 << LIC) - 1;
	get_fh_fw_ic_IC2pow(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	const int X0 = ((tfh0*IW + tfw0) << LIC) + tic0 + XNoffset;
	const int X1 = X0 + 1, X2 = X0 + 2, X3 = X0 + 3;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + (Y_k << LOC));

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow) << LIC;
	zero_float(Xs[buf][ty][tx].x, lx0, tex1Dfetch<float>(X, X0 + xoffset));
	zero_float(Xs[buf][ty][tx].y, lx0, tex1Dfetch<float>(X, X1 + xoffset));
	zero_float(Xs[buf][ty][tx].z, lx0, tex1Dfetch<float>(X, X2 + xoffset));
	zero_float(Xs[buf][ty][tx].w, lx0, tex1Dfetch<float>(X, X3 + xoffset));
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + (Y_k << LOC));

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow) << LIC;
		zero_float(Xs[buf][ty][tx].x, lx0, tex1Dfetch<float>(X, X0 + xoffset));
		zero_float(Xs[buf][ty][tx].y, lx0, tex1Dfetch<float>(X, X1 + xoffset));
		zero_float(Xs[buf][ty][tx].z, lx0, tex1Dfetch<float>(X, X2 + xoffset));
		zero_float(Xs[buf][ty][tx].w, lx0, tex1Dfetch<float>(X, X3 + xoffset));
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  *(float4*)(dst + oc0 + 4) = v1;
	*(float4*)(dst + oc1) = v2;  *(float4*)(dst + oc1 + 4) = v3;
	*(float4*)(dst + oc2) = v4;  *(float4*)(dst + oc2 + 4) = v5;
	*(float4*)(dst + oc3) = v6;  *(float4*)(dst + oc3 + 4) = v7;
	*(float4*)(dst + oc4) = v8;  *(float4*)(dst + oc4 + 4) = v9;
	*(float4*)(dst + oc5) = v10; *(float4*)(dst + oc5 + 4) = v11;
	*(float4*)(dst + oc6) = v12; *(float4*)(dst + oc6 + 4) = v13;
	*(float4*)(dst + oc7) = v14; *(float4*)(dst + oc7 + 4) = v15;
}

#endif

