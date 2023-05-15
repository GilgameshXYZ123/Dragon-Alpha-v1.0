

//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef UERNEL_8_8_V1
#define UERNEL_8_8_V1

#define usk88_v1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	uernel_SK_8_8_v1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			  N, IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//synchronized: 
//GK_slice = 1024, OH = OW =  4, N = 256:
//LB = 4: Size = 1, Time = 1.602 msec, Performace = 1340.5 GFlop/s
//LB = 3: Size = 1, Time = 2.072 msec, Performace = 1036.43 GFlop/s
//GK_slice = 1024, OH = OW = 7, IH = IW = 14
//LB = 4: Size = 0.861328, Time = 1.38 msec, Performace = 1340.35 GFlop/s
//LB = 3: Size = 0.861328, Time = 1.708 msec, Performace = 1082.96 GFlop/s
//for[64, 64] -> [32, 32]: Size = 1.125, Time = 1.782 msec, Performace = 1355.73 GFlop/s
template<int LB, int STEP>
__global__ void uernel_SK_8_8_v1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	const int OW_N = OW * N;
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_oh_ow_n(X_k, X_oh, X_ow, X_n); X_oh *= sh; X_ow *= sw;
	int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
	float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
	if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
	Xs[buf][ty][tx] = xv;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n, Y_oh, Y_ow; get_oh_ow_n(Y_k, Y_ow, Y_oh, Y_n);
	int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
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

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n, X_oh, X_ow; get_oh_ow_n(X_k, X_oh, X_ow, X_n); X_oh *= sh; X_ow *= sw;
		int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
		float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
		if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
		Xs[buf][ty][tx] = xv;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n, Y_oh, Y_ow; get_oh_ow_n(Y_k, Y_ow, Y_oh, Y_n);
		int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
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

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
	*(float4*)(deltaW + oc4) = v8;  *(float4*)(deltaW + oc4 + 4) = v9;
	*(float4*)(deltaW + oc5) = v10; *(float4*)(deltaW + oc5 + 4) = v11;
	*(float4*)(deltaW + oc6) = v12; *(float4*)(deltaW + oc6 + 4) = v13;
	*(float4*)(deltaW + oc7) = v14; *(float4*)(deltaW + oc7 + 4) = v15;
}

#endif



//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef UERNEL_8_8_V2
#define UERNEL_8_8_V2

#define usk88_v2(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	uernel_SK_8_8_v2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			  N, IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//synchronized: 
//GK_slice = 1024, OH = OW =  4, N = 256:
//LB = 4: Size = 1, Time = 1.602 msec, Performace = 1340.5 GFlop/s
//LB = 3: Size = 1, Time = 2.072 msec, Performace = 1036.43 GFlop/s
//GK_slice = 1024, OH = OW = 7, IH = IW = 14
//LB = 4: Size = 0.861328, Time = 1.38 msec, Performace = 1340.35 GFlop/s
//LB = 3: Size = 0.861328, Time = 1.708 msec, Performace = 1082.96 GFlop/s
//for[64, 64] -> [32, 32]: Size = 1.125, Time = 1.782 msec, Performace = 1355.73 GFlop/s
template<int LB, int STEP>
__global__ void uernel_SK_8_8_v2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	const int OW_N = OW * N;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int oh, ow, n; get_oh_ow_n(Y_k, ow, oh, n);
	int yoffset = ((n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	
	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n = X_k % N, X_oh = oh * sh, X_ow = ow * sw;
	int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
	float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
	if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
	Xs[buf][ty][tx] = xv;
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
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

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

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int oh, ow, n; get_oh_ow_n(Y_k, ow, oh, n);
		int yoffset = ((n*OH + oh)*OW + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n = X_k % N, X_oh = oh * sh, X_ow = ow * sw;
		int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
		float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
		if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
		Xs[buf][ty][tx] = xv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
	*(float4*)(deltaW + oc4) = v8;  *(float4*)(deltaW + oc4 + 4) = v9;
	*(float4*)(deltaW + oc5) = v10; *(float4*)(deltaW + oc5 + 4) = v11;
	*(float4*)(deltaW + oc6) = v12; *(float4*)(deltaW + oc6 + 4) = v13;
	*(float4*)(deltaW + oc7) = v14; *(float4*)(deltaW + oc7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef UERNEL_8_8_V3
#define UERNEL_8_8_V3

#define usk88_v3(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	uernel_SK_8_8_v3<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			  N, IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//synchronized: 
//GK_slice = 1024, OH = OW =  4, N = 256:
//LB = 4: Size = 1, Time = 1.602 msec, Performace = 1340.5 GFlop/s
//LB = 3: Size = 1, Time = 2.072 msec, Performace = 1036.43 GFlop/s
//GK_slice = 1024, OH = OW = 7, IH = IW = 14
//LB = 4: Size = 0.861328, Time = 1.38 msec, Performace = 1340.35 GFlop/s
//LB = 3: Size = 0.861328, Time = 1.708 msec, Performace = 1082.96 GFlop/s
//for[64, 64] -> [32, 32]: Size = 1.125, Time = 1.782 msec, Performace = 1355.73 GFlop/s
template<int LB, int STEP>
__global__ void uernel_SK_8_8_v3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	const int OW_N = OW * N;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int oh = Y_k / (OW_N); Y_k -= oh * OW_N;
	int ow = Y_k / N, n = Y_k - ow * N;
	int yoffset = ((n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_oh = oh * sh, X_ow = ow * sw;
	int X_n = X_k - oh * OW_N - ow * N;
	int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
	float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
	if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
	Xs[buf][ty][tx] = xv;
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

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int oh = Y_k / (OW_N); Y_k -= oh * OW_N;
		int ow = Y_k / N, n = Y_k - ow * N;
		int yoffset = ((n*OH + oh)*OW + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_oh = oh * sh, X_ow = ow * sw;
		int X_n = X_k - oh * OW_N - ow * N;
		int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
		float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
		if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
		Xs[buf][ty][tx] = xv;
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

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
	*(float4*)(deltaW + oc4) = v8;  *(float4*)(deltaW + oc4 + 4) = v9;
	*(float4*)(deltaW + oc5) = v10; *(float4*)(deltaW + oc5 + 4) = v11;
	*(float4*)(deltaW + oc6) = v12; *(float4*)(deltaW + oc6 + 4) = v13;
	*(float4*)(deltaW + oc7) = v14; *(float4*)(deltaW + oc7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef UERNEL_8_8_V4
#define UERNEL_8_8_V4

#define usk88_v4(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	uernel_SK_8_8_v4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			  N, IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//synchronized: 
//GK_slice = 1024, OH = OW =  4, N = 256:
//LB = 4: Size = 1, Time = 1.602 msec, Performace = 1340.5 GFlop/s
//LB = 3: Size = 1, Time = 2.072 msec, Performace = 1036.43 GFlop/s
//GK_slice = 1024, OH = OW = 7, IH = IW = 14
//LB = 4: Size = 0.861328, Time = 1.38 msec, Performace = 1340.35 GFlop/s
//LB = 3: Size = 0.861328, Time = 1.708 msec, Performace = 1082.96 GFlop/s
//for[64, 64] -> [32, 32]: Size = 1.125, Time = 1.782 msec, Performace = 1355.73 GFlop/s
template<int LB, int STEP>
__global__ void uernel_SK_8_8_v4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	const int OW_N = OW * N;
	int oh = Y_k / (OW_N), sub0 = oh * OW_N; Y_k -= sub0;
	int ow = Y_k / N, sub1 = ow * N, n = Y_k - sub1, sub = sub0 + sub1;
	int yoffset = ((n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n = X_k - sub, X_oh = oh * sh, X_ow = ow * sw;
	int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
	float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
	if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
	Xs[buf][ty][tx] = xv;
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

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int oh = Y_k / (OW_N), sub0 = oh * OW_N; Y_k -= sub0;
		int ow = Y_k / N, sub1 = ow * N, n = Y_k - sub1, sub = sub0 + sub1;
		int yoffset = ((n*OH + oh)*OW + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n = X_k - sub, X_oh = oh * sh, X_ow = ow * sw;
		int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
		float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
		if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
		Xs[buf][ty][tx] = xv;
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

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
	*(float4*)(deltaW + oc4) = v8;  *(float4*)(deltaW + oc4 + 4) = v9;
	*(float4*)(deltaW + oc5) = v10; *(float4*)(deltaW + oc5 + 4) = v11;
	*(float4*)(deltaW + oc6) = v12; *(float4*)(deltaW + oc6 + 4) = v13;
	*(float4*)(deltaW + oc7) = v14; *(float4*)(deltaW + oc7 + 4) = v15;
}

#endif


//N is power of 2
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef UERNEL_8_8_V5
#define UERNEL_8_8_V5

#define usk88_v5(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GM)\
	uernel_SK_8_8_v5<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			  LN, IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//synchronized: 
//GK_slice = 1024, OH = OW =  4, N = 256:
//LB = 4: Size = 1, Time = 1.602 msec, Performace = 1340.5 GFlop/s
//LB = 3: Size = 1, Time = 2.072 msec, Performace = 1036.43 GFlop/s
//GK_slice = 1024, OH = OW = 7, IH = IW = 14
//LB = 4: Size = 0.861328, Time = 1.38 msec, Performace = 1340.35 GFlop/s
//LB = 3: Size = 0.861328, Time = 1.708 msec, Performace = 1082.96 GFlop/s
//for[64, 64] -> [32, 32]: Size = 1.125, Time = 1.782 msec, Performace = 1355.73 GFlop/s
template<int LB, int STEP>
__global__ void uernel_SK_8_8_v5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	const int OW_N = OW << LN;
	const int N_m1 = (1 << LN) - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int oh = Y_k / (OW_N); Y_k -= oh * OW_N;
	int ow = Y_k >> LN, n = Y_k & N_m1;
	int yoffset = ((n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n = X_k & (N_m1), X_oh = oh * sh, X_ow = ow * sw;
	int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
	float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
	if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
	Xs[buf][ty][tx] = xv;
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
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

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int oh = Y_k / (OW_N); Y_k -= oh * OW_N;
		int ow = Y_k >> LN, n = Y_k & N_m1;
		int yoffset = ((n*OH + oh)*OW + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n = X_k & N_m1, X_oh = oh * sh, X_ow = ow * sw;
		int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
		float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
		if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
		Xs[buf][ty][tx] = xv;
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

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
	*(float4*)(deltaW + oc4) = v8;  *(float4*)(deltaW + oc4 + 4) = v9;
	*(float4*)(deltaW + oc5) = v10; *(float4*)(deltaW + oc5 + 4) = v11;
	*(float4*)(deltaW + oc6) = v12; *(float4*)(deltaW + oc6 + 4) = v13;
	*(float4*)(deltaW + oc7) = v14; *(float4*)(deltaW + oc7 + 4) = v15;
}

#endif


//N is power of 2
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef UERNEL_8_8_V6
#define UERNEL_8_8_V6

#define usk88_v6(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GM)\
	uernel_SK_8_8_v6<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			  LN, IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//synchronized: 
//GK_slice = 1024, OH = OW =  4, N = 256:
//LB = 4: Size = 1, Time = 1.602 msec, Performace = 1340.5 GFlop/s
//LB = 3: Size = 1, Time = 2.072 msec, Performace = 1036.43 GFlop/s
//GK_slice = 1024, OH = OW = 7, IH = IW = 14
//LB = 4: Size = 0.861328, Time = 1.38 msec, Performace = 1340.35 GFlop/s
//LB = 3: Size = 0.861328, Time = 1.708 msec, Performace = 1082.96 GFlop/s
//for[64, 64] -> [32, 32]: Size = 1.125, Time = 1.782 msec, Performace = 1355.73 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void uernel_SK_8_8_v6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	const int OW_N = OW << LN;
	const int N_m1 = (1 << LN) - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1) + GK_start;
	int oh = Y_k / (OW_N); Y_k -= oh * OW_N;
	int ow = Y_k >> LN, n = Y_k & N_m1;
	int yoffset = ((n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ((ty - ((ty >= STEP) << LB >> 1)) << 1) + GK_start;
	int X_n = X_k & (N_m1), X_oh = oh * sh, X_ow = ow * sw;
	int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
	float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
	if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
	Xs[buf][ty][tx] = xv;
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP2][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP2][tx];

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

		//[OH, OW, N]
		//oh0 = (ok*STEP +        Ux) / OW_N
		//oh1 = (ok*STEP + STEP + Ux) / OW_N
		//when N % (2*STEP) == 0
		//we have: oh0 = oh1

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1) + GK_start;
		int oh = Y_k / (OW_N); Y_k -= oh * OW_N;
		int ow = Y_k >> LN, n = Y_k & N_m1;
		int yoffset0 = ((n*OH + oh)*OW + ow)*OC;
		Ys[buf][(tx << 1)    ][ty] = *(float4*)(deltaY + yoffset);
		Ys[buf][(tx << 1) + 1][ty] = *(float4*)(deltaY + yoffset + OH*OW*OC);

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1) + GK_start;
		int X_n = X_k & N_m1, X_oh = oh * sh, X_ow = ow * sw;
		int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
		float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
		if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
		Xs[buf][ty][tx] = xv;
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

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
	*(float4*)(deltaW + oc4) = v8;  *(float4*)(deltaW + oc4 + 4) = v9;
	*(float4*)(deltaW + oc5) = v10; *(float4*)(deltaW + oc5 + 4) = v11;
	*(float4*)(deltaW + oc6) = v12; *(float4*)(deltaW + oc6 + 4) = v13;
	*(float4*)(deltaW + oc7) = v14; *(float4*)(deltaW + oc7 + 4) = v15;
}

#endif


//OW % 16 == 0
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef UERNEL_8_8_V7
#define UERNEL_8_8_V7

#define usk88_v7(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	uernel_SK_8_8_v7<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//synchronized: 
//LB = 4 / 1024:
//OH, OW =  4, N = 256: Size = 1, Time = 1.514 msec, Performace = 1418.42 GFlop/s
//OH, OW =  8, N =  64: Size = 1, Time = 1.498 msec, Performace = 1433.57 GFlop/s
//OH, OW = 16, N =  16: Size = 1, Time = 1.504 msec, Performace = 1427.85 GFlop/s
//LB = 4: Size = 1.125, Time = 1.692 msec, Performace = 1427.85 GFlop/s
//LB = 3: Size = 1.125, Time = 1.882 msec, Performace = 1283.7  GFlop/s
//for small feature, big channel
//LB = 4: Size = 1.125, Time = 1.708 msec, Performace = 1414.47 GFlop/s
//LB = 3: Size = 1.125, Time = 1.892 msec, Performace = 1276.91 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void uernel_SK_8_8_v7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	const int LOH_OW = LOH + LOW;
	const int OW_m1 = (1 << LOW) - 1;
	const int OH_OW_m1 = (1 << LOH_OW) - 1;

	int X_k = ((ty - ((ty >= STEP) << LB >> 1)) << 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	bool lx0 = LOAD_X2(tfh0, tfw0, X_oh, X_ow); 
	bool lx1 = LOAD_X2(tfh0, tfw0, X_oh, (X_ow + sw));
	Xs[buf][(ty << 1)    ][tx] = (lx0 ? *(float4*)(X + xoffset     ) : F32_4_0);
	Xs[buf][(ty << 1) + 1][tx] = (lx1 ? *(float4*)(X + xoffset + (IC*sw)) : F32_4_0);

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1) + GK_start;//(n, oh, ow), ow + 1
	Ys[buf][(tx << 1)    ][ty] = *(float4*)(deltaY + Y_k * OC);
	Ys[buf][(tx << 1) + 1][ty] = *(float4*)(deltaY + Y_k * OC + OC);
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

	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP2][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP2][tx];

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
		int X_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1) + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		bool lx0 = LOAD_X2(tfh0, tfw0, X_oh, X_ow);
		bool lx1 = LOAD_X2(tfh0, tfw0, X_oh, (X_ow + sw));
		Xs[buf][(ty << 1)    ][tx] = (lx0 ? *(float4*)(X + xoffset         ) : F32_4_0);
		Xs[buf][(ty << 1) + 1][tx] = (lx1 ? *(float4*)(X + xoffset + (IC*sw)) : F32_4_0);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1) + GK_start;
		Ys[buf][(tx << 1)    ][ty] = *(float4*)(deltaY + Y_k * OC);
		Ys[buf][(tx << 1) + 1][ty] = *(float4*)(deltaY + Y_k * OC + OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP2][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP2][tx];

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
	*(float4*)(deltaW + oc4) = v8;  *(float4*)(deltaW + oc4 + 4) = v9;
	*(float4*)(deltaW + oc5) = v10; *(float4*)(deltaW + oc5 + 4) = v11;
	*(float4*)(deltaW + oc6) = v12; *(float4*)(deltaW + oc6 + 4) = v13;
	*(float4*)(deltaW + oc7) = v14; *(float4*)(deltaW + oc7 + 4) = v15;
}

#endif