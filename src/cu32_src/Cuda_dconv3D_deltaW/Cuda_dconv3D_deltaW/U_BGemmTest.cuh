

#define kNGemm88(stream, LB, LNslice, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_8_8<LB, (1<<LB>>1), LNslice>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, (N+(1<<LNslice)-1)>>LNslice),\
			dim3(1<<LB, 1<<LB), 0, stream >>>\
				(X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef,\
					N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define bgemm_v1(stream, LB, LGZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_v1<LB, LGZ>\
		<<< dim3(GM>>LB, GN>>LB, (1<<LGZ)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//Size = 0.480652, Time = 42.06 msec, Performace = 24.5409 GFlop/s
template<int LB, int LGZ>
__global__ void kernel_BGemm_v1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
		  float* __restrict__ deltaW, int FH, int FW,
	float mean_coef,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	//prepare for GN = OC
	const int oc = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = FH*FW*IC
	const int j = ((blockIdx.x << LB) + tx) + j_index;
	const int FW_IC = FW * IC; 
	get_fh_fw_ic(j, fh, fw, ic)
	
	//prepare for GK slice
	const int OH_OW = OH * OW, GK = N * OH_OW;
	const int GK_slice = GK >> LGZ;
	int bz = blockIdx.z;
	int GK_start = bz * GK_slice;
	int GK_end = (bz + 1) * GK_slice;
	if (bz == ((1 << LGZ) - 1)) GK_end = GK;

	float v = 0;
	for (int k = GK_start; k < GK_end; k++) 
	{
		int n = k / OH_OW, kr = k - n * OH_OW;
		int oh = kr / OW, ow = kr - oh * OW;

		int ih = fh - oph + (oh*sh);
		int iw = fw - opw + (ow*sw);
		if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

		float dy = get4d(deltaY, n, oh, ow, oc, OH, OW, OC);
		float x = get4d(X, n, ih, iw, ic, IH, IW, IC);
		v += dy * x;
	}

	int Woffset = ((oc*FH + fh)*FW + fw)*IC + ic;
	atomicAdd(deltaW + Woffset, mean_coef * v);
}



#define bgemm_v2(stream, LB, LGZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_v2<LB, LGZ>\
		<<< dim3(GM>>LB, GN>>LB, (1<<LGZ)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

template<int LB, int LGZ>
__global__ void kernel_BGemm_v2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW,
	float mean_coef,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	//prepare for GN = OC
	const int oc = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = FH*FW*IC
	const int j = ((blockIdx.x << LB) + tx) + j_index;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j, fh, fw, ic);

	//prepare for GK slice
	
	int bz = blockIdx.z;
	int N_slice = N >> LGZ;
	int N_start = bz * N_slice, N_end = N_start + N_slice;//[N_start, N_end)
	if (bz == ((1 << LGZ) - 1)) N_end = N;
	X += N_start * IH*IW*IC;
	deltaY += N_start * OH*OW*OC;

	const int OH_OW = OH * OW;//kernel: deltaY[N, OH, OW, OC]
	const int GK_slice = (N_end - N_start) * OH_OW;

	float v = 0;
	for (int k = 0; k < GK_slice; k++)
	{
		int n = k / OH_OW, kr = k - n * OH_OW;
		int oh = kr / OW, ow = kr - oh * OW;

		int ih = fh - oph + (oh*sh);
		int iw = fw - opw + (ow*sw);
		if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

		float dy = get4d(deltaY, n, oh, ow, oc, OH, OW, OC);
		float x = get4d(X, n, ih, iw, ic, IH, IW, IC);
		v += dy * x;
	}

	int Woffset = ((oc*FH + fh)*FW + fw)*IC + ic;
	atomicAdd(deltaW + Woffset, mean_coef * v);
}


#define bgemm_v3(stream, LB, LGZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_v3<LB, LGZ>\
		<<< dim3(GM>>LB, GN>>LB, (1<<LGZ)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

template<int LB, int LGZ>
__global__ void kernel_BGemm_v3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW,
	float mean_coef,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	//prepare for GN = OC
	const int oc = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = FH*FW*IC
	const int j = ((blockIdx.x << LB) + tx) + j_index;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j, fh, fw, ic);

	//prepare for GK_slice
	int bz = blockIdx.z;
	int N_slice = N >> LGZ;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;
	const int OH_OW = OH * OW;
	const int GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; deltaY += N_start * OH_OW*OC;//X[N_start,...], deltaY[N_start,...]

	float v = 0;
	for (int k = 0; k < GK_slice; k++)
	{
		int n = k / OH_OW, kr = k - n * OH_OW;
		int oh = kr / OW, ow = kr - oh * OW;

		int ih = fh - oph + (oh*sh);
		int iw = fw - opw + (ow*sw);
		if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

		float dy = get4d(deltaY, n, oh, ow, oc, OH, OW, OC);
		float x = get4d(X, n, ih, iw, ic, IH, IW, IC);
		v += dy * x;
	}

	int Woffset = ((oc*FH + fh)*FW + fw)*IC + ic;
	atomicAdd(deltaW + Woffset, mean_coef * v);
}


#define bgemm_v4(stream, LB, LGZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_v4<LB, (1<<LB), (1<<LGZ)>\
		<<< dim3(GM>>LB, GN>>LB, (1<<LGZ)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//Size = 0.480652, Time = 2.135 msec, Performace = 483.462 GFlop/s
template<int LB, int STEP, int LGZ>
__global__ void kernel_BGemm_v4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW,
	float mean_coef,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;

	//prepare for GK_slice
	int bz = blockIdx.z;
	int N_slice = N >> LGZ;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;
	const int OH_OW = OH * OW;
	const int GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC;// deltaY[N_start,...]

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

	//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = lx0 ? X[Xoffset0 + xoffset] : 0;
	__syncthreads();

	//compute area-----------------------------------------------------
	float v = 0;
	for (int ok = 1, OK = GK_slice >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float b = Xs[buf][ik][tx];
			float a = dYs[buf][ik][ty];
			v += a * b;
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		Xs[buf][ty][tx] = lx0 ? X[Xoffset0 + xoffset] : 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float b = Xs[buf][ik][tx];
		float a = dYs[buf][ik][ty];
		v += a * b;
	}

	//when GK_slice%STEP != 0 -------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		float a = deltaY[k * OC + oc0];//load 1 element from deltaY

		int X_k = k;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int ih0 = fh0 + X_oh, iw0 = fw0 + X_ow;
		float b;//load 1 element from X
		load4d_S(b, X, X_n, ih0, iw0, ic0, IH, IW, IC);

		v += a * b;
	}
	//when GK_slice%STEP != 0 -------------------------------------------

	int Woffset = oc0 * (FH * FW_IC) + j0;//j = *fh * FW + fw)*IC + ic
	atomicAdd(deltaW + Woffset, mean_coef * v);
}


#define bgemm_v5(stream, LB, LGZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_v5<LB, (1<<LB), LGZ>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1, (1<<LGZ)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef,\
				N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

template<int LB, int STEP, int LGZ>
__global__ void kernel_BGemm_v5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
		  float* __restrict__ deltaW, int FH, int FW, 
	float mean_coef,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float2  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	get_fh_fw_ic(j1, fh1, fw1, ic1); fh1 -= oph; fw1 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;
	const int Xoffset1 = (fh1*IW + fw1)*IC + ic1;

	//prepare for GK_slice
	int bz = blockIdx.z;
	int N_slice = N >> LGZ;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;
	const int OH_OW = OH * OW;
	const int GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx].x = lx0 ? X[Xoffset0 + xoffset] : 0;
	Xs[buf][ty][tx].y = lx1 ? X[Xoffset1 + xoffset] : 0;
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = GK_slice >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float2 a = dYs[buf][ik][ty];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok << LB) + tx);
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
		int X_k = (ok << LB) + ty;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx].x = lx0 ? X[Xoffset0 + xoffset] : 0;
		Xs[buf][ty][tx].y = lx1 ? X[Xoffset1 + xoffset] : 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Xs[buf][ik][tx];
		float2 a = dYs[buf][ik][ty];
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}

	//when GK%STEP != 0 -------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		float2 a = *(float2*)(&deltaY[k * OC + oc0]);//load 2 elements from deltaY

		float2 b;//load 2 elements from X
		int X_k = k, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b.x = lx0 ? X[Xoffset0 + xoffset] : 0;
		b.y = lx1 ? X[Xoffset1 + xoffset] : 0;

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK%STEP != 0 -------------------------------------------
	
	const int FH_FW_IC = FH * FW_IC;
	oc0 = oc0 * FH_FW_IC + j0; int oc1 = oc0 + FH_FW_IC;

	atomicAdd(deltaW + oc0, v0.x * mean_coef); atomicAdd(deltaW + oc0 + 1, v0.y * mean_coef);
	atomicAdd(deltaW + oc1, v1.x * mean_coef); atomicAdd(deltaW + oc1 + 1, v1.y * mean_coef);
}



#define bgemm_v6(stream, LB, LGZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_v6<LB, (1<<LB>>1), LGZ>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, (1<<LGZ)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW_buf, FH, FW, mean_coef,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//LB = 4: Size = 0.5, Time = 1.23 msec, Performace = 872.961 GFlop/s
//LB=4: Size = 1, Time = 2.098 msec, Performace = 1023.59 GFlop/s
//LB=3: Size = 1, Time = 2.638 msec, Performace = 814.057 GFlop/s
template<int LB, int STEP, int LGZ>
__global__ void kernel_BGemm_v6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW,
	float mean_coef,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 dYs[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4  Xs[2][1 << LB >> 1][(2 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	const int toc0 = ((tx & 1) << 2) + oc0;//(oc4 - oc0)*(tx&1) + oc0

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0);
	get_fh_fw_ic(j1, fh1, fw1, ic1);
	get_fh_fw_ic(j2, fh2, fw2, ic2);
	get_fh_fw_ic(j3, fh3, fw3, ic3);
	get_fh_fw_ic(j4, fh4, fw4, ic4);
	get_fh_fw_ic(j5, fh5, fw5, ic5);
	get_fh_fw_ic(j6, fh6, fw6, ic6);
	get_fh_fw_ic(j7, fh7, fw7, ic7);
	bool flagY = (ty & 1);
	const int tic0 = (ic4 - ic0)*flagY + ic0;
	const int tic1 = (ic5 - ic1)*flagY + ic1;
	const int tic2 = (ic6 - ic2)*flagY + ic2;
	const int tic3 = (ic7 - ic3)*flagY + ic3;
	const int tfh0 = (fh4 - fh0)*flagY + fh0 - oph;
	const int tfh1 = (fh5 - fh1)*flagY + fh1 - oph;
	const int tfh2 = (fh6 - fh2)*flagY + fh2 - oph;
	const int tfh3 = (fh7 - fh3)*flagY + fh3 - oph;
	const int tfw0 = (fw4 - fw0)*flagY + fw0 - opw;
	const int tfw1 = (fw5 - fw1)*flagY + fw1 - opw;
	const int tfw2 = (fw6 - fw2)*flagY + fw2 - opw;
	const int tfw3 = (fw7 - fw3)*flagY + fw3 - opw;
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1;
	const int Xoffset2 = (tfh2*IW + tfw2)*IC + tic2;
	const int Xoffset3 = (tfh3*IW + tfw3)*IC + tic3;

	//prepare for GK_slice
	int bz = blockIdx.z;
	int N_slice = N >> LGZ;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;
	const int OH_OW = OH * OW;
	const int GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float4*)(&deltaY[dY_k*OC + toc0]);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
	bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
	bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x].x = lx0 ? X[Xoffset0 + xoffset] : 0;
	Xs[buf][Xs_y][Xs_x].y = lx1 ? X[Xoffset1 + xoffset] : 0;
	Xs[buf][Xs_y][Xs_x].z = lx2 ? X[Xoffset2 + xoffset] : 0;
	Xs[buf][Xs_y][Xs_x].w = lx3 ? X[Xoffset3 + xoffset] : 0;
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
	for (int ok = 1, OK = GK_slice << 1 >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = dYs[buf][ik][(ty << 1)], a1 = dYs[buf][ik][(ty << 1) + 1];
			float4 b0 = Xs[buf][ik][(tx << 1)], b1 = Xs[buf][ik][(tx << 1) + 1];

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
		int dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = *(float4*)(&deltaY[dY_k*OC + toc0]);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
		bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
		bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x].x = lx0 ? X[Xoffset0 + xoffset] : 0;
		Xs[buf][Xs_y][Xs_x].y = lx1 ? X[Xoffset1 + xoffset] : 0;
		Xs[buf][Xs_y][Xs_x].z = lx2 ? X[Xoffset2 + xoffset] : 0;
		Xs[buf][Xs_y][Xs_x].w = lx3 ? X[Xoffset3 + xoffset] : 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = dYs[buf][ik][(ty << 1)], a1 = dYs[buf][ik][(ty << 1) + 1];
		float4 b0 = Xs[buf][ik][(tx << 1)], b1 = Xs[buf][ik][(tx << 1) + 1];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int FH_FW_IC = FH * FW_IC; oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	Mul4( v0, mean_coef,  v0); simdAtomicAdd4(&deltaW[oc0],  v0); Mul4( v1, mean_coef,  v1); simdAtomicAdd4(&deltaW[oc0 + 4],  v1);
	Mul4( v2, mean_coef,  v2); simdAtomicAdd4(&deltaW[oc1],  v2); Mul4( v3, mean_coef,  v3); simdAtomicAdd4(&deltaW[oc1 + 4],  v3);
	Mul4( v4, mean_coef,  v4); simdAtomicAdd4(&deltaW[oc2],  v4); Mul4( v5, mean_coef,  v5); simdAtomicAdd4(&deltaW[oc2 + 4],  v5);
	Mul4( v6, mean_coef,  v6); simdAtomicAdd4(&deltaW[oc3],  v6); Mul4( v7, mean_coef,  v7); simdAtomicAdd4(&deltaW[oc3 + 4],  v7);
	Mul4( v8, mean_coef,  v8); simdAtomicAdd4(&deltaW[oc4],  v8); Mul4( v9, mean_coef,  v9); simdAtomicAdd4(&deltaW[oc4 + 4],  v9);
	Mul4(v10, mean_coef, v10); simdAtomicAdd4(&deltaW[oc5], v10); Mul4(v11, mean_coef, v11); simdAtomicAdd4(&deltaW[oc5 + 4], v11);
	Mul4(v12, mean_coef, v12); simdAtomicAdd4(&deltaW[oc6], v12); Mul4(v13, mean_coef, v13); simdAtomicAdd4(&deltaW[oc6 + 4], v13);
	Mul4(v14, mean_coef, v14); simdAtomicAdd4(&deltaW[oc7], v14); Mul4(v15, mean_coef, v15); simdAtomicAdd4(&deltaW[oc7 + 4], v15);
}



#define bgemm_v7(stream, LB, LGZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, mean_coef, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_v7<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, (1<<LGZ)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, mean_coef,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//v6: LB = 4: Size = 0.5, Time = 1.23 msec, Performace = 872.961 GFlop/s
//LB=4: Size = 1, Time = 2.098 msec, Performace = 1023.59 GFlop/s
//LB=3: Size = 1, Time = 2.638 msec, Performace = 814.057 GFlop/s


//deltaW_buf[gridZ, OC, FH, FW, IC]
template<int LB, int STEP>
__global__ void kernel_BGemm_v7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	float mean_coef,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 dYs[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4  Xs[2][1 << LB >> 1][(2 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	const int toc0 = ((tx & 1) << 2) + oc0;//(oc4 - oc0)*(tx&1) + oc0

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0);
	get_fh_fw_ic(j1, fh1, fw1, ic1);
	get_fh_fw_ic(j2, fh2, fw2, ic2);
	get_fh_fw_ic(j3, fh3, fw3, ic3);
	get_fh_fw_ic(j4, fh4, fw4, ic4);
	get_fh_fw_ic(j5, fh5, fw5, ic5);
	get_fh_fw_ic(j6, fh6, fw6, ic6);
	get_fh_fw_ic(j7, fh7, fw7, ic7);
	bool flagY = (ty & 1);
	const int tic0 = (ic4 - ic0)*flagY + ic0;
	const int tic1 = (ic5 - ic1)*flagY + ic1;
	const int tic2 = (ic6 - ic2)*flagY + ic2;
	const int tic3 = (ic7 - ic3)*flagY + ic3;
	const int tfh0 = (fh4 - fh0)*flagY + fh0 - oph;
	const int tfh1 = (fh5 - fh1)*flagY + fh1 - oph;
	const int tfh2 = (fh6 - fh2)*flagY + fh2 - oph;
	const int tfh3 = (fh7 - fh3)*flagY + fh3 - oph;
	const int tfw0 = (fw4 - fw0)*flagY + fw0 - opw;
	const int tfw1 = (fw5 - fw1)*flagY + fw1 - opw;
	const int tfw2 = (fw6 - fw2)*flagY + fw2 - opw;
	const int tfw3 = (fw7 - fw3)*flagY + fw3 - opw;
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1;
	const int Xoffset2 = (tfh2*IW + tfw2)*IC + tic2;
	const int Xoffset3 = (tfh3*IW + tfw3)*IC + tic3;

	//prepare for GK_slice
	int bz = blockIdx.z;
	int N_slice = N / gridDim.z;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;
	const int OH_OW = OH * OW;
	const int GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float4*)(&deltaY[dY_k*OC + toc0]);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
	bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
	bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x].x = lx0 ? X[Xoffset0 + xoffset] : 0;
	Xs[buf][Xs_y][Xs_x].y = lx1 ? X[Xoffset1 + xoffset] : 0;
	Xs[buf][Xs_y][Xs_x].z = lx2 ? X[Xoffset2 + xoffset] : 0;
	Xs[buf][Xs_y][Xs_x].w = lx3 ? X[Xoffset3 + xoffset] : 0;
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
	for (int ok = 1, OK = GK_slice << 1 >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = dYs[buf][ik][(ty << 1)], a1 = dYs[buf][ik][(ty << 1) + 1];
			float4 b0 = Xs[buf][ik][(tx << 1)], b1 = Xs[buf][ik][(tx << 1) + 1];

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
		int dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = *(float4*)(&deltaY[dY_k*OC + toc0]);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
		bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
		bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x].x = lx0 ? X[Xoffset0 + xoffset] : 0;
		Xs[buf][Xs_y][Xs_x].y = lx1 ? X[Xoffset1 + xoffset] : 0;
		Xs[buf][Xs_y][Xs_x].z = lx2 ? X[Xoffset2 + xoffset] : 0;
		Xs[buf][Xs_y][Xs_x].w = lx3 ? X[Xoffset3 + xoffset] : 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = dYs[buf][ik][(ty << 1)], a1 = dYs[buf][ik][(ty << 1) + 1];
		float4 b0 = Xs[buf][ik][(tx << 1)], b1 = Xs[buf][ik][(tx << 1) + 1];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + (bz - 1)* OC *FH*FW*IC;//deltaW_buf[bz - 1, sizeW], bz >= 1
	float *dst = (blockIdx.z != 0) * (buf_addr - deltaW) + deltaW;
	
	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	Mul4(v0, mean_coef, v0); *(float4*)(&dst[oc0]) = v0; Mul4(v1, mean_coef, v1); *(float4*)(&dst[oc0 + 4]) = v1;
	Mul4(v2, mean_coef, v2); *(float4*)(&dst[oc1]) = v2; Mul4(v3, mean_coef, v3); *(float4*)(&dst[oc1 + 4]) = v3;
	Mul4(v4, mean_coef, v4); *(float4*)(&dst[oc2]) = v4; Mul4(v5, mean_coef, v5); *(float4*)(&dst[oc2 + 4]) = v5;
	Mul4(v6, mean_coef, v6); *(float4*)(&dst[oc3]) = v6; Mul4(v7, mean_coef, v7); *(float4*)(&dst[oc3 + 4]) = v7;
	Mul4(v8, mean_coef, v8); *(float4*)(&dst[oc4]) = v8; Mul4(v9, mean_coef, v9); *(float4*)(&dst[oc4 + 4]) = v9;

	Mul4(v10, mean_coef, v10); *(float4*)(&dst[oc5]) = v10; Mul4(v11, mean_coef, v11); *(float4*)(&dst[oc5 + 4]) = v11;
	Mul4(v12, mean_coef, v12); *(float4*)(&dst[oc6]) = v12; Mul4(v13, mean_coef, v13); *(float4*)(&dst[oc6 + 4]) = v13;
	Mul4(v14, mean_coef, v14); *(float4*)(&dst[oc7]) = v14; Mul4(v15, mean_coef, v15); *(float4*)(&dst[oc7 + 4]) = v15;
}

