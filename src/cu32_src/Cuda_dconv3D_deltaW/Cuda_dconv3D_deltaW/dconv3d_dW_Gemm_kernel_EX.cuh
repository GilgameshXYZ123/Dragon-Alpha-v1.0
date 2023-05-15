#pragma once

#ifndef DECONV3D_DW_GEMM_KERNEL_EX_H
#define DECONV3D_DW_GEMM_KERNEL_EX_H


#ifndef DECONV3D_DW_GEMM_KERNEL_EX_CALL
#define DECONV3D_DW_GEMM_KERNEL_EX_CALL

//LB = log2(BLOCK_SIZE)

//======[OH, OW is power of 2]==========================================
#define kGemm88_ohw2pow(stream, LB, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_8_8_OHW2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm84_ohw2pow(stream, LB, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_8_4_OHW2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm48_ohw2pow(stream, LB, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_4_8_OHW2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm44_ohw2pow(stream, LB, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_4_4_OHW2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm82_ohw2pow(stream, LB, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_8_2_OHW2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm28_ohw2pow(stream, LB, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_2_8_OHW2pow<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#endif


//======[OH, OW is power of 2]==========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: IC % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_8_8_OHW_2POW
#define DECONV3D_DW_GEMM_KERNEL_8_8_OHW_2POW

//LB = 4: Size = 0.5625, Time = 0.886 msec, Performace = 1363.39 GFlop/s
//LB = 3: Size = 0.5625, Time = 0.978 msec, Performace = 1235.13 GFlop/s
//LB = 4: Size = 1, Time = 1.618 msec, Performace = 1327.25 GFlop/s
//LB = 3: Size = 1, Time = 1.828 msec, Performace = 1174.77 GFlop/s
//for small size, big channel
//LB = 4: Size = 1.125, Time = 1.574 msec, Performace = 1534.89 GFlop/s
//LB = 3: Size = 1.125, Time = 1.794 msec, Performace = 1346.67 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_8_8_OHW2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC, 
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//prepare for GK = N * OH * OW
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK = N << LOH_OW;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k*OC);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	int X_n, X_oh, X_ow; get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0),  v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0),  v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0),  v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0),  v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0),  v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4( v0, a0.x, b0); simdMM4( v1, a0.x, b1);
			simdMM4( v2, a0.y, b0); simdMM4( v3, a0.y, b1);
			simdMM4( v4, a0.z, b0); simdMM4( v5, a0.z, b1);
			simdMM4( v6, a0.w, b0); simdMM4( v7, a0.w, b1);
			simdMM4( v8, a1.x, b0); simdMM4( v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int X_n, X_oh, X_ow; get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4( v0, a0.x, b0); simdMM4( v1, a0.x, b1);
		simdMM4( v2, a0.y, b0); simdMM4( v3, a0.y, b1);
		simdMM4( v4, a0.z, b0); simdMM4( v5, a0.z, b1);
		simdMM4( v6, a0.w, b0); simdMM4( v7, a0.w, b1);
		simdMM4( v8, a1.x, b0); simdMM4( v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int Wstride = FH * FW_IC;
	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	const int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	const int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	const int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: IC % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_8_4_OHW_2POW
#define DECONV3D_DW_GEMM_KERNEL_8_4_OHW_2POW

//LB = 4: Size = 1, Time = 1.822 msec, Performace = 1178.64 GFlop/s
//LB = 3: Size = 1, Time = 1.908 msec, Performace = 1125.52 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_8_4_OHW2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;//X += X0

	//prepare for GK = N * OH * OW
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK = N << LOH_OW;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);

			simdMM4(v0, a0.x, b0);
			simdMM4(v2, a0.y, b0);
			simdMM4(v4, a0.z, b0);
			simdMM4(v6, a0.w, b0);
			simdMM4(v8, a1.x, b0);
			simdMM4(v10, a1.y, b0);
			simdMM4(v12, a1.z, b0);
			simdMM4(v14, a1.w, b0);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);

		simdMM4(v0, a0.x, b0);
		simdMM4(v2, a0.y, b0);
		simdMM4(v4, a0.z, b0);
		simdMM4(v6, a0.w, b0);
		simdMM4(v8, a1.x, b0);
		simdMM4(v10, a1.y, b0);
		simdMM4(v12, a1.z, b0);
		simdMM4(v14, a1.w, b0);
	}

	const int FH_FW_IC = FH * FW_IC;
	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(deltaW + oc0) = v0;
	*(float4*)(deltaW + oc1) = v2;
	*(float4*)(deltaW + oc2) = v4;
	*(float4*)(deltaW + oc3) = v6;
	*(float4*)(deltaW + oc4) = v8;
	*(float4*)(deltaW + oc5) = v10;
	*(float4*)(deltaW + oc6) = v12;
	*(float4*)(deltaW + oc7) = v14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: IC % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_4_8_OHW_2POW
#define DECONV3D_DW_GEMM_KERNEL_4_8_OHW_2POW

//LB = 4: Size = 1, Time = 1.964 msec, Performace = 1093.42  GFlop/s
//LB = 3: Size = 1, Time = 2.198 msec, Performace =  977.017 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_4_8_OHW2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//followed k88

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += ((tx & 1) << 1) + oc0;

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//prepare for GK = N * OH * OW
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK = N << LOH_OW;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k * OC);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&Ys[buf][ik][ty << 1]);
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k * OC);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
	}

	const int FH_FW_IC = FH * FW_IC;
	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
}

#endif

   
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: IC % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_4_4_OHW_2POW
#define DECONV3D_DW_GEMM_KERNEL_4_4_OHW_2POW

//LB = 4: Size = 1, Time = 2.342 msec, Performace = 916.944 GFlop/s
//LB = 3: Size = 1, Time = 2.68  msec, Performace = 801.3   GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_4_4_OHW2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW, int FH, int FW, 
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += ((tx & 1) << 1) + oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//prepare for GK = N * OH * OW
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK = N << LOH_OW;

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k * OC);

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);

			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k * OC);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}

	const int FH_FW_IC = FH * FW_IC;
	oc0 = oc0 * FH_FW_IC + j0; //j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;
	
	*(float4*)(deltaW + oc0) = v0;
	*(float4*)(deltaW + oc1) = v1;
	*(float4*)(deltaW + oc2) = v2;
	*(float4*)(deltaW + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_8_2_OHW_2POW
#define DECONV3D_DW_GEMM_KERNEL_8_2_OHW_2POW

//LB = 4: Size = 1.125, Time = 2.428 msec, Performace = 995.024 GFlop/s
//LB = 3: Size = 1.125, Time = 2.828 msec, Performace = 854.285 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_8_2_OHW2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	const int tj0 = j0 + (ty & 1);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;//X += X0

	//prepare for GK = N * OH * OW
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK = N << LOH_OW;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

	//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? X[xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2  v0 = make_float2(0, 0);
	float2  v2 = make_float2(0, 0);
	float2  v4 = make_float2(0, 0);
	float2  v6 = make_float2(0, 0);
	float2  v8 = make_float2(0, 0);
	float2 v10 = make_float2(0, 0);
	float2 v12 = make_float2(0, 0);
	float2 v14 = make_float2(0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float2 b0 = *(float2*)(&Xs[buf][ik][tx << 1]);

			simdMM2(v0, a0.x, b0);
			simdMM2(v2, a0.y, b0);
			simdMM2(v4, a0.z, b0);
			simdMM2(v6, a0.w, b0);
			simdMM2(v8, a1.x, b0);
			simdMM2(v10, a1.y, b0);
			simdMM2(v12, a1.z, b0);
			simdMM2(v14, a1.w, b0);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? X[xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float2 b0 = *(float2*)(&Xs[buf][ik][tx << 1]);

		simdMM2(v0, a0.x, b0);
		simdMM2(v2, a0.y, b0);
		simdMM2(v4, a0.z, b0);
		simdMM2(v6, a0.w, b0);
		simdMM2(v8, a1.x, b0);
		simdMM2(v10, a1.y, b0);
		simdMM2(v12, a1.z, b0);
		simdMM2(v14, a1.w, b0);
	}

	const int Wstride = FH * FW_IC;
	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	*(float2*)(deltaW + oc0) = v0;
	*(float2*)(deltaW + oc1) = v2;
	*(float2*)(deltaW + oc2) = v4;
	*(float2*)(deltaW + oc3) = v6;
	*(float2*)(deltaW + oc4) = v8;
	*(float2*)(deltaW + oc5) = v10;
	*(float2*)(deltaW + oc6) = v12;
	*(float2*)(deltaW + oc7) = v14;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_2_8_OHW_2POW
#define DECONV3D_DW_GEMM_KERNEL_2_8_OHW_2POW

//LB = 4: Size = 1.125, Time = 3.246 msec, Performace = 744.276 GFlop/s
//LB = 3: Size = 1.125, Time = 3.48   msec, Performace = 694.23 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_2_8_OHW2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//followed k88

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += (tx & 1) + oc0;

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//prepare for GK = N * OH * OW
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK = N << LOH_OW;

	//load 1 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = deltaY[Y_k * OC];

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 a0 = *(float2*)(&Ys[buf][ik][ty << 1]);
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][Ys_x][Ys_y] = deltaY[Y_k * OC];

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 a0 = *(float2*)(&Ys[buf][ik][ty << 1]);
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
	}

	const int Wstride = FH * FW_IC;
	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
}

#endif

#endif