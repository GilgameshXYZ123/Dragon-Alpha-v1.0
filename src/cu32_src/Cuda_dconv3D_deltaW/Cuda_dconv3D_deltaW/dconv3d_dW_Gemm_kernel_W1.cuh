#pragma once

#ifndef DECONV3D_DW_GEMM_KERNEL_W1_H
#define DECONV3D_DW_GEMM_KERNEL_W1_H

//We have:
//	(1) FH * FW >= 2
//	(2) GN = OC:                GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW = IC: GM%4 == 0, GM >= 4
//	(4) GK = N  * OH * OW     : GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
#ifndef DECONV3D_DW_GEMM_KERNEL_W1_CALL
#define DECONV3D_DW_GEMM_KERNEL_W1_CALL

#define k44W1(stream, LB, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM)\
	kernel_4_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, N, IC, OC, oc_index, j_index)

#define k42W1(stream, LB, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM)\
	kernel_4_2_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, N, IC, OC, oc_index, j_index)

#define k24W1(stream, LB, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM)\
	kernel_2_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, N, IC, OC, oc_index, j_index)

#define k22W1(stream, LB, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM)\
	kernel_2_2_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, N, IC, OC, oc_index, j_index)

#define k21W1(stream, LB, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM)\
	kernel_2_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, N, IC, OC, oc_index, j_index)

#define k12W1(stream, LB, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM)\
	kernel_1_2_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, N, IC, OC, oc_index, j_index)

#define k11W1(stream, LB, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM)\
	kernel_1_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, N, IC, OC, oc_index, j_index)

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_4_4_W1
#define DECONV3D_DW_GEMM_KERNEL_4_4_W1

//LB = 4: Size = 0.469238, Time = 1.764 msec, Performace = 571.248 GFlop/s
//LB = 3: Size = 0.469238, Time = 3.358 msec, Performace = 300.084 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_4_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, 
		  float* __restrict__ deltaW,
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int toc0 = ((tx & 1) << 1) + oc0;

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tic0 = ((ty & 1) << 1) + ic0;

	//prepare for GK = N * OH * OW
	const int OH_OW = IH * IW, GK = N * OH_OW;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1) ;//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(&X[X_k*IC + tic0]);
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
			float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);

			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(&X[X_k*IC + tic0]);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC, oc2 = oc1 + IC, oc3 = oc2 + IC;

	*(float4*)(deltaW + oc0) = v0;
	*(float4*)(deltaW + oc1) = v1;
	*(float4*)(deltaW + oc2) = v2;
	*(float4*)(deltaW + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_4_2_W1
#define DECONV3D_DW_GEMM_KERNEL_4_2_W1

//LB = 4: Size = 0.469238, Time = 1.77 msec, Performace = 569.312 GFlop/s
//LB = 3: Size = 0.469238, Time = 3.246 msec, Performace = 310.438 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, 
	float* __restrict__ deltaW,
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float   Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int toc0 = ((tx & 1) << 1) + oc0;

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	const int tic0 = (ty & 1) + ic0;

	//prepare for GK = N * OH * OW
	const int OH_OW = IH * IW, GK = N * OH_OW;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

	//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = X[X_k*IC + tic0];
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);
			float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);

			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
			simdMM2(v2, a.z, b);
			simdMM2(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

		//load 1 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = X[X_k*IC + tic0];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);
		float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
		simdMM2(v2, a.z, b);
		simdMM2(v3, a.w, b);
	}

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC, oc2 = oc1 + IC, oc3 = oc2 + IC;

	*(float2*)(deltaW + oc0) = v0;
	*(float2*)(deltaW + oc1) = v1;
	*(float2*)(deltaW + oc2) = v2;
	*(float2*)(deltaW + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_2_4_W1
#define DECONV3D_DW_GEMM_KERNEL_2_4_W1

//LB = 4: Size = 0.469238, Time = 2.14  msec, Performace = 470.879 GFlop/s
//LB = 3: Size = 0.469238, Time = 2.872 msec, Performace = 350.864 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_4_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
		  float* __restrict__ deltaW,
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int toc0 = (tx & 1) + oc0;

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tic0 = ((ty & 1) << 1) + ic0;

	//prepare for GK = N * OH * OW
	const int OH_OW = IH * IW, GK = N * OH_OW;

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = deltaY[dY_k*OC + toc0];

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(&X[X_k*IC + tic0]);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = *(float2*)(&dYs[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = deltaY[dY_k*OC + toc0];

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(&X[X_k*IC + tic0]);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = *(float2*)(&dYs[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
	}

	oc0 = oc0 * IC + ic0; 
	int oc1 = oc0 + IC;

	*(float4*)(deltaW + oc0) = v0;
	*(float4*)(deltaW + oc1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
#ifndef DECONV3D_DW_GEMM_KERNEL_2_2_W1
#define DECONV3D_DW_GEMM_KERNEL_2_2_W1

//LB=4: Size = 0.469238, Time = 2.246 msec, Performace = 448.656 GFlop/s
//LB=3: Size = 0.469238, Time = 3.226 msec, Performace = 312.363 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
		  float* __restrict__ deltaW,
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float2  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + j_index;

	//prepare for GK = N * OH * OW
	const int OH_OW = IH * IW, GK = N * OH_OW;
		
	int OK = (GK >> LB);
	if (OK) {
		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = tx;
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = ty;
		Xs[buf][ty][tx] = *(float2*)(&X[X_k*IC + ic0]);
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = dYs[buf][ik][ty];
			float2 b = Xs[buf][ik][tx];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty;
		Xs[buf][ty][tx] = *(float2*)(&X[X_k*IC + ic0]);
		__syncthreads();
	}
	if(OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = dYs[buf][ik][ty];
			float2 b = Xs[buf][ik][tx];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++) {
		float2 a = *(float2*)(&deltaY[k*OC + oc0]);//load 2 elements from deltaY
		float2 b = *(float2*)(&X[k*IC + ic0]);//load 2 elements from X
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * IC + ic0; 
	int oc1 = oc0 + IC;

	*(float2*)(deltaW + oc0) = v0;
	*(float2*)(deltaW + oc1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1)
#ifndef DECONV3D_DW_GEMM_KERNEL_2_1_W1
#define DECONV3D_DW_GEMM_KERNEL_2_1_W1

//LB = 4: Size = 0.242249, Time = 3.1   msec, Performace = 167.814 GFlop/s
//LB = 3: Size = 0.242249, Time = 3.254 msec, Performace = 159.872 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_1_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
		  float* __restrict__ deltaW, 
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float   Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;

	//prepare for GM = IC
	int ic0 = ((blockIdx.x << LB) + tx) + j_index;

	//prepare for GK = N * OH * OW
	const int OH_OW = IH * IW, GK = N * OH_OW;

	const int OK = (GK >> LB);
	if (OK) {
		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = tx;
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ty;
		Xs[buf][ty][tx] = X[X_k*IC + ic0];
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b =  Xs[buf][ik][tx];
			float2 a = dYs[buf][ik][ty];
			simdMM2(v, b, a);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty;
		Xs[buf][ty][tx] = X[X_k*IC + ic0];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b =  Xs[buf][ik][tx];
			float2 a = dYs[buf][ik][ty];
			simdMM2(v, b, a);
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++) {
		float2 a = *(float2*)(&deltaY[k*OC + oc0]);//load 2 elements from deltaY
		float  b = X[k*IC + ic0];//load 1 element from X
		simdMM2(v, b, a);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC;

	deltaW[oc0] = v.x;
	deltaW[oc1] = v.y;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef DECONV3D_DW_GEMM_KERNEL_1_2_W1
#define DECONV3D_DW_GEMM_KERNEL_1_2_W1

//LB = 4: Size = 0.242249, Time = 2.952 msec, Performace = 176.228 GFlop/s
//LB = 3: Size = 0.242249, Time = 2.802 msec, Performace = 185.662 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
		  float* __restrict__ deltaW,
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float2  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + j_index;

	//prepare for GK = N * OH * OW
	const int OH_OW = IH * IW, GK = N * OH_OW;

	const int OK = GK >> LB;
	if (OK) {
		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = ty;
		Xs[buf][ty][tx] = *(float2*)(&X[X_k*IC + ic0]);
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b =  Xs[buf][ik][tx];
			float  a = dYs[buf][ik][ty];
			simdMM2(v, a, b);
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty;
		Xs[buf][ty][tx] = *(float2*)(&X[X_k*IC + ic0]);
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b =  Xs[buf][ik][tx];
			float  a = dYs[buf][ik][ty];
			simdMM2(v, a, b);
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++) {
		float  a = deltaY[k*OC + oc0];//load 1 element from deltaY
		float2 b = *(float2*)(&X[k*IC + ic0]);//load 2 elements from X
		simdMM2(v, a, b);
	}
	//when GK % STEP != 0--------------------------------------------

	*(float2*)(&deltaW[oc0*IC + ic0]) = v;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1)
#ifndef DECONV3D_DW_GEMM_KERNEL_1_1_W1
#define DECONV3D_DW_GEMM_KERNEL_1_1_W1

//LB = 4: Size = 0.242249, Time = 2.48  msec, Performace = 209.768 GFlop/s
//LB = 3: Size = 0.242249, Time = 2.686 msec, Performace = 193.68 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_1_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
		  float* __restrict__ deltaW, 
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = IC
	int ic0 = ((blockIdx.x << LB) + tx) + j_index;

	//prepare for GK = N * OH * OW
	const int OH_OW = IH * IW, GK = N * OH_OW;

	const int OK = GK >> LB;
	if (OK) {
		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = ty;
		Xs[buf][ty][tx] = X[X_k*IC + ic0];
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float v = 0;
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float a = dYs[buf][ik][ty];
			float b = Xs[buf][ik][tx];
			v += a * b;
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty;
		Xs[buf][ty][tx] = X[X_k*IC + ic0];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float a = dYs[buf][ik][ty];
			float b = Xs[buf][ik][tx];
			v += a * b;
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++) {
		float a = deltaY[k*OC + oc0];//load 1 element from deltaY
		float b = X[k*IC + ic0];//load 1 element from X
		v += a * b;
	}
	//when GK % STEP != 0--------------------------------------------

	deltaW[oc0*IC + ic0] = v;
}

#endif

#endif