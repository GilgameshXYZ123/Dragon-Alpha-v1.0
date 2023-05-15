#pragma once

#ifndef CONV_3D_GEMMR_KERNEL_W1_H
#define CONV_3D_GEMMR_KERNEL_W1_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, OC, IC]
//	(1) IC % 4 == 0, IC >= 4
//	(2) OC % 4 == 0, OC >= 4
//	(3) N  % 4 == 0, N  >= 4
//	(4) FH * FW == 1
//  (5) IH == OH && IW == OW
#ifndef CONV_3D_GEMMR_KERNEL_W1_CALL
#define CONV_3D_GEMMR_KERNEL_W1_CALL

#define conv3d_u88R_W1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM) \
	conv3d_uernel_8_8R_W1<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, IC, OC, oc_index, j_index)

#define conv3d_k88R_W1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM) \
	conv3d_kernel_8_8R_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, IC, OC, oc_index, j_index)

#define conv3d_k84R_W1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM) \
	conv3d_kernel_8_4R_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, IC, OC, oc_index, j_index)

#define conv3d_k48R_W1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM) \
	conv3d_kernel_4_8R_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, IC, OC, oc_index, j_index)

#define conv3d_k44R_W1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM) \
	conv3d_kernel_4_4R_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, IC, OC, oc_index, j_index)

#define conv3d_k82R_W1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM) \
	conv3d_kernel_8_2R_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, IC, OC, oc_index, j_index)

#define conv3d_k42R_W1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM) \
	conv3d_kernel_4_2R_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, IC, OC, oc_index, j_index)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0 
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_UERNEL_8_8R_W1
#define CONV_3D_UERNEL_8_8R_W1

//LB = 4: Size = 1, Time = 1.49135 msec, Performace = 1439.96 GFlop/s
//LB = 3: Size = 1, Time = 1.52803 msec, Performace = 1405.4 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3d_uernel_8_8R_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y,
	int IC, int OC,
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
	int tj0 = (j0 + ((tx >= STEP) << 2))*IC;
	int tj1 = tj0 + IC, tj2 = tj1 + IC, tj3 = tj2 + IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_ic = (tx - ((tx >= STEP) << LB >> 1)) << 1;//k = ic
	float2 x0 = *(float2*)(X + tj0 + X_ic);
	float2 x1 = *(float2*)(X + tj1 + X_ic);
	float2 x2 = *(float2*)(X + tj2 + X_ic);
	float2 x3 = *(float2*)(X + tj3 + X_ic);
	Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	int W_ic = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int woffset0 = W_ic * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
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

	for (int ok = 1, OK = (IC >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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

		//load 4 elements from X[N, IH, IW, IC]
		int X_ic = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		float2 x0 = *(float2*)(X + tj0 + X_ic);
		float2 x1 = *(float2*)(X + tj1 + X_ic);
		float2 x2 = *(float2*)(X + tj2 + X_ic);
		float2 x3 = *(float2*)(X + tj3 + X_ic);
		Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		int W_ic = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int woffset0 = W_ic * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, b0.x, a0);  simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0);  simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0);  simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0);  simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0);  simdMM4(v9, b1.x, a1);
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
//LB = 4, IC % 8 == 0 
#ifndef CONV_3D_KERNEL_8_8R_W1
#define CONV_3D_KERNEL_8_8R_W1

//LB = 4: Size = 0.5, Time = 0.970717 msec, Performace = 1106.13 GFlop/s
//LB = 3: Size = 0.5, Time = 1.14279  msec, Performace =  939.578 GFlop/s
//LB = 4: Size = 1, Time = 1.51417 msec, Performace = 1418.26 GFlop/s
//LB = 3: Size = 1, Time = 1.94755 msec, Performace = 1102.66 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_8_8R_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y,
	int IC, int OC,
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
	int tj0 = (j0 + ((tx >= STEP) << 2))*IC;
	int tj1 = tj0 + IC, tj2 = tj1 + IC, tj3 = tj2 + IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_ic = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_ic * OC);

	//load 4 elements from X[N, IH, IW, IC]
	float4 xv; int X_ic = tx - ((tx >= STEP) << LB >> 1);//k = ic
	xv.x = X[tj0 + X_ic];
	xv.y = X[tj1 + X_ic];
	xv.z = X[tj2 + X_ic];
	xv.w = X[tj3 + X_ic];
	Xs[buf][tx][ty] = xv;
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

	for (int ok = 1, OK = (IC << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
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
		int W_ic = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_ic * OC);

		//load 4 elements from X[N, IH, IW, IC]
		float4 xv; int X_ic = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		xv.z = X[tj2 + X_ic];
		xv.w = X[tj3 + X_ic];
		Xs[buf][tx][ty] = xv;
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef CONV_3D_KERNEL_8_4R_W1
#define CONV_3D_KERNEL_8_4R_W1

//LB = 4: Size = 0.5, Time = 1.05253 msec, Performace = 1020.15 GFlop/s
//LB = 3: Size = 0.5, Time = 1.20914 msec, Performace =  888.018 GFlop/s
//LB = 4: Size = 1, Time = 1.86795 msec, Performace = 1149.65 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_8_4R_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = (((tx & 1) << 1) + j0)*IC, tj1 = tj0 + IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_ic = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_ic * OC);

	//load 2 elements from X[N, IH, IW, IC]
	float2 xv; int X_ic = tx >> 1;//k = ic
	xv.x = X[tj0 + X_ic];
	xv.y = X[tj1 + X_ic];
	const int Xs_x = (tx >> 1), Xs_y = (ty << 1) + (tx & 1);
	Xs[buf][Xs_x][Xs_y] = xv;
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (IC << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
			float4 b0 = *(float4*)(&Xs[buf][ik][ty << 1]);

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_ic = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_ic * OC);

		//load 2 elements from X[N, IH, IW, IC]
		float2 xv; int X_ic = ((ok << LB) + tx) >> 1;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		Xs[buf][Xs_x][Xs_y] = xv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
		float4 b0 = *(float4*)(&Xs[buf][ik][ty << 1]);

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
//LB = 4, IC % 8 == 0 
#ifndef CONV_3D_KERNEL_4_8R_W1
#define CONV_3D_KERNEL_4_8R_W1

//LB = 4: Size = 1, Time = 1.59658 msec, Performace = 1345.05 GFlop/s
//LB = 3: Size = 1, Time = 1.82873 msec, Performace = 1174.3 GFlop/s
//LB = 4: Size = 0.5, Time = 0.886728 msec, Performace = 1210.9 GFlop/s
//LB = 3: Size = 0.5, Time = 0.983879 msec, Performace = 1091.34 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_4_8R_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += ((ty & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = (j0 + ((tx >= STEP) << 2))*IC;
	int tj1 = tj0 + IC, tj2 = tj1 + IC, tj3 = tj2 + IC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_ic = ty >> 1;//k = ic
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_ic * OC);

	//load 4 elements from X[N, IH, IW, IC]
	float4 xv; int X_ic = tx - ((tx >= STEP) << LB >> 1);//k = ic
	xv.x = X[tj0 + X_ic];
	xv.y = X[tj1 + X_ic];
	xv.z = X[tj2 + X_ic];
	xv.w = X[tj3 + X_ic];
	Xs[buf][tx][ty] = xv;
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (IC << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]);
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

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
		int W_ic = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_ic * OC);

		//load 4 elements from X[N, IH, IW, IC]
		float4 xv; int X_ic = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		xv.z = X[tj2 + X_ic];
		xv.w = X[tj3 + X_ic];
		Xs[buf][tx][ty] = xv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]);
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef CONV_3D_KERNEL_4_4R_W1
#define CONV_3D_KERNEL_4_4R_W1

//LB = 4: Size = 0.5, Time = 1.08193 msec, Performace = 992.429 GFlop/s
//LB = 3: Size = 0.5, Time = 1.19178 msec, Performace = 900.956 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_4_4R_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += ((ty & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = (((tx & 1) << 1) + j0)*IC, tj1 = tj0 + IC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_ic = ty >> 1;//k = icw
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_ic * OC);

	//load 2 elements from X[N, IH, IW, IC]
	float2 xv; int X_ic = tx >> 1;//k = ic
	xv.x = X[tj0 + X_ic];
	xv.y = X[tj1 + X_ic];
	const int Xs_x = (tx >> 1), Xs_y = (ty << 1) + (tx & 1);
	Xs[buf][Xs_x][Xs_y] = xv;
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (IC << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a = *(float4*)(&Ws[buf][ik][tx << 1]);
			float4 b = *(float4*)(&Xs[buf][ik][ty << 1]);

			simdMM4(v0, b.x, a);
			simdMM4(v1, b.y, a);
			simdMM4(v2, b.z, a);
			simdMM4(v3, b.w, a);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_ic = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_ic * OC);

		//load 2 elements from X[N, IH, IW, IC]
		float2 xv; int X_ic = ((ok << LB) + tx) >> 1;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		Xs[buf][Xs_x][Xs_y] = xv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&Ws[buf][ik][tx << 1]);
		float4 b = *(float4*)(&Xs[buf][ik][ty << 1]);

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), GK % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef CONV_3D_KERNEL_8_2R_W1
#define CONV_3D_KERNEL_8_2R_W1

//LB = 4: Size = 0.5, Time = 1.49897 msec, Performace = 716.318 GFlop/s
//LB = 3: Size = 0.5, Time = 1.55011 msec, Performace = 692.687 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_8_2R_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index;
	int tj0 = ((tx & 1) + j0)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_ic = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_ic * OC);

	//load 1 element from X[N, IH, IW, IC]
	int X_ic = tx >> 1;//k = ic
	const int Xs_x = (tx >> 1), Xs_y = (ty << 1) + (tx & 1);
	Xs[buf][Xs_x][Xs_y] = X[tj0 + X_ic];
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (IC << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
			float2 b0 = *(float2*)(&Xs[buf][ik][ty << 1]);
			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_ic = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_ic * OC);

		//load 1 element from X[N, IH, IW, IC]
		int X_ic = ((ok << LB) + tx) >> 1;
		Xs[buf][Xs_x][Xs_y] = X[tj0 + X_ic];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
		float2 b0 = *(float2*)(&Xs[buf][ik][ty << 1]);
		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), GK % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef CONV_3D_KERNEL_4_2R_W1
#define CONV_3D_KERNEL_4_2R_W1

//LB = 4: Size = 0.5, Time = 1.63309 msec, Performace = 657.489 GFlop/s
//LB = 3: Size = 0.5, Time = 1.77151 msec, Performace = 606.116 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_4_2R_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += ((ty & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index;
	int tj0 = ((tx & 1) + j0)*IC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_ic = ty >> 1;//k = ic
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_ic * OC);

	//load 1 element from X[N, IH, IW, IC]
	int X_ic = tx >> 1;
	const int Xs_x = (tx >> 1), Xs_y = (ty << 1) + (tx & 1);
	Xs[buf][Xs_x][Xs_y] = X[tj0 + X_ic];
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (IC << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 a = *(float4*)(&Ws[buf][ik][tx << 1]);
			float2 b = *(float2*)(&Xs[buf][ik][ty << 1]);
			simdMM4(v0, b.x, a);
			simdMM4(v1, b.y, a);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_ic = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_ic * OC);

		//load 1 element from X[N, IH, IW, IC]
		int X_ic = ((ok << LB) + tx) >> 1;
		Xs[buf][Xs_x][Xs_y] = X[tj0 + X_ic];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 a = *(float4*)(&Ws[buf][ik][tx << 1]);
		float2 b = *(float2*)(&Xs[buf][ik][ty << 1]);
		simdMM4(v0, b.x, a);
		simdMM4(v1, b.y, a);
	}

	j0 = j0 * OC + oc0; //oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
}

#endif

#endif
