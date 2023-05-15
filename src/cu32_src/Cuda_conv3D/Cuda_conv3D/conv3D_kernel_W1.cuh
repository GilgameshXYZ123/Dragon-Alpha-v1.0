#pragma once

#ifndef CONV_3D_KERNEL_W1_H
#define CONV_3D_KERNEL_W1_H

//Pre Experiment:
//for(FH, FW) = 1, (IC, OC) = (32, 128)
//k88<4>: Size = 0.0625, Time = 0.294 msec, Performace = 456.523 GFlop/s
//        Size = 0.125 , Time = 0.418 msec, Performace = 642.19 GFlop/s
//k88<3>: Size = 0.0625, Time = 0.306 msec, Performace = 438.62 GFlop/s
//	      Size = 0.125 , Time = 0.418 msec, Performace = 642.19  GFlop/s
//k88_ic2pow<4>: Size = 0.0625, Time = 0.282 msec, Performace = 475.949 GFlop/s
//k44<4>: Size = 0.0625, Time = 0.348 msec, Performace = 385.683 GFlop/s
//k44<3>: Size = 0.0625, Time = 0.446 msec, Performace = 300.937 GFlop/s
//
// W is the kernel in the 3D convolution: Y = conv(W, X)
// Y: (N , OC, OH, OW)
// X: (N , IC, IH, IW)
// W: (OC, IC, FH, FW)
//We have: 
//	(1) IC % 4 == 0, IC >= 4
//	(2) OC % 4 == 0, OC >= 4
//	(3) N  % 4 == 0, N  >= 4
//	(4) FH * FW == 1
//  (5) IH == OH && IW == OW
//We have: 
//	(1) GM = N  * OH * OW; GM >= 4, GM % 4 == 0
//	(2) GN = OC;           GN >= 4, GN % 4 == 0
//	(3) GK = FH * FW * IC = IC; GK >= 4, GK % 4 ==0
#ifndef CONV_3D_KERNEL_W1_CALL
#define CONV_3D_KERNEL_W1_CALL

//LB = log2(BLOCK_SIZE)

//=======[Common]==============================================
#define conv3d_k88_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_8_8_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#define conv3d_k84_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_8_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#define conv3d_k48_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_4_8_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#define conv3d_k44_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_4_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#define conv3d_k82_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_8_2_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#define conv3d_k28_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_2_8_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>1, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#define conv3d_k42_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_4_2_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#define conv3d_k24_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_2_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>1, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

//=======[Small]===============================================
#define conv3d_k22_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_2_2_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#define conv3d_k21_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_2_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#define conv3d_k14_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_1_4_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#define conv3d_k12_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_1_2_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#define conv3d_k11_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_kernel_1_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#endif


//=======[Common]==============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef CONV_3D_KERNEL_8_8_W1
#define CONV_3D_KERNEL_8_8_W1

//LB = 4: Size = 0.5, Time = 1.05199 msec, Performace = 1020.67 GFlop/s
//LB = 3: Size = 0.5, Time = 1.29664 msec, Performace = 828.094 GFlop/s
//LB = 4: Size = 0.25, Time = 0.576 msec, Performace = 932.068 GFlop/s
//LB = 3: Size = 0.25, Time = 0.734 msec, Performace = 731.432 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_8_8_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, 
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
	int toc0 = (oc0 + ((ty >= STEP) << 2))*IC;
	int toc1 = toc0 + IC, toc2 = toc1 + IC, toc3 = toc2 + IC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = (j0 + ((tx >= STEP) << 2))*IC;
	int tj1 = tj0 + IC, tj2 = tj1 + IC, tj3 = tj2 + IC;

	//load 4 elements from W[OC, FH, FW, IC]
	float4 wv; int W_ic = ty - ((ty >= STEP) << LB >> 1);//k = ic
	wv.x = W[toc0 + W_ic];
	wv.y = W[toc1 + W_ic];
	wv.z = W[toc2 + W_ic];
	wv.w = W[toc3 + W_ic];
	Ws[buf][ty][tx] = wv;

	//load 4 elements from X[N, IH, IW, IC]
	float4 xv; int X_ic = tx - ((tx >= STEP) << LB >> 1);//k = ic
	xv.x = X[tj0 + X_ic];
	xv.y = X[tj1 + X_ic];
	xv.z = X[tj2 + X_ic];
	xv.w = X[tj3 + X_ic];
	Xs[buf][tx][ty] = xv;
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0),  v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0),  v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0),  v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0),  v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0),  v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (IC << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		float4 wv; int W_ic = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		wv.x = W[toc0 + W_ic];
		wv.y = W[toc1 + W_ic];
		wv.z = W[toc2 + W_ic];
		wv.w = W[toc3 + W_ic];
		Ws[buf][ty][tx] = wv;

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
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

		simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);
		simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);
		simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);
		simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);
		simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef CONV_3D_KERNEL_8_4_W1
#define CONV_3D_KERNEL_8_4_W1

//LB = 4: Size = 0.5, Time = 1.15308 msec, Performace = 931.197 GFlop/s
//LB = 3: Size = 0.5, Time = 1.43592 msec, Performace = 747.772 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_8_4_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
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
	int toc0 = (oc0 + ((ty >= STEP) << 2))*IC;
	int toc1 = toc0 + IC, toc2 = toc1 + IC, toc3 = toc2 + IC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = (((tx & 1) << 1) + j0)*IC, tj1 = tj0 + IC;

	//load 4 elements from W[OC, FH, FW, IC]
	float4 wv; int W_ic = ty - ((ty >= STEP) << LB >> 1);//k = ic
	wv.x = W[toc0 + W_ic];
	wv.y = W[toc1 + W_ic];
	wv.z = W[toc2 + W_ic];
	wv.w = W[toc3 + W_ic];
	Ws[buf][ty][tx] = wv;

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
		float4 wv; int W_ic = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		wv.x = W[toc0 + W_ic];
		wv.y = W[toc1 + W_ic];
		wv.z = W[toc2 + W_ic];
		wv.w = W[toc3 + W_ic];
		Ws[buf][ty][tx] = wv;

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
#ifndef CONV_3D_KERNEL_4_8_W1
#define CONV_3D_KERNEL_4_8_W1

//LB = 4: Size = 0.5, Time = 0.924209 msec, Performace = 1161.8  GFlop/s
//LB = 3: Size = 0.5, Time = 1.38357  msec, Performace = 776.065 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_4_8_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
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
	int toc0 = (((ty & 1) << 1) + oc0)*IC, toc1 = toc0 + IC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = (j0 + ((tx >= STEP) << 2))*IC;
	int tj1 = tj0 + IC, tj2 = tj1 + IC, tj3 = tj2 + IC;

	//load 2 elements from W[OC, FH, FW, IC]
	float2 wv; int W_ic = ty >> 1;//k = ic
	wv.x = W[toc0 + W_ic];
	wv.y = W[toc1 + W_ic];
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = wv;

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
		float2 wv; int W_ic = ((ok << LB) + ty) >> 1;
		wv.x = W[toc0 + W_ic];
		wv.y = W[toc1 + W_ic];
		Ws[buf][Ws_y][Ws_x] = wv;

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
#ifndef CONV_3D_KERNEL_4_4_W1
#define CONV_3D_KERNEL_4_4_W1

//LB = 4: Size = 0.5, Time = 1.10695 msec, Performace = 969.997 GFlop/s
//LB = 3: Size = 0.5, Time = 1.77753 msec, Performace = 604.065 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_4_4_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
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
	int toc0 = (((ty & 1) << 1) + oc0)*IC, toc1 = toc0 + IC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = (((tx & 1) << 1) + j0)*IC, tj1 = tj0 + IC;

	//load 2 elements from W[OC, FH, FW, IC]
	float2 wv; int W_ic = ty >> 1;//k = icw
	wv.x = W[toc0 + W_ic];
	wv.y = W[toc1 + W_ic];
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = wv;

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
		float2 wv; int W_ic = ((ok << LB) + ty) >> 1;
		wv.x = W[toc0 + W_ic];
		wv.y = W[toc1 + W_ic];
		Ws[buf][Ws_y][Ws_x] = wv;

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
#ifndef CONV_3D_KERNEL_8_2_W1
#define CONV_3D_KERNEL_8_2_W1

//LB = 4: Size = 0.5, Time = 1.63225 msec, Performace = 657.83 GFlop/s
//LB = 3: Size = 0.5, Time = 1.90902 msec, Performace = 562.458 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_8_2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
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
	int toc0 = (oc0 + ((ty >= STEP) << 2))*IC;
	int toc1 = toc0 + IC, toc2 = toc1 + IC, toc3 = toc2 + IC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index;
	int tj0 = ((tx & 1) + j0)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	float4 wv; int W_ic = ty - ((ty >= STEP) << LB >> 1);//k = ic
	wv.x = W[toc0 + W_ic];
	wv.y = W[toc1 + W_ic];
	wv.z = W[toc2 + W_ic];
	wv.w = W[toc3 + W_ic];
	Ws[buf][ty][tx] = wv;

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
		float4 wv; int W_ic = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		wv.x = W[toc0 + W_ic];
		wv.y = W[toc1 + W_ic];
		wv.z = W[toc2 + W_ic];
		wv.w = W[toc3 + W_ic];
		Ws[buf][ty][tx] = wv;

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


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef CONV_3D_KERNEL_2_8_W1
#define CONV_3D_KERNEL_2_8_W1

//LB = 4: Size = 0.5, Time = 1.17552 msec, Performace = 913.418 GFlop/s
//LB = 3: Size = 0.5, Time = 1.88593 msec, Performace = 569.345 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_2_8_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 1) + oc_index;
	int toc0 = ((ty & 1) + oc0)*IC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = (j0 + ((tx >= STEP) << 2))*IC;
	int tj1 = tj0 + IC, tj2 = tj1 + IC, tj3 = tj2 + IC;

	//load 4 elements from X[N, IH, IW, IC]
	float4 xv; int X_ic = tx - ((tx >= STEP) << LB >> 1);//k = ic
	xv.x = X[tj0 + X_ic];
	xv.y = X[tj1 + X_ic];
	xv.z = X[tj2 + X_ic];
	xv.w = X[tj3 + X_ic];
	Xs[buf][tx][ty] = xv;

	//load 1 element from W[OC, FH, FW, IC]
	int W_ic = ty >> 1;//k = ic
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = W[toc0 + W_ic];
	__syncthreads();

	//compute area----------------------------------------------------
	float2  v0 = make_float2(0, 0);
	float2  v2 = make_float2(0, 0);
	float2  v4 = make_float2(0, 0);
	float2  v6 = make_float2(0, 0);
	float2  v8 = make_float2(0, 0);
	float2 v10 = make_float2(0, 0);
	float2 v12 = make_float2(0, 0);
	float2 v14 = make_float2(0, 0);

	for (int ok = 1, OK = (IC << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float2 a0 = *(float2*)(&Ws[buf][ik][tx << 1]);

			simdMM2(v0, b0.x, a0);
			simdMM2(v2, b0.y, a0);
			simdMM2(v4, b0.z, a0);
			simdMM2(v6, b0.w, a0);
			simdMM2(v8, b1.x, a0);
			simdMM2(v10, b1.y, a0);
			simdMM2(v12, b1.z, a0);
			simdMM2(v14, b1.w, a0);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		float4 xv; int X_ic = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		xv.z = X[tj2 + X_ic];
		xv.w = X[tj3 + X_ic];
		Xs[buf][tx][ty] = xv;

		//load 1 element from W[OC, FH, FW, IC]
		int W_ic = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = W[toc0 + W_ic];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float2 a0 = *(float2*)(&Ws[buf][ik][tx << 1]);

		simdMM2(v0, b0.x, a0);
		simdMM2(v2, b0.y, a0);
		simdMM2(v4, b0.z, a0);
		simdMM2(v6, b0.w, a0);
		simdMM2(v8, b1.x, a0);
		simdMM2(v10, b1.y, a0);
		simdMM2(v12, b1.z, a0);
		simdMM2(v14, b1.w, a0);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v2;
	*(float2*)(Y + j2) = v4;
	*(float2*)(Y + j3) = v6;
	*(float2*)(Y + j4) = v8;
	*(float2*)(Y + j5) = v10;
	*(float2*)(Y + j6) = v12;
	*(float2*)(Y + j7) = v14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), GK % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef CONV_3D_KERNEL_4_2_W1
#define CONV_3D_KERNEL_4_2_W1

//LB = 4: Size = 0.5, Time = 1.68509 msec, Performace = 637.2 GFlop/s
//LB = 3: Size = 0.5, Time = 2.62026 msec, Performace = 409.785 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_4_2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
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
	int toc0 = (((ty & 1) << 1) + oc0)*IC, toc1 = toc0 + IC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index;
	int tj0 = ((tx & 1) + j0)*IC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_ic = ty >> 1;//k = ic
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x].x = W[toc0 + W_ic];
	Ws[buf][Ws_y][Ws_x].y = W[toc1 + W_ic];

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
		Ws[buf][Ws_y][Ws_x].x = W[toc0 + W_ic];
		Ws[buf][Ws_y][Ws_x].y = W[toc1 + W_ic];

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


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef CONV_3D_KERNEL_2_4_W1
#define CONV_3D_KERNEL_2_4_W1

//LB = 4: Size = 0.5, Time = 1.41238 msec, Performace = 760.234 GFlop/s
//LB = 4: Size = 0.5, Time = 2.31529 msec, Performace = 463.76 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_2_4_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
		  float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 1) + oc_index;
	int toc0 = ((ty & 1) + oc0)*IC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = (((tx & 1) << 1) + j0)*IC, tj1 = tj0 + IC;

	//load 1 element from W[OC, FH, FW, IC]
	int W_ic = ty >> 1;//k = ic
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = W[toc0 + W_ic];

	//load 2 elements from X[N, IH, IW, IC]
	int X_ic = tx >> 1;//k = ic
	const int Xs_x = (tx >> 1), Xs_y = (ty << 1) + (tx & 1);
	Xs[buf][Xs_x][Xs_y].x = X[tj0 + X_ic];
	Xs[buf][Xs_x][Xs_y].y = X[tj1 + X_ic];
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (IC << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 a = *(float2*)(&Ws[buf][ik][tx << 1]);
			float4 b = *(float4*)(&Xs[buf][ik][ty << 1]);

			simdMM2(v0, b.x, a);
			simdMM2(v1, b.y, a);
			simdMM2(v2, b.z, a);
			simdMM2(v3, b.w, a);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_ic = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = W[toc0 + W_ic];

		//load 2 elements from X[N, IH, IW, IC]
		int X_ic = ((ok << LB) + tx) >> 1;
		Xs[buf][Xs_x][Xs_y].x = X[tj0 + X_ic];
		Xs[buf][Xs_x][Xs_y].y = X[tj1 + X_ic];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float2 a = *(float2*)(&Ws[buf][ik][tx << 1]);
		float4 b = *(float4*)(&Xs[buf][ik][ty << 1]);

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


//=======[Small]===============================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
#ifndef CONV_3D_KERNEL_2_2_W1
#define CONV_3D_KERNEL_2_2_W1

//LB = 4: Size = 0.5 , Time = 2.724 msec, Performace = 394.178 GFlop/s
//LB = 4: Size = 0.492188, Time = 3.242 msec, Performace = 326.022 GFlop/s
//LB = 4: Size = 0.25, Time = 1.764 msec, Performace = 304.349 GFlop/s
//LB = 3: Size = 0.25, Time = 2.078 msec, Performace = 258.359 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_2_2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
		  float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	int toc0 = oc0 * IC, toc1 = toc0 + IC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int tj0 = j0 * IC, tj1 = tj0 + IC;

	const int OK = (IC >> LB);
	if (OK) {
		//load 2 elements from W[OC, FH, FW, IC]
		int W_ic = tx;
		Ws[buf][tx][ty].x = W[toc0 + W_ic];
		Ws[buf][tx][ty].y = W[toc1 + W_ic];

		//load 2 elements from X[N, IH, IW, IC]
		int X_ic = ty;
		Xs[buf][ty][tx].x = X[tj0 + X_ic];
		Xs[buf][ty][tx].y = X[tj1 + X_ic];
		__syncthreads();
	}
	
	//compute area-----------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = Ws[buf][ik][ty];
			float2 b = Xs[buf][ik][tx];
			simdMM2(v0, b.x, a);
			simdMM2(v1, b.y, a);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_ic = (ok << LB) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_ic];
		Ws[buf][tx][ty].y = W[toc1 + W_ic];

		//load 2 elements from X[N, IH, IW, IC]
		int X_ic = (ok << LB) + ty;
		Xs[buf][ty][tx].x = X[tj0 + X_ic];
		Xs[buf][ty][tx].y = X[tj1 + X_ic];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = Ws[buf][ik][ty];
			float2 b = Xs[buf][ik][tx];
			simdMM2(v0, b.x, a);
			simdMM2(v1, b.y, a);
		}

	//when IC % STEP != 0---------------------------------------
	for (int ic = IC - (IC & (STEP - 1)); ic < IC; ic++)
	{
		float2 a;//load 2 elements from W
		a.x = W[toc0 + ic];
		a.y = W[toc1 + ic];
		
		float2 b;//load 2 elements from X
		b.x = X[tj0 + ic];
		b.y = X[tj1 + ic];

		simdMM2(v0, b.x, a);
		simdMM2(v1, b.y, a);
	}
	//when IC % STEP != 0---------------------------------------
	
	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC;

	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1)
#ifndef CONV_3D_KERNEL_2_1_W1
#define CONV_3D_KERNEL_2_1_W1

//LB = 4: Size = 0.492188, Time = 3.832 msec, Performace = 275.826 GFlop/s
//LB = 4: Size = 0.25, Time = 2.194 msec, Performace = 244.7   GFlop/s
//LB = 3: Size = 0.25, Time = 1.906 msec, Performace = 281.674 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_2_1_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
		  float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int toc0 = oc0 * IC, toc1 = toc0 + IC;

	//prepare for GM = N * OH * OW
	int j0 = ((blockIdx.x << LB) + tx)  + j_index;
	int tj0 = j0 * IC;

	const int OK = (IC >> LB);
	if (OK) {
		//load 2 elements from W[OC, FH, FW, IC]
		int W_ic = tx;
		Ws[buf][tx][ty].x = W[toc0 + W_ic];
		Ws[buf][tx][ty].y = W[toc1 + W_ic];

		//load 1 element from X[N, IH, IW, IC]
		int X_ic = ty;
		Xs[buf][ty][tx] = X[tj0 + X_ic];
		__syncthreads();
	}

	//compute area-------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = Ws[buf][ik][ty];
			float  b = Xs[buf][ik][tx];
			simdMM2(v, b, a);
		}

		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_ic = (ok << LB) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_ic];
		Ws[buf][tx][ty].y = W[toc1 + W_ic];

		//load 1 element from X[N, IH, IW, IC]
		int X_ic = (ok << LB) + ty;
		Xs[buf][ty][tx] = X[tj0 + X_ic];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = Ws[buf][ik][ty];
			float  b = Xs[buf][ik][tx];
			simdMM2(v, b, a);
		}

	//when IC%STEP != 0 --------------------------------------
	for (int ic = IC - (IC & (STEP - 1)); ic < IC; ic++)
	{
		float2 a;//load 2 elements from W
		a.x = W[toc0 + ic];
		a.y = W[toc1 + ic];

		float b = X[tj0 + ic];//load 1 element from X

		simdMM2(v, b, a);
	}
	//when IC%STEP != 0 --------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	*(float2*)(Y + j0) = v;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*4)
#ifndef CONV_3D_KERNEL_1_4_W1
#define CONV_3D_KERNEL_1_4_W1

//LB = 4: Size = 0.5, Time = 4.278 msec, Performace = 250.992 GFlop/s
//LB = 4: Size = 0.492188, Time = 5.766 msec, Performace = 183.31 GFlop/s
//LB = 4: Size = 0.25, Time = 2.274 msec, Performace = 236.091 GFlop/s
//LB = 3: Size = 0.25, Time = 2.126 msec, Performace = 252.526 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_1_4_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	const int toc0 = oc0 * IC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 * IC, tj1 = tj0 + IC, tj2 = tj1 + IC, tj3 = tj2 + IC;

	const int OK = (IC >> LB);
	if (OK) {
		//load 1 element from W[OC, FH, FW, IC]
		int W_ic = tx;
		Ws[buf][tx][ty] = W[toc0 + W_ic];

		//load 4 elements from X[N, IH, IW, IC]
		float4 xv; int X_ic = ty;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		xv.z = X[tj2 + X_ic];
		xv.w = X[tj3 + X_ic];
		Xs[buf][ty][tx] = xv;
		__syncthreads();
	}

	//compute area-------------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  a = Ws[buf][ik][ty];
			float4 b = Xs[buf][ik][tx];
			simdMM4(v, a, b);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_ic = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[toc0 + W_ic];

		//load 4 elements from X[N, IH, IW, IC]
		float4 xv; int X_ic = (ok << LB) + ty;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		xv.z = X[tj2 + X_ic];
		xv.w = X[tj3 + X_ic];
		Xs[buf][ty][tx] = xv;
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  a = Ws[buf][ik][ty];
			float4 b = Xs[buf][ik][tx];
			simdMM4(v, a, b);
		}

	//when IC % STEP != 0---------------------------------------
#pragma unroll
	for (int ic = IC - (IC & (STEP - 1)); ic < IC; ic++)
	{
		float a = W[toc0 + ic];//load 1 element from W

		float4 b;//load 4 elements from X
		b.x = X[tj0 + ic];
		b.y = X[tj1 + ic];
		b.z = X[tj2 + ic];
		b.w = X[tj3 + ic];

		simdMM4(v, a, b);
	}
	//when IC % STEP != 0---------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC;
	const int j2 = j1 + OC;
	const int j3 = j2 + OC;

	Y[j0] = v.x;
	Y[j1] = v.y;
	Y[j2] = v.z;
	Y[j3] = v.w;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef CONV_3D_KERNEL_1_2_W1
#define CONV_3D_KERNEL_1_2_W1

//LB = 4: Size = 0.5, Time = 4.278 msec, Performace = 250.992 GFlop/s
//LB = 4: Size = 0.492188, Time = 5.766 msec, Performace = 183.31 GFlop/s
//LB = 4: Size = 0.25, Time = 2.274 msec, Performace = 236.091 GFlop/s
//LB = 3: Size = 0.25, Time = 2.126 msec, Performace = 252.526 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_1_2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
		  float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	const int toc0 = oc0 * IC;
	
	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int tj0 = j0 * IC, tj1 = tj0 + IC;

	const int OK = (IC >> LB);
	if (OK) {
		//load 1 element from W[OC, FH, FW, IC]
		int W_ic = tx;
		Ws[buf][tx][ty] = W[toc0 + W_ic];

		//load 2 elements from X[N, IH, IW, IC]
		float2 xv; int X_ic = ty;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		Xs[buf][ty][tx] = xv;
		__syncthreads();
	}
	
	//compute area-------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  a = Ws[buf][ik][ty];
			float2 b = Xs[buf][ik][tx];
			simdMM2(v, a, b);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_ic = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[toc0 + W_ic];

		//load 2 elements from X[N, IH, IW, IC]
		float2 xv; int X_ic = (ok << LB) + ty;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		Xs[buf][ty][tx] = xv;
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  a = Ws[buf][ik][ty];
			float2 b = Xs[buf][ik][tx];
			simdMM2(v, a, b);
		}

	//when IC % STEP != 0---------------------------------------
#pragma unroll
	for (int ic = IC - (IC & (STEP - 1)); ic < IC; ic++)
	{
		float a = W[toc0 + ic];//load 1 element from W
		
		float2 b;//load 2 elements from X
		b.x = X[tj0 + ic];
		b.y = X[tj1 + ic];

		simdMM2(v, a, b);
	}
	//when IC % STEP != 0---------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC;

	Y[j0] = v.x;
	Y[j1] = v.y;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1)
#ifndef CONV_3D_KERNEL_1_1_W1
#define CONV_3D_KERNEL_1_1_W1

//LB = 4: Size = 0.492188, Time = 6.438 msec, Performace = 164.176 GFlop/s
//LB = 4: Size = 0.25, Time = 2.788 msec, Performace = 192.565 GFlop/s
//LB = 3: Size = 0.25, Time = 2.764 msec, Performace = 194.237 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_kernel_1_1_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
		  float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	const int toc0 = oc0 * IC;

	//prepare for GM = N * OH * OW
	const int j0 = ((blockIdx.x << LB) + tx) + j_index;
	const int tj0 = j0 * IC;

	const int OK = (IC >> LB);
	if (OK) {
		//load 1 element from W[OC, FH, FW, IC]
		int W_ic = tx;
		Ws[buf][tx][ty] = W[toc0 + W_ic];

		//load 1 element from X[N, IH, IW, IC]
		int X_ic = ty;
		Xs[buf][ty][tx] = X[tj0 + X_ic];
		__syncthreads();
	}
	
	//compute area----------------------------------------------------
	float v = 0;
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float a = Ws[buf][ik][ty];
			float b = Xs[buf][ik][tx];
			v += a * b;
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_ic = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[toc0 + W_ic];

		//load 1 element from X[N, IH, IW, IC]
		int X_ic = (ok << LB) + ty;
		Xs[buf][ty][tx] = X[tj0 + X_ic];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float a = Ws[buf][ik][ty];
			float b = Xs[buf][ik][tx];
			v += a * b;
		}

	//when IC % STEP != 0---------------------------------------
	for (int ic = IC - (IC & (STEP - 1)); ic < IC; ic++) {
		float a = W[toc0 + ic];
		float b = X[tj0 + ic];
		v += a * b;
	}
	//when IC % STEP != 0---------------------------------------

	Y[j0*OC + oc0] = v;
}

#endif

#endif
