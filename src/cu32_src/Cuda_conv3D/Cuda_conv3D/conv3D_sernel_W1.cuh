#pragma once

#ifndef CONV_3D_SERNEL_W1_H
#define CONV_3D_SERNEL_W1_H

// W is the kernel in the 3D convolution: Y = conv(W, X)
// Y: (N , OC, OH, OW)
// X: (N , IC, IH, IW)
// W: (OC, IC, FH, FW)
// LB = log2(BLOCK_SIZE)
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
#ifndef CONV_3D_SERNEL_W1_CALL
#define CONV_3D_SERNEL_W1_CALL

//LB = log2(BLOCK_SIZE)

//======[Small GN: OC]==================================
#define conv3d_s1_4x2_W1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM) \
	conv3d_sernel_1_4x2_W1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GN<<1>>LB, GM>>LB>>2), dim3(1<<LB>>1, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, IC, OC, oc_index, j_index)

#endif


//======[Small GN: OC]==================================
//(Y: BLOCK_SIZE*0.5, X: BLOCK_SIZE*4)
#ifndef CONV_3D_SERNEL_1_4X2_W1
#define CONV_3D_SERNEL_1_4X2_W1

//LB = 4: Size = 0.5, Time = 4.278 msec, Performace = 250.992 GFlop/s
//LB = 4: Size = 0.492188, Time = 5.766 msec, Performace = 183.31 GFlop/s
//LB = 4: Size = 0.25, Time = 2.274 msec, Performace = 236.091 GFlop/s
//LB = 3: Size = 0.25, Time = 2.126 msec, Performace = 252.526 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3d_sernel_1_4x2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y,
	int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.x, tx = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = ((blockIdx.x << LB >> 1) + ty) + oc_index;
	const int toc0 = oc0 * IC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + tx) << 2) + j_index;
	int tj0 = j0 * IC, tj1 = tj0 + IC, tj2 = tj1 + IC, tj3 = tj2 + IC;

	const int OK = (IC >> LB);
	if (OK) {
		//load 4 elements from X[N, IH, IW, IC]
		float4 xv; int X_ic = ty;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		xv.z = X[tj2 + X_ic];
		xv.w = X[tj3 + X_ic];
		Xs[buf][ty][tx] = xv;

		X_ic += STEP2;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		xv.z = X[tj2 + X_ic];
		xv.w = X[tj3 + X_ic];
		Xs[buf][ty + STEP2][tx] = xv;

		//load 1 element from W[OC, FH, FW, IC]
		int W_ic = tx;
		Ws[buf][tx][ty] = W[toc0 + W_ic];
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

		//load 4 elements from X[N, IH, IW, IC]
		float4 xv; int X_ic = (ok << LB) + ty;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		xv.z = X[tj2 + X_ic];
		xv.w = X[tj3 + X_ic];
		Xs[buf][ty][tx] = xv;

		X_ic += STEP2;
		xv.x = X[tj0 + X_ic];
		xv.y = X[tj1 + X_ic];
		xv.z = X[tj2 + X_ic];
		xv.w = X[tj3 + X_ic];
		Xs[buf][ty + STEP2][tx] = xv;

		//load 1 element from W[OC, FH, FW, IC]
		int W_ic = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[toc0 + W_ic];
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

#endif