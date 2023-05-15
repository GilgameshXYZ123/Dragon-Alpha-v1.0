#pragma once

#ifndef CONV_3D_GEMM_SERNEL_H
#define CONV_3D_GEMM_SERNEL_H

// W is the kernel in a 3D convolution: Y = conv(W, X)
// Y: (N , OH, OW, OC)
// X: (N , IH, IW, IC)
// W: (OC, FH, FW, IC)
// LB = log2(BLOCK_SIZE)
//We have: 
//	(1) IC % 4 == 0, IC >= 4
//	(2) OC % 4 == 0, OC >= 4
//	(3) N  % 4 == 0, N  >= 4
//	(4) FH * FW >= 2
//We have: 
//	(1) GM = N  * OH * OW; GM >= 4, GM % 4 == 0
//	(2) GN = OC;           GN >= 4, GN % 4 == 0
//	(3) GK = FH * FW * IC; GK >= 8, GK % 4 == 0
#ifndef CONV_3D_GEMM_SERNEL_CALL
#define CONV_3D_GEMM_SERNEL_CALL

//LB = log2(BLOCK_SIZE)

//======[Small GN: OC]===================================
#define conv3dGemm_s1_4x2(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_sernel_1_4x2<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN<<1>>LB), dim3(1<<LB, 1<<LB>>1), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dGemm_s1_2x2(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_sernel_1_2x2<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN<<1>>LB), dim3(1<<LB, 1<<LB>>1), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#endif


//======[Small GN: OC]===================================
//(Y: BLOCK_SIZE*0.5£¬X: BLOCK_SIZE*4), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef CONV_3D_GEMM_SERNEL_1_4X2
#define CONV_3D_GEMM_SERNEL_1_4X2

//[OC = 8]: LB = 3: Size = 1, Time = 6.72 msec , Performace = 319.566 GFlop/s
//[OC = 4]: LB = 3: Size = 0.28125, Time = 7.67791 msec, Performace = 78.6646 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_sernel_1_4x2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB >> 1) + ty) + oc_index;
	W += oc0 * GK;//W[oc0, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	get_n_oh_ow(j2, n2, oh2, ow2);
	get_n_oh_ow(j3, n3, oh3, ow3);
	const int toh0 = oh0 * sh - ph, tow0 = ow0 * sw - pw;
	const int toh1 = oh1 * sh - ph, tow1 = ow1 * sw - pw;
	const int toh2 = oh2 * sh - ph, tow2 = ow2 * sw - pw;
	const int toh3 = oh3 * sh - ph, tow3 = ow3 * sw - pw;
	const int X0 = ((n0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((n1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((n2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((n3*IH + toh3)*IW + tow3)*IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty, X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	Xs[buf][ty][tx] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);

	X_k = ty + STEP2; 
	get_X_fh_fw(X_k, X_fh, X_fw);
	xoffset = X_fh * IW_IC + X_k;
	Xs[buf][ty + STEP2][tx] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);

	//load 1 element from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty] = W[W_k];
	__syncthreads();

	//compute area------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  a = Ws[buf][ik][ty];
			float4 b = Xs[buf][ik][tx];
			simdMM4(v, a, b);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		X_k = (ok << LB) + ty + STEP2;
		get_X_fh_fw(X_k, X_fh, X_fw);
		xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty + STEP2][tx] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[W_k];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  a = Ws[buf][ik][ty];
		float4 b = Xs[buf][ik][tx];
		simdMM4(v, a, b);
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float a = W[k];//load 1 element from W

		int X_k = k;//load 4 elements from X
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		float4 b = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		simdMM4(v, a, b);
	}
	//when GK % STEP != 0---------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	j1 = j0 + OC; j2 = j1 + OC; j3 = j2 + OC;

	Y[j0] = v.x;
	Y[j1] = v.y;
	Y[j2] = v.z;
	Y[j3] = v.w;
}

#endif


//(Y: BLOCK_SIZE*0.5£¬X: BLOCK_SIZE*4), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef CONV_3D_GEMM_SERNEL_1_2X2
#define CONV_3D_GEMM_SERNEL_1_2X2

//[OC = 8]: LB = 3: Size = 0.28125, Time = 5.51867 msec, Performace = 109.443 GFlop/s
//[OC = 4]: LB = 3: Size = 0.28125, Time = 5.453   msec, Performace = 110.761 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_sernel_1_2x2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB >> 1) + ty) + oc_index;
	W += oc0 * GK;//W[oc0, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	const int toh0 = oh0 * sh - ph, tow0 = ow0 * sw - pw;
	const int toh1 = oh1 * sh - ph, tow1 = ow1 * sw - pw;
	const int X0 = ((n0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((n1*IH + toh1)*IW + tow1)*IC;

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty, X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	Xs[buf][ty][tx] = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1);

	X_k = ty + STEP2;
	get_X_fh_fw(X_k, X_fh, X_fw);
	xoffset = X_fh * IW_IC + X_k;
	Xs[buf][ty + STEP2][tx] = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1);

	//load 1 element from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty] = W[W_k];
	__syncthreads();

	//compute area------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float  a = Ws[buf][ik][ty];
			simdMM2(v, a, b);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx] = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1);

		X_k = (ok << LB) + ty + STEP2;
		get_X_fh_fw(X_k, X_fh, X_fw);
		xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty + STEP2][tx] = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1);

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[W_k];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Xs[buf][ik][tx];
		float  a = Ws[buf][ik][ty];
		simdMM2(v, a, b);
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float a = W[k];//load 1 element from W

		int X_k = k;//load 2 elements from X
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		float2 b = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1);

		simdMM2(v, a, b);
	}
	//when GK % STEP != 0---------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	j1 = j0 + OC;

	Y[j0] = v.x;
	Y[j1] = v.y;
}

#endif

#endif