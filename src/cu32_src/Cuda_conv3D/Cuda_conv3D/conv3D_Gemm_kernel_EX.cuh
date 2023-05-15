#pragma once

#ifndef CONV_3D_GEMM_KERNEL_EX_H
#define CONV_3D_GEMM_KERNEL_EX_H

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
#ifndef CONV_3D_GEMM_KERNEL_EX_CALL
#define CONV_3D_GEMM_KERNEL_EX_CALL

//======[FW, IC is power of 2]================================
#define conv3dGemm_k88x4_fw_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, W, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8x4_FW_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,LFW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dGemm_k88_fw_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, W, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8_FW_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,LFW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//=======[IC is power of 2]===================================
#define conv3dGemm_k88x4_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8x4_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dGemm_k88_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8_IC2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#endif


//=======[IC is power of 2]===================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, (OH, OW) % 4 == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8X4_IC_2POW
#define CONV_3D_GEMM_KERNEL_8_8X4_IC_2POW

//LB = 4: Size = 1, Time = 1.58544 msec, Performace = 1354.5 GFlop/s
//LB = 3: Size = 1, Time = 1.84585 msec, Performace = 1163.41 GFlop/s
//LB = 4: Size = 1.125, Time = 1.73957 msec, Performace = 1388.8 GFlop/s
//LB = 3: Size = 1.125, Time = 2.16596 msec, Performace = 1115.4 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8x4_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc0 = (oc0 + ((ty >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X += X1;
	const int sw_IC = sw << LIC;

	//load 4 elements from W[OC, FH, FW, IC]
	float4 wv; int W_k = ty - ((ty >= STEP) << LB >> 1);
	wv.x = W[toc0 + W_k];
	wv.y = W[toc1 + W_k];
	wv.z = W[toc2 + W_k];
	wv.w = W[toc3 + W_k];
	Ws[buf][ty][tx] = wv;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh, int X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	int xoffset = (X_fh << LIC)*IW + X_k;
	Xs[buf][tx][ty] = SaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
		toh0, tow0, tow1, tow2, tow3);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
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
		float4 wv; int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		wv.x = W[toc0 + W_k];
		wv.y = W[toc1 + W_k];
		wv.z = W[toc2 + W_k];
		wv.w = W[toc3 + W_k];
		Ws[buf][ty][tx] = wv;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh, int X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
		int xoffset = (X_fh << LIC)*IW + X_k;
		Xs[buf][tx][ty] = SaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
			toh0, tow0, tow1, tow2, tow3);
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef CONV_3D_GEMM_KERNEL_8_8_IC_2POW
#define CONV_3D_GEMM_KERNEL_8_8_IC_2POW

//LB = 4: Size = 1, Time = 1.60456 msec, Performace = 1338.36 GFlop/s
//LB = 3: Size = 1, Time = 1.86319 msec, Performace = 1152.59 GFlop/s
//LB = 4: Size = 1.125, Time = 1.79325 msec, Performace = 1347.23 GFlop/s
//LB = 3: Size = 1.125, Time = 2.14679 msec, Performace = 1125.36 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc0 = (oc0 + ((ty >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
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

	//load 4 elements from W[OC, FH, FW, IC]
	float4 wv; int W_k = ty - ((ty >= STEP) << LB >> 1);
	wv.x = W[toc0 + W_k];
	wv.y = W[toc1 + W_k];
	wv.z = W[toc2 + W_k];
	wv.w = W[toc3 + W_k];
	Ws[buf][ty][tx] = wv;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1); 
	int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	int xoffset = ((X_fh * IW) << LIC) + X_k;
	Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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
		float4 wv; int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		wv.x = W[toc0 + W_k];
		wv.y = W[toc1 + W_k];
		wv.z = W[toc2 + W_k];
		wv.w = W[toc3 + W_k];
		Ws[buf][ty][tx] = wv;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
		int xoffset = ((X_fh * IW) << LIC) + X_k;
		Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset, 
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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


//======[FW, IC is power of 2]================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0, (OH, OW) % 4 == 0
//LB = 4, GK % 8 == 0
#ifndef CONV_3D_GEMM_KERNEL_8_8X4_FW_IC_2POW
#define CONV_3D_GEMM_KERNEL_8_8X4_FW_IC_2POW

//LB = 4: Size = 1, Time = 1.4748 msec, Performace = 1456.12 GFlop/s
//LB = 3: Size = 1, Time = 1.8118 msec, Performace = 1185.27 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8x4_FW_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int LFW,
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int LFW_IC = LFW + LIC, LFW_IC_m1 = (1 << LFW_IC) - 1;
	const int GK = FH << LFW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc0 = (oc0 + ((ty >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC; //X += X1;
	const int sw_IC = sw << LIC;

	//load 4 elements from W[OC, FH, FW, IC]
	float4 wv; int W_k = ty - ((ty >= STEP) << LB >> 1);
	wv.x = W[toc0 + W_k];
	wv.y = W[toc1 + W_k];
	wv.z = W[toc2 + W_k];
	wv.w = W[toc3 + W_k];
	Ws[buf][ty][tx] = wv;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
	int xoffset = (X_fh << LIC)*IW + X_k;
	Xs[buf][tx][ty] = SaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
		toh0, tow0, tow1, tow2, tow3);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

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
		float4 wv; int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		wv.x = W[toc0 + W_k];
		wv.y = W[toc1 + W_k];
		wv.z = W[toc2 + W_k];
		wv.w = W[toc3 + W_k];
		Ws[buf][ty][tx] = wv;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
		int xoffset = (X_fh << LIC)*IW + X_k;
		Xs[buf][tx][ty] = SaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
			toh0, tow0, tow1, tow2, tow3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef CONV_3D_GEMM_KERNEL_8_8_FW_IC_2POW
#define CONV_3D_GEMM_KERNEL_8_8_FW_IC_2POW

//LB = 4: Size = 1, Time = 1.5151  msec, Performace = 1417.39 GFlop/s
//LB = 3: Size = 1, Time = 1.84425 msec, Performace = 1164.42 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8_FW_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int LFW,
	      float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int LFW_IC = LFW + LIC, LFW_IC_m1 = (1 << LFW_IC) - 1;
	const int GK = FH << LFW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc0 = (oc0 + ((ty >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
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

	//load 4 elements from W[OC, FH, FW, IC]
	float4 wv; int W_k = ty - ((ty >= STEP) << LB >> 1);
	wv.x = W[toc0 + W_k];
	wv.y = W[toc1 + W_k];
	wv.z = W[toc2 + W_k];
	wv.w = W[toc3 + W_k];
	Ws[buf][ty][tx] = wv;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
	int xoffset = ((X_fh * IW) << LIC) + X_k;
	Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset, 
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
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
		float4 wv; int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		wv.x = W[toc0 + W_k];
		wv.y = W[toc1 + W_k];
		wv.z = W[toc2 + W_k];
		wv.w = W[toc3 + W_k];
		Ws[buf][ty][tx] = wv;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
		int xoffset = ((X_fh * IW) << LIC) + X_k;
		Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);
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

#endif
