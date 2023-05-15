#pragma once

#ifndef CONV_3D_GEMM_KERNEL_TEXTURE2_H
#define CONV_3D_GEMM_KERNEL_TEXTURE2_H

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
#ifndef CONV_3D_GEMM_KERNEL_TEXTURE2_CALL
#define CONV_3D_GEMM_KERNEL_TEXTURE2_CALL

//=====FH = 3, FW = 3=========================================
#define conv3dGemm_k88W3_tex(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8W3_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k88W3_ic2pow_tex(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8W3_IC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k88W3x4_tex(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8W3x4_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k88W3x4_ic2pow_tex(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8W3x4_IC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

//=====FH = 5, FW = 5=========================================
#define conv3dGemm_k88W5_tex(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8W5_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k88W5_ic2pow_tex(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8W5_IC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k88W5x4_tex(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8W5x4_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k88W5x4_ic2pow_tex(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8W5x4_IC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, oc_index, j_index)

#endif


//=====[FH = FW = 3[===========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8W3_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8W3_TEXTURE

//LB = 4: Size = 0.5625, Time = 0.992803 msec, Performace = 1216.72 GFlop/s
//LB = 3: Size = 0.5625, Time = 1.13212  msec, Performace = 1066.98 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8W3_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int GK = 9 * IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc0 = (oc0 + ((ty >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx].x = W[toc0 + W_k];
	Ws[buf][ty][tx].y = W[toc1 + W_k];
	Ws[buf][ty][tx].z = W[toc2 + W_k];
	Ws[buf][ty][tx].w = W[toc3 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
	Xs[buf][tx][ty] = SaveX4_tex(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx].x = W[toc0 + W_k];
		Ws[buf][ty][tx].y = W[toc1 + W_k];
		Ws[buf][ty][tx].z = W[toc2 + W_k];
		Ws[buf][ty][tx].w = W[toc3 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
		Xs[buf][tx][ty] = SaveX4_tex(X, X_fh, X_fw, IH, IW, xoffset,
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8W3_IC_2POW_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8W3_IC_2POW_TEXTURE

//LB = 4: Size = 0.5625, Time = 0.946127 msec, Performace = 1276.74 GFlop/s
//LB = 3: Size = 0.5625, Time = 1.06252  msec, Performace = 1136.88 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8W3_IC2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc0 = (oc0 + ((ty >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	int OH_OW = OH * OW;
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
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx].x = W[toc0 + W_k];
	Ws[buf][ty][tx].y = W[toc1 + W_k];
	Ws[buf][ty][tx].z = W[toc2 + W_k];
	Ws[buf][ty][tx].w = W[toc3 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	Xs[buf][tx][ty] = SaveX4_tex(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx].x = W[toc0 + W_k];
		Ws[buf][ty][tx].y = W[toc1 + W_k];
		Ws[buf][ty][tx].z = W[toc2 + W_k];
		Ws[buf][ty][tx].w = W[toc3 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		Xs[buf][tx][ty] = SaveX4_tex(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);
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

//------[(OH, OW) % 4 == 0]------------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8W3_X4_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8W3_X4_TEXTURE

//LB = 4: Size = 0.5625, Time = 0.982331 msec, Performace = 1229.69 GFlop/s
//LB = 3: Size = 0.5625, Time = 1.13241  msec, Performace = 1066.72 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8W3x4_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int GK = 9 * IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc1 = (oc0 + ((ty >= STEP) << 2) + 1) * GK;
	W += toc1;//W[toc1, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph;
	tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	int X1 = ((tn0*IH + toh0)*IW + tow1)*IC; //X += X1;
	const int sw_IC = sw * IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx].x = W[W_k - GK];//W0
	Ws[buf][ty][tx].y = W[W_k];
	Ws[buf][ty][tx].z = W[W_k + GK];//W1
	Ws[buf][ty][tx].w = W[W_k + (GK << 1)];//W2

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = X1 + (X_fh * IW + X_fw - Idx)*IC + X_k;
	Xs[buf][tx][ty] = SaveX4x_tex(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
		toh0, tow0, tow1, tow2, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx].x = W[W_k - GK];//W0
		Ws[buf][ty][tx].y = W[W_k];
		Ws[buf][ty][tx].z = W[W_k + GK];//W1
		Ws[buf][ty][tx].w = W[W_k + (GK << 1)];//W2

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = X1 + (X_fh * IW + X_fw - Idx)*IC + X_k;
		Xs[buf][tx][ty] = SaveX4x_tex(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8W3_X4_IC_2POW_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8W3_X4_IC_2POW_TEXTURE

//LB = 4: Size = 0.5625, Time = 0.935223 msec, Performace = 1291.63 GFlop/s
//LB = 3: Size = 0.5625, Time = 1.04369  msec, Performace = 1157.39 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8W3x4_IC2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc1 = (oc0 + ((ty >= STEP) << 2) + 1) * GK;
	W += toc1;//W[toc1, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph;
	tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	int X1 = ((tn0*IH + toh0)*IW + tow1) << LIC; //X += X1;
	const int sw_IC = sw << LIC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx].x = W[W_k - GK];//W0
	Ws[buf][ty][tx].y = W[W_k];
	Ws[buf][ty][tx].z = W[W_k + GK];//W1
	Ws[buf][ty][tx].w = W[W_k + (GK << 1)];//W2

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = X1 + ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	Xs[buf][tx][ty] = SaveX4x_tex(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
		toh0, tow0, tow1, tow2, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx].x = W[W_k - GK];//W0
		Ws[buf][ty][tx].y = W[W_k];
		Ws[buf][ty][tx].z = W[W_k + GK];//W1
		Ws[buf][ty][tx].w = W[W_k + (GK << 1)];//W2

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = X1 + ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		Xs[buf][tx][ty] = SaveX4x_tex(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
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


//=====[FH = FW = 5]===========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8W5_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8W5_TEXTURE

//LB = 4: Size = 1.5625, Time = 2.34493 msec, Performace = 1430.94 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.77691 msec, Performace = 1208.34 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8W5_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int GK = 25 * IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc0 = (oc0 + ((ty >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx].x = W[toc0 + W_k];
	Ws[buf][ty][tx].y = W[toc1 + W_k];
	Ws[buf][ty][tx].z = W[toc2 + W_k];
	Ws[buf][ty][tx].w = W[toc3 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
	Xs[buf][tx][ty] = SaveX4_tex(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx].x = W[toc0 + W_k];
		Ws[buf][ty][tx].y = W[toc1 + W_k];
		Ws[buf][ty][tx].z = W[toc2 + W_k];
		Ws[buf][ty][tx].w = W[toc3 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
		Xs[buf][tx][ty] = SaveX4_tex(X, X_fh, X_fw, IH, IW, xoffset,
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8W5_IC_2POW_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8W5_IC_2POW_TEXTURE

//LB = 4: Size = 1.5625, Time = 2.20457 msec, Performace = 1522.04 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.71572 msec, Performace = 1235.56 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8W5_IC2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int GK = 25 << LIC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc0 = (oc0 + ((ty >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	int OH_OW = OH * OW;
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
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx].x = W[toc0 + W_k];
	Ws[buf][ty][tx].y = W[toc1 + W_k];
	Ws[buf][ty][tx].z = W[toc2 + W_k];
	Ws[buf][ty][tx].w = W[toc3 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	Xs[buf][tx][ty] = SaveX4_tex(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx].x = W[toc0 + W_k];
		Ws[buf][ty][tx].y = W[toc1 + W_k];
		Ws[buf][ty][tx].z = W[toc2 + W_k];
		Ws[buf][ty][tx].w = W[toc3 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		Xs[buf][tx][ty] = SaveX4_tex(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);
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

//------[(OH, OW) % 4 == 0]------------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8W5_X4_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8W5_X4_TEXTURE

//LB = 4: Size = 1.5625, Time = 2.32721 msec, Performace = 1441.83 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.89523 msec, Performace = 1158.95 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8W5x4_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int GK = 25 * IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc1 = (oc0 + ((ty >= STEP) << 2) + 1) * GK;
	W += toc1;//W[toc1, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph;
	tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	int X1 = ((tn0*IH + toh0)*IW + tow1)*IC; //X += X1;
	const int sw_IC = sw * IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx].x = W[W_k - GK];//W0
	Ws[buf][ty][tx].y = W[W_k];
	Ws[buf][ty][tx].z = W[W_k + GK];//W1
	Ws[buf][ty][tx].w = W[W_k + (GK << 1)];//W2

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = X1 + (X_fh * IW + X_fw - Idx)*IC + X_k;
	Xs[buf][tx][ty] = SaveX4x_tex(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
		toh0, tow0, tow1, tow2, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx].x = W[W_k - GK];//W0
		Ws[buf][ty][tx].y = W[W_k];
		Ws[buf][ty][tx].z = W[W_k + GK];//W1
		Ws[buf][ty][tx].w = W[W_k + (GK << 1)];//W2

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = X1 + (X_fh * IW + X_fw - Idx)*IC + X_k;
		Xs[buf][tx][ty] = SaveX4x_tex(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8W5_X4_IC_2POW_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8W5_X4_IC_2POW_TEXTURE

//LB = 4: Size = 1.5625, Time = 2.20919 msec, Performace = 1518.85 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.84148 msec, Performace = 1180.88 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8W5x4_IC2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int GK = 25 << LIC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc1 = (oc0 + ((ty >= STEP) << 2) + 1) * GK;
	W += toc1;//W[toc1, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph;
	tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	int X1 = ((tn0*IH + toh0)*IW + tow1) << LIC; //X += X1;
	const int sw_IC = sw << LIC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx].x = W[W_k - GK];//W0
	Ws[buf][ty][tx].y = W[W_k];
	Ws[buf][ty][tx].z = W[W_k + GK];//W1
	Ws[buf][ty][tx].w = W[W_k + (GK << 1)];//W2

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = X1 + ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	Xs[buf][tx][ty] = SaveX4x_tex(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
		toh0, tow0, tow1, tow2, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx].x = W[W_k - GK];//W0
		Ws[buf][ty][tx].y = W[W_k];
		Ws[buf][ty][tx].z = W[W_k + GK];//W1
		Ws[buf][ty][tx].w = W[W_k + (GK << 1)];//W2

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = X1 + ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		Xs[buf][tx][ty] = SaveX4x_tex(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
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

#endif