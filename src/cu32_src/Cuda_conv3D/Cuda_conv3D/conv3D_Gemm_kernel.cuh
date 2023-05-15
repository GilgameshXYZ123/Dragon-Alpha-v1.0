#pragma once

#ifndef CONV_3D_GEMM_KERNEL_H
#define CONV_3D_GEMM_KERNEL_H

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
#ifndef CONV_3D_GEMM_KERNEL_CALL
#define CONV_3D_GEMM_KERNEL_CALL

//LB = log2(BLOCK_SIZE)

#define conv3dGemm_k88x4(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8x4<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dGemm_k88(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dGemm_k44(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//======[Direct Conv]=========================================
#define conv3dPure_k84(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_8_4<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_k48(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_4_8<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_k44(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_4_4<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_k82(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_8_2<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_k28(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_2_8<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>1, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_k42(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_4_2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_k24(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_2_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//======[Small]===============================================
#define conv3dGemm_k22(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_2_2<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw, ph,pw,\
			oc_index, j_index)

//------------------------------------------------------------
#define conv3dGemm_k41(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k21(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_2_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//------------------------------------------------------------
#define conv3dGemm_k14(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_1_4<LB, (1<<LB)>\
		<<< dim3(GN>>LB, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dGemm_k12(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_1_2<LB, (1<<LB)>\
		<<< dim3(GN>>LB, GM>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dGemm_k11(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_1_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#endif


//======[Common]==============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, (OH, OW) % 4 == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8X4
#define CONV_3D_GEMM_KERNEL_8_8X4

//LB = 4: Size = 1, Time = 1.6475 msec, Performace = 1303.48 GFlop/s
//LB = 3: Size = 1, Time = 1.9929 msec, Performace = 1077.57 GFlop/s
//LB = 4: Size = 1.125, Time = 1.82218 msec, Performace = 1325.84 GFlop/s
//LB = 3: Size = 1.125, Time = 2.25114 msec, Performace = 1073.2 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8x4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc1 = (oc0 + ((ty >= STEP) << 2) + 1) * GK;
	W += toc1;//W[toc1, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;
	const int sw_IC = sw * IC;

	//load 4 elements from W[OC, FH, FW, IC]
	float4 wv; int W_k = ty - ((ty >= STEP) << LB >> 1);
	wv.x = W[W_k - GK];//W0
	wv.y = W[W_k];//W1
	wv.z = W[W_k + GK];//W2
	wv.w = W[W_k + (GK << 1)];//W3
	Ws[buf][ty][tx] = wv;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	Xs[buf][tx][ty] = SaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
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
		float4 wv; int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		wv.x = W[W_k - GK];//W0
		wv.y = W[W_k];//W1
		wv.z = W[W_k + GK];//W2
		wv.w = W[W_k + (GK << 1)];//W3
		Ws[buf][ty][tx] = wv;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
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
#ifndef CONV_3D_GEMM_KERNEL_8_8
#define CONV_3D_GEMM_KERNEL_8_8

//LB = 4: Size = 1, Time = 1.71798 msec, Performace = 1250.01 GFlop/s
//LB = 3: Size = 1, Time = 1.998   msec, Performace = 1074.82 GFlop/s
//LB = 4: Size = 1.125, Time = 1.90045 msec, Performace = 1271.24 GFlop/s
//LB = 3: Size = 1.125, Time = 2.28429 msec, Performace = 1057.62 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

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
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	float4 wv; int W_k = ty - ((ty >= STEP) << LB >> 1);
	wv.x = W[toc0 + W_k];
	wv.y = W[toc1 + W_k];
	wv.z = W[toc2 + W_k];
	wv.w = W[toc3 + W_k];
	Ws[buf][ty][tx] = wv;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1); 
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset, 
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0),  v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0),  v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0),  v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0),  v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0),  v9 = make_float4(0, 0, 0, 0);
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
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_4
#define CONV_3D_GEMM_KERNEL_4_4

//LB = 4: Size = 1, Time = 2.54592 msec, Performace = 843.499 GFlop/s
//LB = 3: Size = 1, Time = 4.0012  msec, Performace = 536.71 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	int toc0 = (((tx & 1) << 1) + oc0)*GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw; 
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;

	//load 2 elements from W[OC, FH, FW, IC]
	float2 wv; int W_k = tx >> 1;
	wv.x = W[toc0 + W_k];
	wv.y = W[toc1 + W_k];
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = wv;

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, b.x, a);
			simdMM4(v1, b.y, a);
			simdMM4(v2, b.z, a);
			simdMM4(v3, b.w, a);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		wv.x = W[toc0 + W_k];
		wv.y = W[toc1 + W_k];
		Ws[buf][Ws_x][Ws_y] = wv;

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
		Xs[buf][Xs_y][Xs_x] = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

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


//======[Direct Conv]=========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_8_4
#define CONV_3D_PURE_KERNEL_8_4

//LB = 4: Size = 1, Time = 2.01824 msec, Performace = 1064.04 GFlop/s
//LB = 3: Size = 1, Time = 2.32371 msec, Performace =  924.161 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_8_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	const int GK = FH * FW * IC;
	const int toc0 = (oc0 + ((ty >= STEP) << 2)) * GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);

	const int Xs_x = (tx >> 1), Xs_y = (ty << 1) + (tx & 1);
	const int OIC = (IC << 1 >> LB), SX = (IW - FW)*IC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, W += IC)
		{
			//load 4 elem from W
			float4 wv; int Wic = ty - ((ty >= STEP) << LB >> 1);
			wv.x = W[toc0 + Wic];
			wv.y = W[toc1 + Wic];
			wv.z = W[toc2 + Wic];
			wv.w = W[toc3 + Wic];
			Ws[buf][ty][tx] = wv;

			//load 2 elem from X
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			float2 xv; int Xic = tx >> 1;
			xv.x = (lx0 ? X[X0 + Xic] : 0);
			xv.y = (lx1 ? X[X1 + Xic] : 0);
			Xs[buf][Xs_x][Xs_y] = xv;
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++) 
				{
					float4 b0 = *(float4*)(&Xs[buf][ik][ty << 1]);
					float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

					simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
					simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
					simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
					simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
				}
				buf ^= 1;

				//load 4 elem from W
				float4 wv; int Wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
				wv.x = W[toc0 + Wic];
				wv.y = W[toc1 + Wic];
				wv.z = W[toc2 + Wic];
				wv.w = W[toc3 + Wic];
				Ws[buf][ty][tx] = wv;

				//load 2 elem from X
				float2 xv; int Xic = ((oic << LB) + tx) >> 1;
				xv.x = (lx0 ? X[X0 + Xic] : 0);
				xv.y = (lx1 ? X[X1 + Xic] : 0);
				Xs[buf][Xs_x][Xs_y] = xv;
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) 
			{
				float4 b0 = *(float4*)(&Xs[buf][ik][ty << 1]);
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0; //oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_4_8
#define CONV_3D_PURE_KERNEL_4_8

//LB = 4: Size = 1, Time = 1.85052 msec, Performace = 1160.48  GFlop/s
//LB = 3: Size = 1, Time = 2.38262 msec, Performace =  901.312 GFlop/s
//[OC = 64]:
//LB = 4: Size = 1.125, Time = 2.3151  msec, Performace = 1043.55  GFlop/s
//LB = 3: Size = 1.125, Time = 3.12201 msec, Performace =  773.835 GFlop/s
//[OC = 32]
//LB = 3: Size = 1.125, Time = 3.09771 msec, Performace = 779.905 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_4_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	const int GK = FH * FW * IC;
	const int toc0 = (((ty & 1) << 1) + oc0) * GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int tihs2 = toh2 * sh - ph, tiws2 = tow2 * sw - pw;
	const int tihs3 = toh3 * sh - ph, tiws3 = tow3 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;
	const int X2 = (((tn2 *IH) + tihs2) * IW + tiws2) * IC;
	const int X3 = (((tn3 *IH) + tihs3) * IW + tiws3) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0);
	float4 v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0);
	float4 v7 = make_float4(0, 0, 0, 0);

	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, W += IC)
		{
			//load 2 elem from W
			float2 wv; int Wic = ty >> 1;
			wv.x = W[toc0 + Wic];
			wv.y = W[toc1 + Wic];
			Ws[buf][Ws_y][Ws_x] = wv;

			//load 4 elem from X
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			bool lx2 = LOAD_X(tihs2, tiws2, fh, fw);
			bool lx3 = LOAD_X(tihs3, tiws3, fh, fw);
			float4 xv; int Xic = tx - ((tx >= STEP) << LB >> 1);
			xv.x = (lx0 ? X[X0 + Xic] : 0);
			xv.y = (lx1 ? X[X1 + Xic] : 0);
			xv.z = (lx2 ? X[X2 + Xic] : 0);
			xv.w = (lx3 ? X[X3 + Xic] : 0);
			Xs[buf][tx][ty] = xv;
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
					float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]);

					simdMM4(v0, b0.x, a0); 
					simdMM4(v1, b0.y, a0);
					simdMM4(v2, b0.z, a0);
					simdMM4(v3, b0.w, a0);
					simdMM4(v4, b1.x, a0);
					simdMM4(v5, b1.y, a0); 
					simdMM4(v6, b1.z, a0);
					simdMM4(v7, b1.w, a0);
				}
				buf ^= 1;

				//load 2 elem from W
				float2 wv; int Wic = ((oic << LB) + ty) >> 1;
				wv.x = W[toc0 + Wic];
				wv.y = W[toc1 + Wic];
				Ws[buf][Ws_y][Ws_x] = wv;
				
				//load 2 elem from X
				float4 xv; int Xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
				xv.x = (lx0 ? X[X0 + Xic] : 0);
				xv.y = (lx1 ? X[X1 + Xic] : 0);
				xv.z = (lx2 ? X[X2 + Xic] : 0);
				xv.w = (lx3 ? X[X3 + Xic] : 0);
				Xs[buf][tx][ty] = xv;
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
				float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]);

				simdMM4(v0, b0.x, a0);
				simdMM4(v1, b0.y, a0);
				simdMM4(v2, b0.z, a0);
				simdMM4(v3, b0.w, a0);
				simdMM4(v4, b1.x, a0);
				simdMM4(v5, b1.y, a0);
				simdMM4(v6, b1.z, a0);
				simdMM4(v7, b1.w, a0);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0; //oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4;
	*(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6;
	*(float4*)(Y + j7) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_4_4
#define CONV_3D_PURE_KERNEL_4_4

//LB = 4: Size = 1, Time = 2.25289 msec, Performace = 953.214 GFlop/s
//LB = 3: Size = 1, Time = 3.1469  msec, Performace = 682.413 GFlop/s
//[OC = 64]:
//LB = 4: Size = 1.125, Time = 2.64793 msec, Performace = 912.381 GFlop/s
//LB = 3: Size = 1.125, Time = 3.73953 msec, Performace = 646.049 GFlop/s
//[OC = 32]
//LB = 3: Size = 1.125, Time = 3.76488 msec, Performace = 641.7 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_4_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	const int GK = FH * FW * IC;
	const int toc0 = (((ty & 1) << 1) + oc0) * GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);

	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	const int Xs_x = (tx >> 1), Xs_y = (ty << 1) + (tx & 1);
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, W += IC)
		{
			//load 2 elem from W
			float2 wv; int Wic = ty >> 1;
			wv.x = W[toc0 + Wic];
			wv.y = W[toc1 + Wic];
			Ws[buf][Ws_y][Ws_x] = wv;

			//load 2 elem from X
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			float2 xv; int Xic = tx >> 1;
			xv.x = (lx0 ? X[X0 + Xic] : 0);
			xv.y = (lx1 ? X[X1 + Xic] : 0);
			Xs[buf][Xs_x][Xs_y] = xv;
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++) 
				{
					float4 b = *(float4*)(&Xs[buf][ik][ty << 1]);
					float4 a = *(float4*)(&Ws[buf][ik][tx << 1]);

					simdMM4(v0, b.x, a);
					simdMM4(v1, b.y, a);
					simdMM4(v2, b.z, a);
					simdMM4(v3, b.w, a);
				}
				buf ^= 1;

				//load 2 elem from W
				float2 wv; int Wic = ((oic << LB) + ty) >> 1;
				wv.x = W[toc0 + Wic];
				wv.y = W[toc1 + Wic];
				Ws[buf][Ws_y][Ws_x] = wv;

				//load 2 elem from X
				float2 xv; int Xic = ((oic << LB) + tx) >> 1;
				xv.x = (lx0 ? X[X0 + Xic] : 0);
				xv.y = (lx1 ? X[X1 + Xic] : 0);
				Xs[buf][Xs_x][Xs_y] = xv;
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) 
			{
				float4 b = *(float4*)(&Xs[buf][ik][ty << 1]);
				float4 a = *(float4*)(&Ws[buf][ik][tx << 1]);

				simdMM4(v0, b.x, a);
				simdMM4(v1, b.y, a);
				simdMM4(v2, b.z, a);
				simdMM4(v3, b.w, a);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0; //oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_8_2
#define CONV_3D_PURE_KERNEL_8_2

//LB = 4: Size = 1, Time = 2.81343 msec, Performace = 763.297 GFlop/s
//LB = 3: Size = 1, Time = 3.21861 msec, Performace = 667.209 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_8_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	const int GK = FH * FW * IC;
	const int toc0 = (oc0 + ((ty >= STEP) << 2)) * GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index;
	int tj0 = j0 + (tx & 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);

	const int Xs_x = (tx >> 1), Xs_y = (ty << 1) + (tx & 1);
	const int OIC = (IC << 1 >> LB), SX = (IW - FW)*IC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, W += IC)
		{
			//load 4 elem from W
			float4 wv; int Wic = ty - ((ty >= STEP) << LB >> 1);
			wv.x = W[toc0 + Wic];
			wv.y = W[toc1 + Wic];
			wv.z = W[toc2 + Wic];
			wv.w = W[toc3 + Wic];
			Ws[buf][ty][tx] = wv;

			//load 1 elem from X
			int Xic = tx >> 1; bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			Xs[buf][Xs_x][Xs_y] = (lx0 ? X[X0 + Xic] : 0);
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float2 b0 = *(float2*)(&Xs[buf][ik][ty << 1]);
					float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
					simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
					simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				}
				buf ^= 1;

				//load 4 elem from W
				float4 wv; int Wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
				wv.x = W[toc0 + Wic];
				wv.y = W[toc1 + Wic];
				wv.z = W[toc2 + Wic];
				wv.w = W[toc3 + Wic];
				Ws[buf][ty][tx] = wv;

				//load 1 elem from X
				int Xic = ((oic << LB) + tx) >> 1;
				Xs[buf][Xs_x][Xs_y] = (lx0 ? X[X0 + Xic] : 0);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float2 b0 = *(float2*)(&Xs[buf][ik][ty << 1]);
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0; //oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_2_8
#define CONV_3D_PURE_KERNEL_2_8

//LB = 4: Size = 1.125, Time = 2.80256 msec, Performace = 862.039 GFlop/s
//LB = 3: Size = 1.125, Time = 3.78655 msec, Performace = 638.026 GFlop/s
//[OC = 64, N*2]:
//LB = 4: Size = 1.125, Time = 2.82779 msec, Performace = 854.349 GFlop/s
//LB = 3: Size = 1.125, Time = 4.24681 msec, Performace = 568.879 GFlop/s
//[OC = 32, N*4]:
//LB = 4: Size = 1.125, Time = 2.92862 msec, Performace = 824.934 GFlop/s
//LB = 3: Size = 1.125, Time = 4.08606 msec, Performace = 591.258 GFlop/s
//[OC = 16, N*8]
//LB = 3: Size = 1.125, Time = 4.70802 msec, Performace = 513.149 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_2_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 1) + oc_index;
	const int GK = FH * FW * IC;
	W += (oc0 + (ty & 1)) * GK;//W[toc0, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int tihs2 = toh2 * sh - ph, tiws2 = tow2 * sw - pw;
	const int tihs3 = toh3 * sh - ph, tiws3 = tow3 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;
	const int X2 = (((tn2 *IH) + tihs2) * IW + tiws2) * IC;
	const int X3 = (((tn3 *IH) + tihs3) * IW + tiws3) * IC;

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	float2 v4 = make_float2(0, 0);
	float2 v5 = make_float2(0, 0);
	float2 v6 = make_float2(0, 0);
	float2 v7 = make_float2(0, 0);

	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, W += IC)
		{
			//load 4 elem from X
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			bool lx2 = LOAD_X(tihs2, tiws2, fh, fw);
			bool lx3 = LOAD_X(tihs3, tiws3, fh, fw);
			float4 x; int Xic = tx - ((tx >= STEP) << LB >> 1);
			x.x = (lx0 ? X[X0 + Xic] : 0);
			x.y = (lx1 ? X[X1 + Xic] : 0);
			x.z = (lx2 ? X[X2 + Xic] : 0);
			x.w = (lx3 ? X[X3 + Xic] : 0);
			Xs[buf][tx][ty] = x;

			//load 1 elem from W
			int Wic = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = W[Wic];
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
					float2 a0 = *(float2*)(&Ws[buf][ik][tx << 1]);

					simdMM2(v0, b0.x, a0);
					simdMM2(v1, b0.y, a0);
					simdMM2(v2, b0.z, a0);
					simdMM2(v3, b0.w, a0);
					simdMM2(v4, b1.x, a0);
					simdMM2(v5, b1.y, a0);
					simdMM2(v6, b1.z, a0);
					simdMM2(v7, b1.w, a0);
				}
				buf ^= 1;

				//load 2 elem from X
				float4 x; int Xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
				x.x = (lx0 ? X[X0 + Xic] : 0);
				x.y = (lx1 ? X[X1 + Xic] : 0);
				x.z = (lx2 ? X[X2 + Xic] : 0);
				x.w = (lx3 ? X[X3 + Xic] : 0);
				Xs[buf][tx][ty] = x;

				//load 1 elem from W
				int Wic = ((oic << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = W[Wic];
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
				float2 a0 = *(float2*)(&Ws[buf][ik][tx << 1]);

				simdMM2(v0, b0.x, a0);
				simdMM2(v1, b0.y, a0);
				simdMM2(v2, b0.z, a0);
				simdMM2(v3, b0.w, a0);
				simdMM2(v4, b1.x, a0);
				simdMM2(v5, b1.y, a0);
				simdMM2(v6, b1.z, a0);
				simdMM2(v7, b1.w, a0);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
	*(float2*)(Y + j2) = v2;
	*(float2*)(Y + j3) = v3;
	*(float2*)(Y + j4) = v4;
	*(float2*)(Y + j5) = v5;
	*(float2*)(Y + j6) = v6;
	*(float2*)(Y + j7) = v7;
}

#endif
	

//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_4_2
#define CONV_3D_PURE_KERNEL_4_2

//LB = 4: Size = 1, Time = 2.8471  msec, Performace = 754.27  GFlop/s
//LB = 3: Size = 1, Time = 4.16679 msec, Performace = 515.381 GFlop/s
//LB = 4: Size = 1.125, Time = 3.25056 msec, Performace = 743.233 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_4_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int GK = FH * FW * IC;
	const int toc0 = (((tx & 1) << 1) + oc0) * GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int tj0 = j0 + (ty & 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tihs0 = toh0 * sh - ph;
	const int tiws0 = tow0 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);

	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, W += IC)
		{
			//load 2 elem from W
			float2 wv; int Wic = tx >> 1;
			wv.x = W[toc0 + Wic];
			wv.y = W[toc1 + Wic];
			Ws[buf][Ws_x][Ws_y] = wv;

			//load 1 elem from X
			int Xic = ty >> 1;
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			Xs[buf][Xs_y][Xs_x] = (lx0 ? X[X0 + Xic] : 0);
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++) {
					float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
					float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);
					simdMM4(v0, b.x, a);
					simdMM4(v1, b.y, a);
				}
				buf ^= 1;

				//load 2 elem from W
				float2 wv; int Wic = ((oic << LB) + tx) >> 1;
				wv.x = W[toc0 + Wic];
				wv.y = W[toc1 + Wic];
				Ws[buf][Ws_x][Ws_y] = wv;

				//load 2 elem from X
				int Xic = ((oic << LB) + ty) >> 1;
				Xs[buf][Xs_y][Xs_x] = (lx0 ? X[X0 + Xic] : 0);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
				float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);
				simdMM4(v0, b.x, a);
				simdMM4(v1, b.y, a);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC;
	
	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_2_4
#define CONV_3D_PURE_KERNEL_2_4

//LB = 4: Size = 1, Time = 3.35356 msec, Performace = 640.359 GFlop/s
//LB = 3: Size = 1, Time = 5.58619 msec, Performace = 384.427 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_2_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int GK = FH * FW * IC;
	const int toc0 = ((tx & 1) + oc0) * GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int Xoffset0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int Xoffset1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);

	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, W += IC)
		{
			//load 1 elem from W
			int Wic = tx >> 1;
			Ws[buf][Ws_x][Ws_y] = W[toc0 + Wic];

			//load 2 elem from X
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw); 
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			float2 xv; int Xic = ty >> 1;
			xv.x = (lx0 ? X[Xoffset0 + Xic] : 0);
			xv.y = (lx1 ? X[Xoffset1 + Xic] : 0);
			Xs[buf][Xs_y][Xs_x] = xv;
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++) {
					float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
					float2 a = *(float2*)(&Ws[buf][ik][ty << 1]);

					simdMM2(v0, b.x, a);
					simdMM2(v1, b.y, a);
					simdMM2(v2, b.z, a);
					simdMM2(v3, b.w, a);
				}
				buf ^= 1;

				//load 1 elem from W
				int Wic = ((oic << LB) + tx) >> 1;
				Ws[buf][Ws_x][Ws_y] = W[toc0 + Wic];

				//load 2 elem from X
				float2 xv; int Xic = ((oic << LB) + ty) >> 1;
				xv.x = (lx0 ? X[Xoffset0 + Xic] : 0);
				xv.y = (lx1 ? X[Xoffset1 + Xic] : 0);
				Xs[buf][Xs_y][Xs_x] = xv;
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
				float2 a = *(float2*)(&Ws[buf][ik][ty << 1]);

				simdMM2(v0, b.x, a);
				simdMM2(v1, b.y, a);
				simdMM2(v2, b.z, a);
				simdMM2(v3, b.w, a);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0; //oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
	*(float2*)(Y + j2) = v2;
	*(float2*)(Y + j3) = v3;
}

#endif


//======[Small]===============================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef CONV_3D_GEMM_KERNEL_2_2
#define CONV_3D_GEMM_KERNEL_2_2

//LB = 4: Size = 1, Time = 4.19393 msec, Performace = 512.046 GFlop/s
//LB = 3: Size = 1, Time = 6.23865 msec, Performace = 344.223 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_2_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int toc0 = oc0 * GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	const int toh0 = oh0 * sh - ph, tow0 = ow0 * sw - pw;
	const int toh1 = oh1 * sh - ph, tow1 = ow1 * sw - pw;
	const int X0 = ((n0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((n1*IH + toh1)*IW + tow1)*IC;

	//load 2 elements from W[OC, FH, FW, IC]
	float2 wv; int W_k = tx;
	wv.x = W[toc0 + W_k];
	wv.y = W[toc1 + W_k];
	Ws[buf][tx][ty] = wv;

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty;
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	Xs[buf][ty][tx] = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1);
	__syncthreads();

	//compute area-----------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float2 a = Ws[buf][ik][ty];
			simdMM2(v0, b.x, a);
			simdMM2(v1, b.y, a);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		float2 wv; int W_k = (ok << LB) + tx;
		wv.x = W[toc0 + W_k];
		wv.y = W[toc1 + W_k];
		Ws[buf][tx][ty] = wv;

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty; 
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][ty][tx] = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Xs[buf][ik][tx];
		float2 a = Ws[buf][ik][ty];
		simdMM2(v0, b.x, a);
		simdMM2(v1, b.y, a);
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float2 a;//load 2 elements from W
		a.x = W[toc0 + k];
		a.y = W[toc1 + k];

		//load 2 elements from X
		int X_k = k, X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		float2 b = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1);

		simdMM2(v0, b.x, a);
		simdMM2(v1, b.y, a);
	}
	//when GK % STEP != 0---------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	j1 = j0 + OC;

	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
}

#endif


//------------------------------------------------------------
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef CONV_3D_GEMM_KERNEL_4_1
#define CONV_3D_GEMM_KERNEL_4_1

//LB = 4: Size = 1, Time = 4.16  msec, Performace = 516.222 GFlop/s
//LB = 3: Size = 1, Time = 5.664 msec, Performace = 379.146 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float   Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	int toc0 = oc0 * GK, toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	const int toh0 = oh0 * sh - ph;
	const int tow0 = ow0 * sw - pw;
	const int Xoffset0 = ((n0*IH + toh0)*IW + tow0)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];
	Ws[buf][tx][ty].z = W[toc2 + W_k];
	Ws[buf][tx][ty].w = W[toc3 + W_k];

	//load 1 element from X[N, IH, IW, IC]
	int X_k = ty, X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
	__syncthreads();
	
	//compute area----------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Xs[buf][ik][tx];
			float4 a = Ws[buf][ik][ty];
			simdMM4(v, b, a);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];
		Ws[buf][tx][ty].z = W[toc2 + W_k];
		Ws[buf][tx][ty].w = W[toc3 + W_k];

		//load 1 element from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty, X_fh, X_fw; 
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  b = Xs[buf][ik][tx];
		float4 a = Ws[buf][ik][ty];
		simdMM4(v, b, a);
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float4 a;//load 4 elements from W
		a.x = W[toc0 + k];
		a.y = W[toc1 + k];
		a.z = W[toc2 + k];
		a.w = W[toc3 + k];

		float b;//load 1 element from X
		int X_k = k, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b = (lx0 ? X[Xoffset0 + xoffset] : 0);

		simdMM4(v, b, a);
	}
	//when GK % STEP != 0---------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	*(float4*)(Y + j0) = v;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
#ifndef CONV_3D_GEMM_KERNEL_2_1
#define CONV_3D_GEMM_KERNEL_2_1

//LB=4: Size = 1, Time = 6.286 msec, Performace = 341.63 GFlop/s
//LB=3: Size = 1, Time = 9.032 msec, Performace = 237.764 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_2_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int toc0 = oc0 * GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	const int toh0 = oh0 * sh - ph;
	const int tow0 = ow0 * sw - pw;
	const int Xoffset0 = ((n0*IH + toh0)*IW + tow0)*IC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty].x = W[toc0 + W_k]; 
	Ws[buf][tx][ty].y = W[toc1 + W_k]; 

	//load 1 element from X[N, IH, IW, IC]
	int X_k = ty, X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
	__syncthreads();
	
	//compute area-------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//1<<LB = STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Xs[buf][ik][tx];
			float2 a = Ws[buf][ik][ty];
			simdMM2(v, b, a)
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];

		//load 1 element from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty; get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  b = Xs[buf][ik][tx];
		float2 a = Ws[buf][ik][ty];
		simdMM2(v, b, a)
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float2 a;//load 2 elements from W
		a.x = W[toc0 + k];
		a.y = W[toc1 + k];

		float b;//load 1 element from X
		int X_k = k, X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b = (lx0 ? X[Xoffset0 + xoffset] : 0);

		simdMM2(v, b, a);
	}
	//when GK % STEP != 0---------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	*(float2*)(Y + j0) = v;
}

#endif


//------------------------------------------------------------
//(Y: BLOCK_SIZE*1£¬X: BLOCK_SIZE*4), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef CONV_3D_GEMM_KERNEL_1_4
#define CONV_3D_GEMM_KERNEL_1_4

//(GN, GM, GK) = (128, 8192, 512)
//LB = 4: Size = 1, Time = 6.72 msec , Performace = 319.566 GFlop/s
//LB = 3: Size = 1, Time = 7.79  msec, Performace = 275.672 GFlop/s
//LB = 3: Size = 0.28125, Time = 2.17512 msec, Performace = 277.676 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_1_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	int oc0 = ((blockIdx.x << LB) + tx) + oc_index;
	W += oc0 * GK;//W[oc0, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
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
	int X_k = tx; 
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);

	//load 1 element from W[OC, FH, FW, IC]
	int W_k = ty;
	Ws[buf][ty][tx] = W[W_k];
	__syncthreads();

	//compute area------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = Xs[buf][ik][ty];
			float  a = Ws[buf][ik][tx];
			simdMM4(v, a, b);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + tx;
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + ty;
		Ws[buf][ty][tx] = W[W_k];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b = Xs[buf][ik][ty];
		float  a = Ws[buf][ik][tx];
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


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
#ifndef CONV_3D_GEMM_KERNEL_1_2
#define CONV_3D_GEMM_KERNEL_1_2

//LB = 4: Size = 1, Time =  7.682 msec, Performace = 279.547 GFlop/s
//LB = 3: Size = 1, Time = 10.024 msec, Performace = 214.234 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_1_2(
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
	const int oc0 = ((blockIdx.x << LB) + tx) + oc_index;
	const int toc0 = oc0 * GK;
	
	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index, j1 = j0 + 1;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	const int toh0 = oh0 * sh - ph, tow0 = ow0 * sw - pw;
	const int toh1 = oh1 * sh - ph, tow1 = ow1 * sw - pw;
	const int X0 = ((n0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((n1*IH + toh1)*IW + tow1)*IC;

	//load 1 element from W[OC, FH, FW, IC]
	int W_k = ty;
	Ws[buf][ty][tx] = W[toc0 + W_k];

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = tx;
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	const int IW_IC = IW * IC;
	float2 xv; int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	xv.x = (LOAD_X(toh0, tow0, X_fh, X_fw) ? X[X0 + xoffset] : 0);
	xv.y = (LOAD_X(toh1, tow1, X_fh, X_fw) ? X[X1 + xoffset] : 0);
	Xs[buf][tx][ty] = xv;
	__syncthreads();

	//compute area-------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][ty];
			float  a = Ws[buf][ik][tx];
			simdMM2(v, a, b);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + ty;
		Ws[buf][ty][tx] = W[toc0 + W_k];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + tx;
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		float2 xv; int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		xv.x = (LOAD_X(toh0, tow0, X_fh, X_fw) ? X[X0 + xoffset] : 0);
		xv.y = (LOAD_X(toh1, tow1, X_fh, X_fw) ? X[X1 + xoffset] : 0);
		Xs[buf][tx][ty] = xv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Xs[buf][ik][ty];
		float  a = Ws[buf][ik][tx];
		simdMM2(v, a, b);
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float a = W[toc0 + k];//load 1 element from W

		int X_k = k;//load 2 elements from X
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		float2 b; int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b.x = (LOAD_X(toh0, tow0, X_fh, X_fw) ? X[X0 + xoffset] : 0);
		b.y = (LOAD_X(toh1, tow1, X_fh, X_fw) ? X[X1 + xoffset] : 0);

		simdMM2(v, a, b);
	}
	//when GK % STEP != 0---------------------------------------
	
	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	j1 = j0 + OC;

	Y[j0] = v.x;
	Y[j1] = v.y;
}

#endif


//LB = 4, GK >= 16
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
#ifndef CONV_3D_GEMM_KERNEL_1_1
#define CONV_3D_GEMM_KERNEL_1_1

//LB = 4: Size = 0.710273, Time = 8.012 msec, Performace = 190.377 GFlop/s
//LB = 3: Size = 0.5, Time = 8.14  msec, Performace = 131.909 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_1_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	const int toc0 = oc0 * GK;

	//prepare for GM = N * OH * OW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	const int toh0 = oh0 * sh - ph, tow0 = ow0 * sw - pw;
	const int Xoffset0 = ((n0*IH + toh0)*IW + tow0)*IC;

	//load 1 element from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty] = W[toc0 + W_k];

	//load 1 element from X[N, IH, IW, IC]
	int X_k = ty, X_fh, X_fw; 
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
	__syncthreads();

	//compute area----------------------------------------------------
	float v = 0;
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float b = Xs[buf][ik][tx];
			float a = Ws[buf][ik][ty];
			v += a * b;
		}

		buf ^= 1;
		//load 1 element from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[toc0 + W_k];

		//load 1 element from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float b = Xs[buf][ik][tx];
		float a = Ws[buf][ik][ty];
		v += a * b;
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		//load 1 element from W
		float a = W[toc0 + k];

		float b;//load 1 element from X
		int X_k = k, X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b = (lx0 ? X[Xoffset0 + xoffset] : 0);

		v += a * b;
	}
	//when GK % STEP != 0---------------------------------------

	Y[j0*OC + oc0] = v;
}

#endif

#endif
