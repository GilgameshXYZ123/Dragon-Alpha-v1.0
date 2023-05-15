#pragma once

#ifndef CONV_3D_GEMMR_KERNEL_H
#define CONV_3D_GEMMR_KERNEL_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef CONV_3D_GEMMR_KERNEL_CALL
#define CONV_3D_GEMMR_KERNEL_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]==============================================
#define conv3dGemm_k88R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8R<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#define conv3dGemm_k88R4(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8R4<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#define conv3dGemm_k44R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_4R<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//======[Pure: Direct Conv]===================================
#define conv3dPure_k84R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_8_4R<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_k48R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_4_8R<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_k44R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_4_4R<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_k82R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_8_2R<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_k42R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_4_2R<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#endif


//======[Common]==============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8R
#define CONV_3D_GEMM_KERNEL_8_8R

//LB = 4: Size = 1, Time = 1.52632 msec, Performace = 1406.97 GFlop/s
//LB = 3: Size = 1, Time = 1.87689 msec, Performace = 1144.17 GFlop/s
//LB = 4: Size = 1.125, Time = 1.73012 msec, Performace = 1396.39 GFlop/s
//LB = 3: Size = 1.125, Time = 2.05591 msec, Performace = 1175.11 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
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
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]
	
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

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1,
		X2, toh2, tow2,
		X3, toh3, tow3);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
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
	
		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][tx][ty] = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, (OH, OW) % 4 == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8R4
#define CONV_3D_GEMM_KERNEL_8_8R4

//LB = 4: Size = 1, Time = 1.50435 msec, Performace = 1427.52 GFlop/s
//LB = 3: Size = 1, Time = 1.85301 msec, Performace = 1158.92 GFlop/s
//LB = 4: Size = 1.125, Time = 1.7168  msec, Performace = 1407.23 GFlop/s
//LB = 3: Size = 1.125, Time = 2.04266 msec, Performace = 1182.73 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8R4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
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
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;
	const int sw_IC = sw * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	Xs[buf][tx][ty] = SaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
		toh0, tow0, tow1, tow2, tow3);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
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

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][tx][ty] = SaveX4x(X, X_fh, X_fw, IH, IW, xoffset, sw_IC,
			toh0, tow0, tow1, tow2, tow3);

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * OC);
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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_4R
#define CONV_3D_GEMM_KERNEL_4_4R

//LB = 4: Size = 1, Time = 2.40098 msec, Performace = 894.42  GFlop/s
//LB = 3: Size = 1, Time = 3.37053 msec, Performace = 637.134 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_4R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += ((tx & 1) << 1) + oc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
		X0, toh0, tow0,
		X1, toh1, tow1);

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
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

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][Xs_y][Xs_x] = SaveX2(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1);

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y] = *(float2*)(CW + W_k * OC);
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
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
}

#endif


//======[Pure: Direct Conv]===================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_8_4R
#define CONV_3D_PURE_KERNEL_8_4R

//LB = 4: Size = 1, Time = 1.83531 msec, Performace = 1170.09 GFlop/s
//LB = 3: Size = 1, Time = 2.02446 msec, Performace = 1060.77 GFlop/s
//LB = 4: Size = 1.125, Time = 2.10696 msec, Performace = 1146.64 GFlop/s
//LB = 3: Size = 1.125, Time = 2.30691 msec, Performace = 1047.25 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_8_4R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
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
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

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
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC, SCW = IC * OC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, CW += SCW)
		{
			//load 2 elem from X
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			float2 x; int Xic = tx >> 1;
			x.x = (lx0 ? X[X0 + Xic] : 0);
			x.y = (lx1 ? X[X1 + Xic] : 0);
			Xs[buf][Xs_x][Xs_y] = x;

			//load 4 elem from W
			int Wic = ty - ((ty >= STEP) << LB >> 1);
			Ws[buf][ty][tx] = *(float4*)(CW + Wic * OC);
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

				//load 2 elem from X
				float2 x; int Xic = ((oic << LB) + tx) >> 1;
				x.x = (lx0 ? X[X0 + Xic] : 0);
				x.y = (lx1 ? X[X1 + Xic] : 0);
				Xs[buf][Xs_x][Xs_y] = x;

				//load 4 elem from W
				int Wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
				Ws[buf][ty][tx] = *(float4*)(CW + Wic * OC);
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

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_4_8R
#define CONV_3D_PURE_KERNEL_4_8R

//LB = 4: Size = 1, Time = 1.87628 msec, Performace = 1144.54 GFlop/s
//LB = 3: Size = 1, Time = 2.10694 msec, Performace = 1019.25 GFlop/s
//LB = 4: Size = 1.125, Time = 2.02514 msec, Performace = 1192.96 GFlop/s
//LB = 3: Size = 1.125, Time = 2.35402 msec, Performace = 1026.3 GFlop/s
//[OC = 64]:
//LB = 4: Size = 1, Time = 1.86299 msec, Performace = 1152.71 GFlop/s
//LB = 3: Size = 1, Time = 2.21645 msec, Performace = 968.884 GFlop/s
//LB = 4: Size = 1.125, Time = 2.08311 msec, Performace = 1159.77 GFlop/s
//LB = 3: Size = 1.125, Time = 2.69313 msec, Performace = 897.069 GFlop/s
//[OC = 32]:
//LB = 3: Size = 1.125, Time = 3.68506 msec, Performace = 655.598 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_4_8R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
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
	CW += oc0 + ((ty & 1) << 1);//CW[0, 0, 0, toc0]

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
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC, SCW = IC * OC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, CW += SCW)
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

			//load 2 elem from W
			int Wic = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + Wic * OC);
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

				//load 4 elem from X
				float4 x; int Xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
				x.x = (lx0 ? X[X0 + Xic] : 0);
				x.y = (lx1 ? X[X1 + Xic] : 0);
				x.z = (lx2 ? X[X2 + Xic] : 0);
				x.w = (lx3 ? X[X3 + Xic] : 0);
				Xs[buf][tx][ty] = x;

				//load 2 elem from W
				int Wic = ((oic << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + Wic * OC);
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
#ifndef CONV_3D_PURE_KERNEL_4_4R
#define CONV_3D_PURE_KERNEL_4_4R

//LB = 4: Size = 1.125, Time = 2.37496 msec, Performace = 1017.24  GFlop/s
//LB = 3: Size = 1.125, Time = 2.66854 msec, Performace =  905.334 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_4_4R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
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
	CW += oc0 + ((ty & 1) << 1);//CW[0, 0, 0, toc0]

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
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC, SCW = IC * OC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, CW += SCW)
		{
			//load 2 elem from X
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			float2 x; int Xic = tx >> 1;
			x.x = (lx0 ? X[X0 + Xic] : 0);
			x.y = (lx1 ? X[X1 + Xic] : 0);
			Xs[buf][Xs_x][Xs_y] = x;

			//load 2 elem from W
			int Wic = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + Wic * OC);
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

				//load 2 elem from X
				float2 x; int Xic = ((oic << LB) + tx) >> 1;
				x.x = (lx0 ? X[X0 + Xic] : 0);
				x.y = (lx1 ? X[X1 + Xic] : 0);
				Xs[buf][Xs_x][Xs_y] = x;

				//load 2 elem from W
				int Wic = ((oic << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + Wic * OC);
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
#ifndef CONV_3D_PURE_KERNEL_8_2R
#define CONV_3D_PURE_KERNEL_8_2R

//LB = 4: Size = 1.125, Time = 3.0908  msec, Performace = 781.649 GFlop/s
//LB = 3: Size = 1.125, Time = 3.25482 msec, Performace = 742.26 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_8_2R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
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
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

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
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC, SCW = IC * OC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, CW += SCW)
		{
			//load 1 elem from X
			int Xic = tx >> 1; bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			Xs[buf][Xs_x][Xs_y] = (lx0 ? X[X0 + Xic] : 0);

			//load 4 elem from W
			int Wic = ty - ((ty >= STEP) << LB >> 1);
			Ws[buf][ty][tx] = *(float4*)(CW + Wic * OC);
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++) {
					float2 b0 = *(float2*)(&Xs[buf][ik][ty << 1]);
					float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
					simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
					simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				}
				buf ^= 1;

				//load 2 elem from X
				int Xic = ((oic << LB) + tx) >> 1;
				Xs[buf][Xs_x][Xs_y] = (lx0 ? X[X0 + Xic] : 0);

				//load 4 elem from W
				int Wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
				Ws[buf][ty][tx] = *(float4*)(CW + Wic * OC);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float2 b0 = *(float2*)(&Xs[buf][ik][ty << 1]);
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
}


#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), IC % (BLOCK_SIZE/2) == 0
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_4_2R
#define CONV_3D_PURE_KERNEL_4_2R

//LB = 4: Size = 1, Time = 2.71167 msec, Performace = 791.942 GFlop/s
//LB = 3: Size = 1, Time = 3.56612 msec, Performace = 602.191 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_4_2R(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.y, ty = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty & 1) << 1);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + ty) << 1) + j_index;
	int tj0 = j0 + (tx & 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);

	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	const int Xs_x = (tx >> 1), Xs_y = (ty << 1) + (tx & 1);
	const int OIC = (IC << 1 >> LB), SX = (IW - FW) *IC, SCW = IC * OC;
	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, CW += SCW)
		{
			//load 1 elem from X
			int Xic = tx >> 1; bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			Xs[buf][Xs_x][Xs_y] = (lx0 ? X[X0 + Xic] : 0);

			//load 2 elem from W
			int Wic = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + Wic * OC);
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++) {
					float2 b = *(float2*)(&Xs[buf][ik][ty << 1]);
					float4 a = *(float4*)(&Ws[buf][ik][tx << 1]);
					simdMM4(v0, b.x, a);
					simdMM4(v1, b.y, a);
				}
				buf ^= 1;

				//load 2 elem from X
				int Xic = ((oic << LB) + tx) >> 1;
				Xs[buf][Xs_x][Xs_y] = (lx0 ? X[X0 + Xic] : 0);

				//load 2 elem from W
				int Wic = ((oic << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + Wic * OC);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float2 b = *(float2*)(&Xs[buf][ik][ty << 1]);
				float4 a = *(float4*)(&Ws[buf][ik][tx << 1]);
				simdMM4(v0, b.x, a);
				simdMM4(v1, b.y, a);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0; //oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
}

#endif

#endif