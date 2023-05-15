#pragma once

#ifndef CONV_3D_GEMM_KERNEL_NO_PADDING_H
#define CONV_3D_GEMM_KERNEL_NO_PADDING_H

//ph = pw = 0
#ifndef CONV_3D_GEMM_KERNEL_NO_PADDING_CALL
#define CONV_3D_GEMM_KERNEL_NO_PADDING_CALL

#define conv3dGemm_k88x4_np(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM) \
	conv3dGemm_kernel_8_8x4_np<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, oc_index, j_index)

#define conv3dGemm_k88_np(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM) \
	conv3dGemm_kernel_8_8_np<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, oc_index, j_index)

#define conv3dGemm_k84_np(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM) \
	conv3dGemm_kernel_8_4_np<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, oc_index, j_index)

#define conv3dGemm_k48_np(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM) \
	conv3dGemm_kernel_4_8_np<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, oc_index, j_index)

#define conv3dGemm_k22_np(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM) \
	conv3dGemm_kernel_2_2_np<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, oc_index, j_index)

#define conv3dGemm_k41_np(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM) \
	conv3dGemm_kernel_4_1_np<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, oc_index, j_index)

#define conv3dGemm_k14_np(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM) \
	conv3dGemm_kernel_1_4_np<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, oc_index, j_index)

#endif


//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2)
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8X4_NP
#define CONV_3D_GEMM_KERNEL_8_8X4_NP

//k88<4>: Size = 1, Time = 1.734 msec, Performace = 1238.46 GFlop/s
//LB = 4: Size = 1, Time = 1.496 msec, Performace = 1435.48 GFlop/s
//LB = 3: Size = 1, Time = 1.764 msec, Performace = 1217.39 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8x4_np(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw,
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
	Y += oc0;//Y[0, 0, 0, oc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh;
	tow0 = tow0 * sw;
	X += ((tn0*IH + toh0)*IW + tow0 + sw)*IC; //X += X1;
	const int sw_IC = sw * IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx].x = W[W_k - GK];//W0
	Ws[buf][ty][tx].y = W[W_k];
	Ws[buf][ty][tx].z = W[W_k + GK];//W1
	Ws[buf][ty][tx].w = W[W_k + (GK << 1)];//W2

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][tx][ty].x = X[xoffset - sw_IC];
	Xs[buf][tx][ty].y = X[xoffset];
	Xs[buf][tx][ty].z = X[xoffset + sw_IC];
	Xs[buf][tx][ty].w = X[xoffset + (sw_IC << 1)];
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
		int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][tx][ty].x = X[xoffset - sw_IC];
		Xs[buf][tx][ty].y = X[xoffset];
		Xs[buf][tx][ty].z = X[xoffset + sw_IC];
		Xs[buf][tx][ty].w = X[xoffset + (sw_IC << 1)];
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

	j0 = j0 * OC;//oc = f(by), j = f(bx) -> (n, oh, ow)
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2)
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8_NP
#define CONV_3D_GEMM_KERNEL_8_8_NP

//k88<4>: Size = 1, Time = 1.734 msec, Performace = 1238.46 GFlop/s
//LB = 4: Size = 1, Time = 1.526 msec, Performace = 1407.26 GFlop/s
//LB = 3: Size = 1, Time = 1.766 msec, Performace = 1216.02 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8_np(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	      float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw,
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
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	const int Xoffset0 = ((tn0*IH + toh0 * sh)*IW + tow0 * sw)*IC;
	const int Xoffset1 = ((tn1*IH + toh1 * sh)*IW + tow1 * sw)*IC;
	const int Xoffset2 = ((tn2*IH + toh2 * sh)*IW + tow2 * sw)*IC;
	const int Xoffset3 = ((tn3*IH + toh3 * sh)*IW + tow3 * sw)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][ty][tx].x = W[toc0 + W_k];
	Ws[buf][ty][tx].y = W[toc1 + W_k];
	Ws[buf][ty][tx].z = W[toc2 + W_k];
	Ws[buf][ty][tx].w = W[toc3 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][tx][ty].x = X[Xoffset0 + xoffset];
	Xs[buf][tx][ty].y = X[Xoffset1 + xoffset];
	Xs[buf][tx][ty].z = X[Xoffset2 + xoffset];
	Xs[buf][tx][ty].w = X[Xoffset3 + xoffset];
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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx].x = W[toc0 + W_k];
		Ws[buf][ty][tx].y = W[toc1 + W_k];
		Ws[buf][ty][tx].z = W[toc2 + W_k];
		Ws[buf][ty][tx].w = W[toc3 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][tx][ty].x = X[Xoffset0 + xoffset];
		Xs[buf][tx][ty].y = X[Xoffset1 + xoffset];
		Xs[buf][tx][ty].z = X[Xoffset2 + xoffset];
		Xs[buf][tx][ty].w = X[Xoffset3 + xoffset];
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
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_4_NP
#define CONV_3D_GEMM_KERNEL_8_4_NP

//k84<4>: Size = 1, Time = 2.192 msec, Performace = 979.691 GFlop/s
//LB = 4: Size = 1, Time = 1.944 msec, Performace = 1104.67 GFlop/s
//LB = 3: Size = 1, Time = 2.52  msec, Performace =  852.176 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_4_np(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	      float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, 
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	int toc0 = (oc0 + ((tx >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	const int Xoffset0 = ((tn0*IH + toh0 * sh)*IW + tow0 * sw)*IC;
	const int Xoffset1 = ((tn1*IH + toh1 * sh)*IW + tow1 * sw)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];
	Ws[buf][tx][ty].z = W[toc2 + W_k];
	Ws[buf][tx][ty].w = W[toc3 + W_k];

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty >> 1;
	int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][Xs_y][Xs_x].x = X[Xoffset0 + xoffset];
	Xs[buf][Xs_y][Xs_x].y = X[Xoffset1 + xoffset];
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];
		Ws[buf][tx][ty].z = W[toc2 + W_k];
		Ws[buf][tx][ty].w = W[toc3 + W_k];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][Xs_y][Xs_x].x = X[Xoffset0 + xoffset];
		Xs[buf][Xs_y][Xs_x].y = X[Xoffset1 + xoffset];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);

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
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_4_8_NP
#define CONV_3D_GEMM_KERNEL_4_8_NP

//k48<4>: Size = 1, Time = 2.322 msec, Performace = 924.842 GFlop/s
//LB = 4: Size = 1, Time = 2.052 msec, Performace = 1046.53 GFlop/s
//LB = 3: Size = 1, Time = 2.742 msec, Performace = 783.181 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_8_np(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	      float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	int toc0 = (((tx & 1) << 1) + oc0)*GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	const int Xoffset0 = ((tn0*IH + toh0 * sh)*IW + tow0 * sw)*IC;
	const int Xoffset1 = ((tn1*IH + toh1 * sh)*IW + tow1 * sw)*IC;
	const int Xoffset2 = ((tn2*IH + toh2 * sh)*IW + tow2 * sw)*IC;
	const int Xoffset3 = ((tn3*IH + toh3 * sh)*IW + tow3 * sw)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
	Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx].x = X[Xoffset0 + xoffset];
	Xs[buf][ty][tx].y = X[Xoffset1 + xoffset];
	Xs[buf][ty][tx].z = X[Xoffset2 + xoffset];
	Xs[buf][ty][tx].w = X[Xoffset3 + xoffset];
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4( v0, b0.x, a0);
			simdMM4( v2, b0.y, a0);
			simdMM4( v4, b0.z, a0);
			simdMM4( v6, b0.w, a0);
			simdMM4( v8, b1.x, a0);
			simdMM4(v10, b1.y, a0);
			simdMM4(v12, b1.z, a0);
			simdMM4(v14, b1.w, a0);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][ty][tx].x = X[Xoffset0 + xoffset];
		Xs[buf][ty][tx].y = X[Xoffset1 + xoffset];
		Xs[buf][ty][tx].z = X[Xoffset2 + xoffset];
		Xs[buf][ty][tx].w = X[Xoffset3 + xoffset];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&Ws[buf][ik][ty << 1]);
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4( v0, b0.x, a0);
		simdMM4( v2, b0.y, a0);
		simdMM4( v4, b0.z, a0);
		simdMM4( v6, b0.w, a0);
		simdMM4( v8, b1.x, a0);
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


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef CONV_3D_GEMM_KERNEL_2_2_NP
#define CONV_3D_GEMM_KERNEL_2_2_NP

//k22<4>: Size = 1, Time = 4.216 msec, Performace = 509.365 GFlop/s
//LB = 4: Size = 1, Time = 3.764 msec, Performace = 570.532 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_2_2_np(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw,
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
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	const int Xoffset0 = ((n0*IH + oh0 * sh)*IW + ow0 * sw)*IC;
	const int Xoffset1 = ((n1*IH + oh1 * sh)*IW + ow1 * sw)*IC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty; 
	int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx].x = X[Xoffset0 + xoffset];
	Xs[buf][ty][tx].y = X[Xoffset1 + xoffset];
	__syncthreads();

	//compute area-----------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
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
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty; 
		int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx].x = X[Xoffset0 + xoffset];
		Xs[buf][ty][tx].y = X[Xoffset1 + xoffset];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = Ws[buf][ik][ty];
		float2 b = Xs[buf][ik][tx];
		simdMM2(v0, b.x, a);
		simdMM2(v1, b.y, a);
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float2 a;//load 2 elements from W
		a.x = W[toc0 + k];
		a.y = W[toc1 + k];

		float2 b;//load 2 elements from X
		int X_k = k;
		int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b.x = X[Xoffset0 + xoffset];
		b.y = X[Xoffset1 + xoffset];

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef CONV_3D_GEMM_KERNEL_4_1_NP
#define CONV_3D_GEMM_KERNEL_4_1_NP

//k41<4>: Size = 1, Time = 4.16  msec, Performace = 516.222 GFlop/s
//LB = 4: Size = 1, Time = 3.844 msec, Performace = 558.659 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_1_np(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	      float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, 
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
	const int Xoffset0 = ((n0*IH + oh0 * sh)*IW + ow0 * sw)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];
	Ws[buf][tx][ty].z = W[toc2 + W_k];
	Ws[buf][tx][ty].w = W[toc3 + W_k];

	//load 1 element from X[N, IH, IW, IC]
	int X_k = ty;
	int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx] = X[Xoffset0 + xoffset];
	__syncthreads();

	//compute area----------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 a = Ws[buf][ik][ty];
			float  b = Xs[buf][ik][tx];
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
		int X_k = (ok << LB) + ty;
		int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx] = X[Xoffset0 + xoffset];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 a = Ws[buf][ik][ty];
		float  b = Xs[buf][ik][tx];
		simdMM4(v, b, a);
	}

	//when GK%STEP != 0 --------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float4 a;//load 4 elements from W
		a.x = W[toc0 + k];
		a.y = W[toc1 + k];
		a.z = W[toc2 + k];
		a.w = W[toc3 + k];

		float b;//load 1 element from X
		int X_k = k;
		int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b = X[Xoffset0 + xoffset];

		simdMM4(v, b, a);
	}
	//when GK%STEP != 0 --------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	*(float4*)(Y + j0) = v;
}

#endif


//(Y: BLOCK_SIZE*1£¬X: BLOCK_SIZE*4), GK >= BLOCK_SIZE, 
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef CONV_3D_GEMM_KERNEL_1_4_NP
#define CONV_3D_GEMM_KERNEL_1_4_NP

//k14<4>: Size = 1, Time = 6.72 msec , Performace = 319.566 GFlop/s
//LB = 4: Size = 1, Time = 5.968 msec, Performace = 359.833 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_1_4_np(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	int toc0 = oc0 * GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	get_n_oh_ow(j2, n2, oh2, ow2);
	get_n_oh_ow(j3, n3, oh3, ow3);
	const int Xoffset0 = ((n0*IH + oh0 * sh)*IW + ow0 * sw)*IC;
	const int Xoffset1 = ((n1*IH + oh1 * sh)*IW + ow1 * sw)*IC;
	const int Xoffset2 = ((n2*IH + oh2 * sh)*IW + ow2 * sw)*IC;
	const int Xoffset3 = ((n3*IH + oh3 * sh)*IW + ow3 * sw)*IC;

	//load 1 element from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty] = W[toc0 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty; 
	int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
	int  IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx].x = X[Xoffset0 + xoffset];
	Xs[buf][ty][tx].y = X[Xoffset1 + xoffset];
	Xs[buf][ty][tx].z = X[Xoffset2 + xoffset];
	Xs[buf][ty][tx].w = X[Xoffset3 + xoffset];
	__syncthreads();

	//compute area------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = GK >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  a = Ws[buf][ik][ty];
			float4 b = Xs[buf][ik][tx];
			simdMM4(v, a, b);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[toc0 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty;
		int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx].x = X[Xoffset0 + xoffset];
		Xs[buf][ty][tx].y = X[Xoffset1 + xoffset];
		Xs[buf][ty][tx].z = X[Xoffset2 + xoffset];
		Xs[buf][ty][tx].w = X[Xoffset3 + xoffset];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  a = Ws[buf][ik][ty];
		float4 b = Xs[buf][ik][tx];
		simdMM4(v, a, b);
	}

	//when GK%STEP != 0 --------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		//load 1 element from W
		float a = W[toc0 + k];

		float4 b;//load 4 elements from X
		int X_k = k;
		int X_fh = X_k / FW_IC; X_k -= X_fh * FW_IC;
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b.x = X[Xoffset0 + xoffset];
		b.y = X[Xoffset1 + xoffset];
		b.z = X[Xoffset2 + xoffset];
		b.w = X[Xoffset3 + xoffset];

		simdMM4(v, a, b);
	}
	//when GK%STEP != 0 --------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	j1 = j0 + OC; j2 = j1 + OC; j3 = j2 + OC;

	Y[j0] = v.x;
	Y[j1] = v.y;
	Y[j2] = v.z;
	Y[j3] = v.w;
}

#endif

#endif
