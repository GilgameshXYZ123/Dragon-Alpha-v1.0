#pragma once

#ifndef DECONV3D_DW_GEMM_KERNEL_EX2_H
#define DECONV3D_DW_GEMM_KERNEL_EX2_H

//deltaY_pe is the 4D convolution kernel: 
//deltaW_e = conv(X, deltaY_pe)| step=1, padding=(oph, opw)
//logically:
//deltaY[N, OH, OW, OC]  -> deltaYpe[OC, OHp, OWp, N]
//     X[N, IH, IW, IC]  ->       Xe[IC, IH, IW, N]
//deltaW[OC, FH, FW, IC] ->  deltaWe[IC, FH, FW, OC]
//
//we have:
//	(1) FH * FW >= 2
//	(2) GN = OC:           GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW: GM%4 == 0, GM >= 8
//	(4) GK = N  * OH * OW: GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress
#ifndef DECONV3D_DW_GEMM_KERNEL_EX2_CALL
#define DECONV3D_DW_GEMM_KERNEL_EX2_CALL

//======[Template: fixed feature size]==================================
#define fGemm88(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	fernel_Gemm_8_8<LB, (1<<LB>>1), OH, OW, (OH*OW)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#endif


//======[Template: fixed feature size]==================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_FERNEL_8_8
#define DECONV3D_DW_GEMM_FERNEL_8_8

//small size, big channel
//[OH = OW = 2]
//LB = 4: Size = 1.125, Time = 1.588 msec, Performace = 1521.36 GFlop/s
//LB = 3: Size = 1.125, Time = 1.8   msec, Performace = 1342.18 GFlop/s
//[OH = OW = 3]
//LB = 4: Size = 1.26562, Time = 2.026 msec, Performace = 1341.51 GFlop/s
//LB = 3: Size = 1.26562, Time = 2.258 msec, Performace = 1203.68 GFlop/s
//[OH = OW = 5]
//LB = 4: Size = 1.75781, Time = 2.716 msec, Performace = 1389.86 GFlop/s
//LB = 3: Size = 1.75781, Time = 3.042 msec, Performace = 1240.92 GFlop/s
//[OH = OW = 7] 
//LB = 4: Size = 1.72266, Time = 2.66  msec, Performace = 1390.74 GFlop/s
//LB = 3: Size = 1.72266, Time = 2.968 msec, Performace = 1246.42 GFlop/s
//SK kernel is faster
template<int LB, int STEP, int OH, int OW, int OH_OW>
__global__ void fernel_Gemm_8_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int Wstride = FH * FW_IC;
	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	const int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	const int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	const int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
	*(float4*)(deltaW + oc4) = v8;  *(float4*)(deltaW + oc4 + 4) = v9;
	*(float4*)(deltaW + oc5) = v10; *(float4*)(deltaW + oc5 + 4) = v11;
	*(float4*)(deltaW + oc6) = v12; *(float4*)(deltaW + oc6 + 4) = v13;
	*(float4*)(deltaW + oc7) = v14; *(float4*)(deltaW + oc7 + 4) = v15;
}

#endif

#endif