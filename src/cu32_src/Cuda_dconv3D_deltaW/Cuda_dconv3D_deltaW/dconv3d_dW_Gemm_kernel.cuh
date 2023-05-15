#pragma once

#ifndef DECONV3D_DW_GEMM_KERNEL_H
#define DECONV3D_DW_GEMM_KERNEL_H

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
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
//	(6) ph = oph, pw = opw
#ifndef DECONV3D_DW_GEMM_KERNEL_CALL
#define DECONV3D_DW_GEMM_KERNEL_CALL

//LB = log2(BLOCK_SIZE)

//=====[Common]=========================================================
#define kGemm88(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_8_8<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm84(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_8_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm48(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_4_8<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm44(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_4_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm82(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_8_2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm28(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_2_8<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm42(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_4_2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm24(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_2_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//=====[Small]==========================================================
#define kGemm41(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_4_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm14(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_1_4<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm22(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_2_2<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm21(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_2_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm12(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_1_2<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm11(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_1_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#endif


//=====[Common]=========================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_8_8
#define DECONV3D_DW_GEMM_KERNEL_8_8

//LB = 4: Size = 1, Time = 1.788 msec, Performace = 1201.05 GFlop/s
//LB = 3: Size = 1, Time = 2.15  msec, Performace =  998.829 GFlop/s
//LB = 4: Size = 1.125, Time = 1.76 msec, Performace = 1372.68 GFlop/s
//LB = 3: Size = 1.125, Time = 2.06 msec, Performace = 1172.78 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_8_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
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
	float4  v0 = make_float4(0, 0, 0, 0),  v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0),  v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0),  v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0),  v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0),  v9 = make_float4(0, 0, 0, 0);
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

			simdMM4( v0, a0.x, b0); simdMM4( v1, a0.x, b1);
			simdMM4( v2, a0.y, b0); simdMM4( v3, a0.y, b1);
			simdMM4( v4, a0.z, b0); simdMM4( v5, a0.z, b1);
			simdMM4( v6, a0.w, b0); simdMM4( v7, a0.w, b1);
			simdMM4( v8, a1.x, b0); simdMM4( v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
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

		simdMM4( v0, a0.x, b0); simdMM4( v1, a0.x, b1);
		simdMM4( v2, a0.y, b0); simdMM4( v3, a0.y, b1);
		simdMM4( v4, a0.z, b0); simdMM4( v5, a0.z, b1);
		simdMM4( v6, a0.w, b0); simdMM4( v7, a0.w, b1);
		simdMM4( v8, a1.x, b0); simdMM4( v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int Wstride = FH * FW_IC; 
	oc0 = oc0*Wstride + j0;//j = (fh * FW + fw)*IC + ic
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_8_4
#define DECONV3D_DW_GEMM_KERNEL_8_4

//LB = 4: Size = 1, Time = 2.358 msec, Performace = 910.722 GFlop/s
//LB = 3: Size = 1, Time = 2.53  msec, Performace = 848.808 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_8_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;//X += X0

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);

			simdMM4(v0, a0.x, b0); 
			simdMM4(v2, a0.y, b0); 
			simdMM4(v4, a0.z, b0); 
			simdMM4(v6, a0.w, b0);
			simdMM4(v8, a1.x, b0);
			simdMM4(v10, a1.y, b0); 
			simdMM4(v12, a1.z, b0);
			simdMM4(v14, a1.w, b0);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);

		simdMM4(v0, a0.x, b0);
		simdMM4(v2, a0.y, b0);
		simdMM4(v4, a0.z, b0);
		simdMM4(v6, a0.w, b0);
		simdMM4(v8, a1.x, b0);
		simdMM4(v10, a1.y, b0);
		simdMM4(v12, a1.z, b0);
		simdMM4(v14, a1.w, b0);
	}

	const int FH_FW_IC = FH * FW_IC;
	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(deltaW + oc0) = v0;
	*(float4*)(deltaW + oc1) = v2;
	*(float4*)(deltaW + oc2) = v4;
	*(float4*)(deltaW + oc3) = v6;
	*(float4*)(deltaW + oc4) = v8;
	*(float4*)(deltaW + oc5) = v10; 
	*(float4*)(deltaW + oc6) = v12; 
	*(float4*)(deltaW + oc7) = v14; 
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_4_8
#define DECONV3D_DW_GEMM_KERNEL_4_8

//LB = 4: Size = 1, Time = 2.474 msec, Performace = 868.021 GFlop/s
//LB = 3: Size = 1, Time = 2.572 msec, Performace = 834.947 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_4_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//followed k88

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += ((tx & 1) << 1) + oc0;

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k * OC);

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
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&Ys[buf][ik][ty << 1]);
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k * OC);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
	}

	const int FH_FW_IC = FH * FW_IC;
	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_4_4
#define DECONV3D_DW_GEMM_KERNEL_4_4

//LB = 4: Size = 1, Time = 2.75 msec, Performace = 780.903 GFlop/s
//LB = 3: Size = 1, Time = 3.542 msec, Performace = 606.291 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_4_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW, 
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += ((tx & 1) << 1) + oc0;

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;//X += X0

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) ;//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k * OC);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) 
		{
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);

			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k*OC);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}

	const int FH_FW_IC = FH * FW_IC; 
	oc0 = oc0 * FH_FW_IC + j0; //j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC; 

	*(float4*)(deltaW + oc0) = v0;
	*(float4*)(deltaW + oc1) = v1;
	*(float4*)(deltaW + oc2) = v2;
	*(float4*)(deltaW + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_8_2
#define DECONV3D_DW_GEMM_KERNEL_8_2

//LB = 4: Size = 1.125, Time = 2.726 msec, Performace = 886.25 GFlop/s
//LB = 3: Size = 1.125, Time = 3.678 msec, Performace = 656.857 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_8_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	const int tj0 = j0 + (ty & 1);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;//X += X0

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

	//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? X[xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2  v0 = make_float2(0, 0);
	float2  v2 = make_float2(0, 0);
	float2  v4 = make_float2(0, 0);
	float2  v6 = make_float2(0, 0);
	float2  v8 = make_float2(0, 0);
	float2 v10 = make_float2(0, 0);
	float2 v12 = make_float2(0, 0);
	float2 v14 = make_float2(0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float2 b0 = *(float2*)(&Xs[buf][ik][tx << 1]);

			simdMM2(v0, a0.x, b0);
			simdMM2(v2, a0.y, b0);
			simdMM2(v4, a0.z, b0);
			simdMM2(v6, a0.w, b0);
			simdMM2(v8, a1.x, b0);
			simdMM2(v10, a1.y, b0);
			simdMM2(v12, a1.z, b0);
			simdMM2(v14, a1.w, b0);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? X[xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float2 b0 = *(float2*)(&Xs[buf][ik][tx << 1]);

		simdMM2(v0, a0.x, b0);
		simdMM2(v2, a0.y, b0);
		simdMM2(v4, a0.z, b0);
		simdMM2(v6, a0.w, b0);
		simdMM2(v8, a1.x, b0);
		simdMM2(v10, a1.y, b0);
		simdMM2(v12, a1.z, b0);
		simdMM2(v14, a1.w, b0);
	}

	const int Wstride = FH * FW_IC;
	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	*(float2*)(deltaW + oc0) = v0;
	*(float2*)(deltaW + oc1) = v2;
	*(float2*)(deltaW + oc2) = v4;
	*(float2*)(deltaW + oc3) = v6;
	*(float2*)(deltaW + oc4) = v8;
	*(float2*)(deltaW + oc5) = v10;
	*(float2*)(deltaW + oc6) = v12;
	*(float2*)(deltaW + oc7) = v14;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_2_8
#define DECONV3D_DW_GEMM_KERNEL_2_8

//LB = 4: Size = 1.125, Time = 3.35 msec, Performace = 721.17 GFlop/s
//LB = 3: Size = 1.125, Time = 4.05 msec, Performace = 596.523 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_2_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//followed k88

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += (tx & 1) + oc0;

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 1 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = deltaY[Y_k * OC];

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 a0 = *(float2*)(&Ys[buf][ik][ty << 1]);
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][Ys_x][Ys_y] = deltaY[Y_k * OC];

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 a0 = *(float2*)(&Ys[buf][ik][ty << 1]);
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
	}

	const int Wstride = FH * FW_IC;
	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2),  GK >= (BLOCK_SIZE/2)
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_4_2
#define DECONV3D_DW_GEMM_KERNEL_4_2

//LB=4: Size = 1, Time = 3.606 msec, Performace = 595.531 GFlop/s
//LB=3: Size = 1, Time = 5.416 msec, Performace = 396.507 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_4_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
		  float* __restrict__ deltaW, int FH, int FW, 
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float   Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int toc0 = ((tx & 1) << 1) + oc0;
	
	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int tj0 = j0 + (ty & 1);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0;

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (lx0 ? X[Xoffset0 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 b = *(float2*)(&Xs[buf][ik][tx << 1]); 
			float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);

			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
			simdMM2(v2, a.z, b);
			simdMM2(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k* OC + toc0]);

		//load 1 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
		simdMM2(v2, a.z, b);
		simdMM2(v3, a.w, b);
	}

	const int FH_FW_IC = FH * FW_IC; 
	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	*(float2*)(deltaW + oc0) = v0;
    *(float2*)(deltaW + oc1) = v1;
	*(float2*)(deltaW + oc2) = v2;
	*(float2*)(deltaW + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_2_4
#define DECONV3D_DW_GEMM_KERNEL_2_4

//LB = 4: Size = 1, Time = 3.922 msec, Performace = 547.548 GFlop/s
//LB = 3: Size = 1, Time = 5.48 msec, Performace = 391.877 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_2_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
		  float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += (tx & 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;//X += X0

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	Ys[buf][dYs_x][dYs_y] = deltaY[Y_k*OC];

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			float2 a = *(float2*)(&Ys[buf][ik][ty << 1]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] :
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][dYs_x][dYs_y] = deltaY[Y_k*OC];

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		float2 a = *(float2*)(&Ys[buf][ik][ty << 1]);
		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
	}

	const int FH_FW_IC = FH * FW_IC;
	oc0 = oc0 * FH_FW_IC + j0;//j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC;

	*(float4*)(deltaW + oc0) = v0;
	*(float4*)(deltaW + oc1) = v1;
}

#endif


//=====[Small]==========================================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMM_KERNEL_2_2
#define DECONV3D_DW_GEMM_KERNEL_2_2

//LB=4: Size = 1, Time = 4.464 msec, Performace = 481.067 GFlop/s
//LB=3: Size = 1, Time = 5.932 msec, Performace = 362.017 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_2_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
		  float* __restrict__ deltaW, int FH, int FW, 
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float2  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	get_fh_fw_ic(j1, fh1, fw1, ic1); fh1 -= oph; fw1 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;
	const int Xoffset1 = (fh1*IW + fw1)*IC + ic1;

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
	Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b =  Xs[buf][ik][tx];
			float2 a = dYs[buf][ik][ty];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok << LB) + tx);
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b =  Xs[buf][ik][tx];
		float2 a = dYs[buf][ik][ty];
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		float2 a = *(float2*)(&deltaY[k * OC + oc0]);//load 2 elements from deltaY

		float2 b;//load 2 elements from X
		int X_k = k, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b.x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		b.y = (lx1 ? X[Xoffset1 + xoffset] : 0);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP != 0--------------------------------------------
	
	const int FH_FW_IC = FH * FW_IC;
	oc0 = oc0 * FH_FW_IC + j0; //j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC;
	
	*(float2*)(deltaW + oc0) = v0;
	*(float2*)(deltaW + oc1) = v1;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMM_KERNEL_4_1
#define DECONV3D_DW_GEMM_KERNEL_4_1

//LB = 4: Size = 1, Time = 4.226 msec, Performace = 508.16 GFlop/s
//LB = 4: Size = 0.469238, Time = 2.394 msec, Performace = 420.92  GFlop/s
//LB = 3: Size = 0.469238, Time = 3.8   msec, Performace = 265.179 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_4_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW, 
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float   Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + oc0]);

	//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 a = dYs[buf][ik][ty];
			float  b =  Xs[buf][ik][tx];
			simdMM4(v, b, a);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + oc0]);

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 a = dYs[buf][ik][ty];
		float  b = Xs[buf][ik][tx];
		simdMM4(v, b, a);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		float4 a = *(float4*)(&deltaY[k*OC + oc0]);//load 4 elements from deltaY

		float b;//load 1 element from X
		int X_k = k;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b = (lx0 ? X[Xoffset0 + xoffset] : 0);

		simdMM4(v, b, a);
	}
	//when GK % STEP != 0--------------------------------------------

	const int FH_FW_IC = FH * FW_IC; 
	oc0 = oc0 * FH_FW_IC + j0;//j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	deltaW[oc0] = v.x;
	deltaW[oc1] = v.y;
	deltaW[oc2] = v.z;
	deltaW[oc3] = v.w;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*4), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMM_KERNEL_1_4
#define DECONV3D_DW_GEMM_KERNEL_1_4

//(correct)
//LB = 4: Size = 0.469238, Time = 3.764 msec, Performace = 267.716 GFlop/s
//LB = 3: Size = 0.469238, Time = 3.94  msec, Performace = 255.757 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_1_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
		  float* __restrict__ deltaW, int FH, int FW, 
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	get_fh_fw_ic(j1, fh1, fw1, ic1); fh1 -= oph; fw1 -= opw;
	get_fh_fw_ic(j2, fh2, fw2, ic2); fh2 -= oph; fw2 -= opw;
	get_fh_fw_ic(j3, fh3, fw3, ic3); fh3 -= oph; fw3 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;
	const int Xoffset1 = (fh1*IW + fw1)*IC + ic1;
	const int Xoffset2 = (fh2*IW + fw2)*IC + ic2;
	const int Xoffset3 = (fh3*IW + fw3)*IC + ic3;

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 1 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
	bool lx2 = (fh2 >= -X_oh) && (fh2 < IH - X_oh) && (fw2 >= -X_ow) && (fw2 < IW - X_ow);
	bool lx3 = (fh3 >= -X_oh) && (fh3 < IH - X_oh) && (fw3 >= -X_ow) && (fw3 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
	Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
	Xs[buf][ty][tx].z = (lx2 ? X[Xoffset2 + xoffset] : 0);
	Xs[buf][ty][tx].w = (lx3 ? X[Xoffset3 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = GK >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b =  Xs[buf][ik][tx];
			float  a = dYs[buf][ik][ty];
			simdMM4(v, a, b);
		}
		buf ^= 1;

		//load 1 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		bool lx2 = (fh2 >= -X_oh) && (fh2 < IH - X_oh) && (fw2 >= -X_ow) && (fw2 < IW - X_ow);
		bool lx3 = (fh3 >= -X_oh) && (fh3 < IH - X_oh) && (fw3 >= -X_ow) && (fw3 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
		Xs[buf][ty][tx].z = (lx2 ? X[Xoffset2 + xoffset] : 0);
		Xs[buf][ty][tx].w = (lx3 ? X[Xoffset3 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b =  Xs[buf][ik][tx];
		float  a = dYs[buf][ik][ty];
		simdMM4(v, a, b);
	}

	//when GK%STEP != 0 -------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		float a = deltaY[k*OC + oc0];//load 1 element from deltaY

		float4 b;//load 4 elements from X
		int X_k = k;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		bool lx2 = (fh2 >= -X_oh) && (fh2 < IH - X_oh) && (fw2 >= -X_ow) && (fw2 < IW - X_ow);
		bool lx3 = (fh3 >= -X_oh) && (fh3 < IH - X_oh) && (fw3 >= -X_ow) && (fw3 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b.x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		b.y = (lx1 ? X[Xoffset1 + xoffset] : 0);
		b.z = (lx2 ? X[Xoffset2 + xoffset] : 0);
		b.w = (lx3 ? X[Xoffset3 + xoffset] : 0);

		simdMM4(v, a, b);
	}
	//when GK%STEP != 0 -------------------------------------------

	*(float4*)(&deltaW[oc0*(FH * FW_IC) + j0]) = v;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMM_KERNEL_2_1
#define DECONV3D_DW_GEMM_KERNEL_2_1

//LB = 4: Size = 1, Time =  8.504 msec, Performace = 252.526 GFlop/s
//LB = 3: Size = 1, Time = 13.324 msec, Performace = 161.174 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_2_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
		  float* __restrict__ deltaW, int FH, int FW, 
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float   Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

	//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b =  Xs[buf][ik][tx];
			float2 a = dYs[buf][ik][ty];
			simdMM2(v, b, a);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  b =  Xs[buf][ik][tx];
		float2 a = dYs[buf][ik][ty];
		simdMM2(v, b, a);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		float2 a = *(float2*)(&deltaY[k * OC + oc0]);//load 2 elements from deltaY

		float b;//load 1 element from X
		int X_k = k, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b = (lx0 ? X[Xoffset0 + xoffset] : 0);

		simdMM2(v, b, a);
	}
	//when GK % STEP != 0--------------------------------------------

	const int FH_FW_IC = FH * FW_IC;
	oc0 = oc0 * FH_FW_IC + j0;//j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC;

	deltaW[oc0] = v.x;
	deltaW[oc1] = v.y;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMM_KERNEL_1_2
#define DECONV3D_DW_GEMM_KERNEL_1_2

//LB = 4: Size = 0.703857, Time = 5.938 msec, Performace = 254.551 GFlop/s
//LB = 3: Size = 0.703857, Time = 7.592 msec, Performace = 199.094 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_1_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
		  float* __restrict__ deltaW, int FH, int FW, 
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float2  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	
	//prepare for GM = IC * FH * FW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	get_fh_fw_ic(j1, fh1, fw1, ic1); fh1 -= oph; fw1 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;
	const int Xoffset1 = (fh1*IW + fw1)*IC + ic1;

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = deltaY[dY_k *OC + oc0];
	
	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
	Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
	__syncthreads();
	
	//compute area-----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b =  Xs[buf][ik][tx];
			float  a = dYs[buf][ik][ty];
			simdMM2(v, a, b);
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b =  Xs[buf][ik][tx];
		float  a = dYs[buf][ik][ty];
		simdMM2(v, a, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		float a = deltaY[k * OC + oc0];//load 1 element from deltaY

		float2 b;//load 2 elements from X
		int X_k = k, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b.x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		b.y = (lx1 ? X[Xoffset1 + xoffset] : 0);

		simdMM2(v, a, b);
	}
	//when GK % STEP != 0--------------------------------------------

	*(float2*)(&deltaW[oc0*(FH * FW_IC) + j0]) = v;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMM_KERNEL_1_1
#define DECONV3D_DW_GEMM_KERNEL_1_1

//LB = 4: Size = 1, Time = 11.05  msec, Performace = 194.342 GFlop/s
//LB = 3: Size = 1, Time = 16.746 msec, Performace = 128.239 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_1_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
		  float* __restrict__ deltaW, int FH, int FW, 
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

	//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
	__syncthreads();
	
	//compute area-----------------------------------------------------
	float v = 0;
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float b =  Xs[buf][ik][tx];
			float a = dYs[buf][ik][ty];
			v += a * b;
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float b =  Xs[buf][ik][tx];
		float a = dYs[buf][ik][ty];
		v += a * b;
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		float a = deltaY[k * OC + oc0];//load 1 element from deltaY

		float b;//load 1 element from X
		int X_k = k;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b = (lx0 ? X[Xoffset0 + xoffset] : 0);

		v += a * b;
	}
	//when GK % STEP != 0--------------------------------------------
	
	deltaW[oc0 *(FH * FW_IC) + j0] = v;//j = *fh * FW + fw)*IC + ic
}

#endif

#endif