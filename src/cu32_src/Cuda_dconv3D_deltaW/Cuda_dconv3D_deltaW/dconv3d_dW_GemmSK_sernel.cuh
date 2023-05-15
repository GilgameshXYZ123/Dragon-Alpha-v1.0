
#pragma once

#ifndef DECONV3D_DW_GEMMSK_SERNEL_H
#define DECONV3D_DW_GEMMSK_SERNEL_H

//Split K to improve parallism
//we have:
//	(1) FH * FW >= 2
//	(2) GN = OC:           GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW: GM%4 == 0, GM >= 8
//	(4) GK = N  * OH * OW: GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
//	(6) ph = oph, pw = opw
//	(7) IC % 4 == 0
#ifndef DECONV3D_DW_GEMMSK_SERNEL_CALL
#define DECONV3D_DW_GEMMSK_SERNEL_CALL

//LB = log2(BLOCK_SIZE)
//GZ = gridDim.z

//======[Small GM]===========================================
#define sGemmSK_8x2_1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	sernel_GemmSK_8x2_1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM<<1>>LB, GN>>LB>>3, GZ), dim3(1<<LB>>1, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define sGemmSK_4x2_1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	sernel_GemmSK_4x2_1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM<<1>>LB, GN>>LB>>2, GZ), dim3(1<<LB>>1, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define sGemmSK_2x2_1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	sernel_GemmSK_2x2_1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM<<1>>LB, GN>>LB>>1, GZ), dim3(1<<LB>>1, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//======[Small GN]===========================================
#define sGemmSK_1_4x2(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	sernel_GemmSK_1_4x2<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN<<1>>LB, GZ), dim3(1<<LB, 1<<LB>>1), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define sGemmSK_1_2x2(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	sernel_GemmSK_1_2x2<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN<<1>>LB, GZ), dim3(1<<LB, 1<<LB>>1), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#endif


//======[Small GM]===========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*0.5), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_SERNEL_8X2_1
#define DECONV3D_DW_GEMMSK_SERNEL_8X2_1

//[GM =  8 * 3 * 3]: LB = 3: Size = 0.5625, Time = 3.766 msec, Performace = 320.754 GFlop/s
//[GM =  4 * 3 * 3]: LB = 3: sSize = 0.5625, Time = 3.81 msec, Performace = 317.05  GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmSK_8x2_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH_OW, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(2 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================s

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB >> 1) + tx) + j_index;
	get_fh_fw_ic(j0, fh0, fw0, ic0);
	fh0 -= oph; fw0 -= opw;
	X += (fh0*IW + fw0)*IC + ic0;//X += X0

	//load 8 elem from deltaY[N, OH, OW, OC]
	int Y_k = tx + GK_start;
	int yoffset1 = Y_k * OC;
	Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset1);
	Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset1 + 4);

	int yoffset2 = yoffset1 + (OC << LB >> 1);
	Ys[buf][tx + STEP2][(ty << 1)] = *(float4*)(deltaY + yoffset2);
	Ys[buf][tx + STEP2][(ty << 1) + 1] = *(float4*)(deltaY + yoffset2 + 4);
	
	//load 1 elem from X[N, IH, IW, IC]
	int X_k = ty + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(fh0, fw0) ? X[xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float  b = Xs[buf][ik][tx];
			float4 a0 = Ys[buf][ik][(ty << 1)];
			float4 a1 = Ys[buf][ik][(ty << 1) + 1];

			simdMM4(v0, b, a0);
			simdMM4(v1, b, a1);
		}
		buf ^= 1;

		//load 4 elem from deltaY[N, OH, OW, OC]
		int Y_k = (ok << LB) + tx + GK_start;
		int yoffset1 = Y_k * OC;
		Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset1);
		Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset1 + 4);

		int yoffset2 = yoffset1 + (OC << LB >> 1);
		Ys[buf][tx + STEP2][(ty << 1)] = *(float4*)(deltaY + yoffset2);
		Ys[buf][tx + STEP2][(ty << 1) + 1] = *(float4*)(deltaY + yoffset2 + 4);

		//load 1 elem from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(fh0, fw0) ? X[xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float  b = Xs[buf][ik][tx];
		float4 a0 = Ys[buf][ik][(ty << 1)];
		float4 a1 = Ys[buf][ik][(ty << 1) + 1];

		simdMM4(v0, b, a0);
		simdMM4(v1, b, a1);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		//load 4 elements from deltaY
		int yoffset = (k + GK_start) * OC;
		float4 a0 = *(float4*)(deltaY + yoffset);
		float4 a1 = *(float4*)(deltaY + yoffset + 4);

		float b;//load 1 element from X
		int X_k = k + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b = (LOAD_X(fh0, fw0) ? X[xoffset] : 0);

		simdMM4(v0, b, a0);
		simdMM4(v1, b, a1);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * Wstride + j0;//j = *fh * FW + fw)*IC + ic
	const int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	const int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	const int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	deltaW[oc0] = v0.x;
	deltaW[oc1] = v0.y;
	deltaW[oc2] = v0.z;
	deltaW[oc3] = v0.w;
	deltaW[oc4] = v1.x;
	deltaW[oc5] = v1.y;
	deltaW[oc6] = v1.z;
	deltaW[oc7] = v1.w;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*0.5), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_SERNEL_4X2_1
#define DECONV3D_DW_GEMMSK_SERNEL_4X2_1

//[GM =  8 * 3 * 3]: LB = 3: Size = 0.5625, Time = 3.948 msec, Performace = 305.967 GFlop/s
//[GM =  4 * 3 * 3]: LB = 3: Size = 0.5625, Time = 3.94  msec, Performace = 306.589 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmSK_4x2_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH_OW, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//=====================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB >> 1) + tx) + j_index;
	get_fh_fw_ic(j0, fh0, fw0, ic0);
	fh0 -= oph; fw0 -= opw;
	X += (fh0*IW + fw0)*IC + ic0;//X += X0

	//load 4 elem from deltaY[N, OH, OW, OC]
	int Y_k = tx + GK_start;
	int yoffset1 = Y_k * OC, yoffset2 = yoffset1 + (OC << LB >> 1);
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset1);
	Ys[buf][tx + STEP2][ty] = *(float4*)(deltaY + yoffset2);

	//load 1 elem from X[N, IH, IW, IC]
	int X_k = ty + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(fh0, fw0) ? X[xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = GK_slice >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Xs[buf][ik][tx];
			float4 a = Ys[buf][ik][ty];
			simdMM4(v0, b, a);
		}
		buf ^= 1;

		//load 4 elem from deltaY[N, OH, OW, OC]
		int Y_k = (ok << LB) + tx + GK_start;
		int yoffset1 = Y_k * OC, yoffset2 = yoffset1 + (OC << LB >> 1);
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset1);
		Ys[buf][tx + STEP2][ty] = *(float4*)(deltaY + yoffset2);

		//load 1 elem from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(fh0, fw0) ? X[xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  b = Xs[buf][ik][tx];
		float4 a = Ys[buf][ik][ty];
		simdMM4(v0, b, a);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		//load 4 elements from deltaY
		float4 a = *(float4*)(deltaY + (k + GK_start) * OC);

		float b;//load 1 element from X
		int X_k = k + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b = (LOAD_X(fh0, fw0) ? X[xoffset] : 0);

		simdMM4(v0, b, a);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * Wstride + j0;//j = *fh * FW + fw)*IC + ic
	const int oc1 = oc0 + Wstride;
	const int oc2 = oc1 + Wstride;
	const int oc3 = oc2 + Wstride;

	deltaW[oc0] = v0.x;
	deltaW[oc1] = v0.y;
	deltaW[oc2] = v0.z;
	deltaW[oc3] = v0.w;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*0.5), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_SERNEL_2X2_1
#define DECONV3D_DW_GEMMSK_SERNEL_2X2_1

//[GM =  8 * 3 * 3]: LB = 3: Size = 0.5625, Time = 5.9   msec, Performace = 204.739 GFlop/s
//[GM =  4 * 3 * 3]: LB = 3: Size = 0.5625, Time = 5.914 msec, Performace = 204.254 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmSK_2x2_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH_OW, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//=====================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += oc0;

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB >> 1) + tx) + j_index;
	get_fh_fw_ic(j0, fh0, fw0, ic0);
	fh0 -= oph; fw0 -= opw;
	const int X0 = (fh0*IW + fw0)*IC + ic0;

	//load 2 elem from deltaY[N, OH, OW, OC]
	int Y_k = tx + GK_start;
	int yoffset1 = Y_k * OC, yoffset2 = yoffset1 + (OC << LB >> 1);
	Ys[buf][tx][ty] = *(float2*)(deltaY + yoffset1);
	Ys[buf][tx + STEP2][ty] = *(float2*)(deltaY + yoffset2);

	//load 1 elem from X[N, IH, IW, IC]
	int X_k = ty + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Xs[buf][ik][tx];
			float2 a = Ys[buf][ik][ty];
			simdMM2(v0, b, a);
		}
		buf ^= 1;

		//load 2 elem from deltaY[N, OH, OW, OC]
		int Y_k = (ok << LB) + tx + GK_start;
		int yoffset1 = Y_k * OC, yoffset2 = yoffset1 + (OC << LB >> 1);
		Ys[buf][tx][ty] = *(float2*)(deltaY + yoffset1);
		Ys[buf][tx + STEP2][ty] = *(float2*)(deltaY + yoffset2);

		//load 1 elem from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  b = Xs[buf][ik][tx];
		float2 a = Ys[buf][ik][ty];
		simdMM2(v0, b, a);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		//load 2 elements from deltaY
		float2 a = *(float2*)(deltaY + (k + GK_start) * OC);

		float b;//load 1 element from X
		int X_k = k + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);

		simdMM2(v0, b, a);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * Wstride + j0;//j = *fh * FW + fw)*IC + ic
	const int oc1 = oc0 + Wstride;

	deltaW[oc0] = v0.x;
	deltaW[oc1] = v0.y;
}

#endif


//======[Small GN]===========================================
//(Y: BLOCK_SIZE*0.5, X: BLOCK_SIZE*4), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_SERNEL_1_4X2
#define DECONV3D_DW_GEMMSK_SERNEL_1_4X2

//[OC = 16]: LB = 3: Size = 0.5625, Time = 7.338 msec, Performace = 164.617 GFlop/s
//[OC =  8]: LB = 3: Size = 0.5625, Time = 6.944 msec, Performace = 173.957 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmSK_1_4x2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH_OW, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//=====================================================================

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB >> 1) + ty) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC * FH * FW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	get_fh_fw_ic(j1, fh1, fw1, ic1); fh1 -= oph; fw1 -= opw;
	get_fh_fw_ic(j2, fh2, fw2, ic2); fh2 -= oph; fw2 -= opw;
	get_fh_fw_ic(j3, fh3, fw3, ic3); fh3 -= oph; fw3 -= opw;
	const int X0 = (fh0*IW + fw0)*IC + ic0;
	const int X1 = (fh1*IW + fw1)*IC + ic1;
	const int X2 = (fh2*IW + fw2)*IC + ic2;
	const int X3 = (fh3*IW + fw3)*IC + ic3;

	//load 1 elem rom deltaY[N, OH, OW, OC]
	int Y_k = tx + GK_start;
	Ys[buf][tx][ty] = deltaY[Y_k * OC];

	//load 2 elem from X[N, IH, IW, IC]
	int X_k1 = ty + GK_start, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k1, X_n, X_oh, X_ow);
	float4 xv; int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
	xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
	xv.z = (LOAD_X(fh2, fw2) ? X[X2 + xoffset] : 0);
	xv.w = (LOAD_X(fh3, fw3) ? X[X3 + xoffset] : 0);
	Xs[buf][ty][tx] = xv;

	int X_k2 = ty + GK_start + STEP2;
	get_X_n_oh_ow(X_k2, X_n, X_oh, X_ow);
	xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
	xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
	xv.z = (LOAD_X(fh2, fw2) ? X[X2 + xoffset] : 0);
	xv.w = (LOAD_X(fh3, fw3) ? X[X3 + xoffset] : 0);
	Xs[buf][ty + STEP2][tx] = xv;
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = GK_slice >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = Xs[buf][ik][tx];
			float  a = Ys[buf][ik][ty];
			simdMM4(v0, a, b);
		}
		buf ^= 1;

		//load 1 elem from deltaY[N, OH, OW, OC]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = deltaY[Y_k * OC];

		//load 2 elem from X[N, IH, IW, IC]
		int X_k1 = (ok << LB) + ty + GK_start, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k1, X_n, X_oh, X_ow);
		float4 xv; int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
		xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
		xv.z = (LOAD_X(fh2, fw2) ? X[X2 + xoffset] : 0);
		xv.w = (LOAD_X(fh3, fw3) ? X[X3 + xoffset] : 0);
		Xs[buf][ty][tx] = xv;

		int X_k2 = (ok << LB) + ty + GK_start + STEP2;
		get_X_n_oh_ow(X_k2, X_n, X_oh, X_ow);
		xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
		xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
		xv.z = (LOAD_X(fh2, fw2) ? X[X2 + xoffset] : 0);
		xv.w = (LOAD_X(fh3, fw3) ? X[X3 + xoffset] : 0);
		Xs[buf][ty + STEP2][tx] = xv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b = Xs[buf][ik][tx];
		float  a = Ys[buf][ik][ty];
		simdMM4(v0, a, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice & (STEP - 1)); k < GK_slice; k++)
	{
		//load 1 element  from deltaY
		float a = deltaY[(k + GK_start) * OC];

		//load 2 elements from X
		int X_k1 = k + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k1, X_n, X_oh, X_ow);
		float4 b; int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
		b.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
		b.z = (LOAD_X(fh2, fw2) ? X[X2 + xoffset] : 0);
		b.w = (LOAD_X(fh3, fw3) ? X[X3 + xoffset] : 0);

		simdMM4(v0, a, b);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * Wstride + j0; //j = *fh * FW + fw)*IC + ic
	*(float4*)(deltaW + oc0) = v0;
}

#endif


//(Y: BLOCK_SIZE*0.5, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_SERNEL_1_2X2
#define DECONV3D_DW_GEMMSK_SERNEL_1_2X2

//[OC = 16]: LB = 3: Size = 0.5625, Time = 9.872 msec, Performace = 122.362 GFlop/s
//[OC =  8]: LB = 3: Size = 0.5625, Time = 9.986 msec, Performace = 120.965 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmSK_1_2x2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH_OW, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//=====================================================================

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB >> 1) + ty) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC * FH * FW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	get_fh_fw_ic(j1, fh1, fw1, ic1); fh1 -= oph; fw1 -= opw;
	const int X0 = (fh0*IW + fw0)*IC + ic0;
	const int X1 = (fh1*IW + fw1)*IC + ic1;

	//load 1 elem from deltaY[N, OH, OW, OC]
	int Y_k = tx + GK_start;
	Ys[buf][tx][ty] = deltaY[Y_k * OC];

	//load 2 elem from X[N, IH, IW, IC]
	int X_k1 = ty + GK_start, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k1, X_n, X_oh, X_ow);
	float2 xv; int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
	xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
	Xs[buf][ty][tx] = xv;

	int X_k2 = ty + GK_start + STEP2;
	get_X_n_oh_ow(X_k2, X_n, X_oh, X_ow);
	xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
	xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
	Xs[buf][ty + STEP2][tx] = xv;
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float  a = Ys[buf][ik][ty];
			simdMM2(v0, a, b);
		}
		buf ^= 1;

		//load 1 elem from deltaY[N, OH, OW, OC]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = deltaY[Y_k * OC];

		//load 2 elem from X[N, IH, IW, IC]
		int X_k1 = (ok << LB) + ty + GK_start, X_n, X_oh, X_ows;
		get_X_n_oh_ow(X_k1, X_n, X_oh, X_ow);
		float2 xv; int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
		xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
		Xs[buf][ty][tx] = xv;

		int X_k2 = (ok << LB) + ty + GK_start + STEP2;
		get_X_n_oh_ow(X_k2, X_n, X_oh, X_ow);
		xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
		xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
		Xs[buf][ty + STEP2][tx] = xv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Xs[buf][ik][tx];
		float  a = Ys[buf][ik][ty];
		simdMM2(v0, a, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice  & (STEP - 1)); k < GK_slice; k++)
	{
		//load 1 element  from deltaY
		float a = deltaY[(k + GK_start) * OC];

		//load 2 elements from X
		int X_k = k + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		float2 b; int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
		b.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);

		simdMM2(v0, a, b);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * Wstride + j0; //j = *fh * FW + fw)*IC + ic
	*(float2*)(deltaW + oc0) = v0;
}

#endif

#endif