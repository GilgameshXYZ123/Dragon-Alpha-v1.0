#pragma once

#ifndef DECONV3D_DW_GEMMSK_KERNEL_H
#define DECONV3D_DW_GEMMSK_KERNEL_H

//Split K to improve parallism:
//	(1) FH * FW >= 2
//	(2) GN = OC:           GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW: GM%4 == 0, GM >= 8
//	(4) GK = N  * OH * OW: GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
//	(6) ph = oph, pw = opw
//	(7) IC % 4 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_CALL
#define DECONV3D_DW_GEMMSK_KERNEL_CALL

//LB = log2(BLOCK_SIZE)
//GZ = gridDim.z

//======[Common]=======================================================
#define kGemmSK88(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_8_8<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK84(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_8_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK48(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_4_8<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK44(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_4_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK82(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_8_2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK28(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_2_8<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK42(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_4_2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK24(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_2_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//======[Small]========================================================
#define kGemmSK22(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_2_2<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK81(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_8_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK41(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_4_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK21(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_2_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//---------------------------------------------------------------------
#define kGemmSK14(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_1_4<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK12(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_1_2<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, (OH*OW), OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define kGemmSK11(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_GemmSK_1_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#endif


//======[Common]=============================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_8_8
#define DECONV3D_DW_GEMMSK_KERNEL_8_8

//synchronized: 
//GK_slice = 1024, OH = OW =  4, N = 256:
//LB = 4: Size = 1, Time = 1.602 msec, Performace = 1340.5 GFlop/s
//LB = 3: Size = 1, Time = 2.072 msec, Performace = 1036.43 GFlop/s
//GK_slice = 1024, OH = OW = 7, IH = IW = 14
//LB = 4: Size = 0.861328, Time = 1.38 msec, Performace = 1340.35 GFlop/s
//LB = 3: Size = 0.861328, Time = 1.708 msec, Performace = 1082.96 GFlop/s
//for[64, 64] -> [32, 32]: Size = 1.125, Time = 1.782 msec, Performace = 1355.73 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_8_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
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
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int OH_OW = OH * OW;
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
	if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
	Xs[buf][ty][tx] = xv;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
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

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
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

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
		if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
		Xs[buf][ty][tx] = xv;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
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

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_8_4
#define DECONV3D_DW_GEMMSK_KERNEL_8_4

//synchronized: 
//GK_slice = 1024, OH, OW =  4, N = 256:
//LB = 4: Size = 1, Time = 1.922 msec, Performace = 1117.32 GFlop/s
//LB = 3: Size = 1, Time = 2.468 msec, Performace =  870.131 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_8_4(
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
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

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
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
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

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

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

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, a0.x, b0);
		simdMM4(v2, a0.y, b0);
		simdMM4(v4, a0.z, b0);
		simdMM4(v6, a0.w, b0);
		simdMM4(v8, a1.x, b0);
		simdMM4(v10, a1.y, b0);
		simdMM4(v12, a1.z, b0);
		simdMM4(v14, a1.w, b0);
	}

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	const int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	const int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	const int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_4_8
#define DECONV3D_DW_GEMMSK_KERNEL_4_8

//synchronized: 
//GK_slice = 1024, OH, OW =  4, N = 256:
//LB = 4: Size = 1, Time = 2.102 msec, Performace = 1021.64  GFlop/s
//LB = 3: Size = 1, Time = 2.532 msec, Performace =  848.137 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_4_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

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
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += ((tx & 1) << 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int OH_OW = OH * OW;
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + Y_k * OC);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = *(float4*)(&Ys[buf][ik][ty << 1]);

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + Y_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = *(float4*)(&Ys[buf][ik][ty << 1]);

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
	}

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride;
	int oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_4_4
#define DECONV3D_DW_GEMMSK_KERNEL_4_4

//synchronized: 
//GK_slice = 1024, OH, OW =  4, N = 256:
//LB = 4: Size = 1, Time = 2.44  msec, Performace = 880.116 GFlop/s
//LB = 3: Size = 1, Time = 3.482 msec, Performace = 616.739 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_4_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

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
	deltaY += ((tx & 1) << 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int OH_OW = OH * OW;
	int X_k = (ty >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + Y_k * OC);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
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

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + Y_k * OC);
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

	oc0 = oc0 * Wstride + j0; //j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride;
	int oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride;

	*(float4*)(deltaW + oc0) = v0;
	*(float4*)(deltaW + oc1) = v1;
	*(float4*)(deltaW + oc2) = v2;
	*(float4*)(deltaW + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_8_2
#define DECONV3D_DW_GEMMSK_KERNEL_8_2

//synchronized: 
//GK_slice = 1024, OH, OW =  4, N = 256:
//LB = 4: Size = 1, Time = 2.546 msec, Performace = 843.474 GFlop/s
//LB = 3: Size = 1, Time = 3.548 msec, Performace = 605.266 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_8_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

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
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	const int tj0 = j0 + (ty & 1);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int OH_OW = OH * OW;
	int X_k = (ty >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? X[xoffset] : 0);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
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

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 b0 = *(float2*)(&Xs[buf][ik][tx << 1]);
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

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

		//load 1 element  from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? X[xoffset] : 0);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 b0 = *(float2*)(&Xs[buf][ik][tx << 1]);
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

		simdMM2(v0, a0.x, b0);
		simdMM2(v2, a0.y, b0);
		simdMM2(v4, a0.z, b0);
		simdMM2(v6, a0.w, b0);
		simdMM2(v8, a1.x, b0);
		simdMM2(v10, a1.y, b0);
		simdMM2(v12, a1.z, b0);
		simdMM2(v14, a1.w, b0);
	}

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_2_8
#define DECONV3D_DW_GEMMSK_KERNEL_2_8

//synchronized: 
//GK_slice = 1024, OH, OW =  4, N = 256:
//LB = 4: Size = 1, Time = 3.046 msec, Performace = 705.018 GFlop/s
//LB = 3: Size = 1, Time = 3.624 msec, Performace = 592.573 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_2_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

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
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += (tx & 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int OH_OW = OH * OW;
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

	//load 1 element from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = deltaY[Y_k * OC];
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float2 a0 = *(float2*)(&Ys[buf][ik][ty << 1]);
			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

		//load 1 element  from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		Ys[buf][Ys_x][Ys_y] = deltaY[Y_k * OC];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float2 a0 = *(float2*)(&Ys[buf][ik][ty << 1]);
		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
	}

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride;

	*(float4*)(deltaW + oc0) = v0; *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2; *(float4*)(deltaW + oc1 + 4) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_4_2
#define DECONV3D_DW_GEMMSK_KERNEL_4_2

//synchronized: 
//GK_slice = 1024, OH, OW =  4, N = 256:
//LB = 4: Size = 1, Time = 3.942 msec, Performace = 544.77 GFlop/s
//LB = 3: Size = 1, Time = 6.2 msec, Performace = 346.368 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_4_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];

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
	deltaY += ((tx & 1) << 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	const int tj0 = j0 + (ty & 1);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int OH_OW = OH * OW;
	int X_k = (ty >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? X[xoffset] : 0);

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + Y_k * OC);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);

			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
			simdMM2(v2, a.z, b);
			simdMM2(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? X[xoffset] : 0);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + Y_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
		simdMM2(v2, a.z, b);
		simdMM2(v3, a.w, b);
	}

	oc0 = oc0 * Wstride + j0; //j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride;
	int oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride;

	*(float2*)(deltaW + oc0) = v0;
	*(float2*)(deltaW + oc1) = v1;
	*(float2*)(deltaW + oc2) = v2;
	*(float2*)(deltaW + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0, IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_2_4
#define DECONV3D_DW_GEMMSK_KERNEL_2_4
//synchronized: 
//GK_slice = 1024, OH, OW =  4, N = 256:
//LB = 4: Size = 1, Time = 3.954 msec, Performace = 543.117 GFlop/s
//LB = 3: Size = 1, Time = 5.634 msec, Performace = 381.165 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_2_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

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
	deltaY += (tx & 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int OH_OW = OH * OW;
	int X_k = (ty >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);

	//load 1 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = deltaY[Y_k * OC];
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			float2 a = *(float2*)(&Ys[buf][ik][ty << 1]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		Ys[buf][Ys_x][Ys_y] = deltaY[Y_k * OC];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		float2 a = *(float2*)(&Ys[buf][ik][ty << 1]);
		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
	}

	oc0 = oc0 * Wstride + j0; //j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride;

	*(float4*)(deltaW + oc0) = v0;
	*(float4*)(deltaW + oc1) = v1;
}

#endif


//======[Small]========================================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_KERNEL_2_2
#define DECONV3D_DW_GEMMSK_KERNEL_2_2

//LB = 4: Size = 1, Time = 4.87 msec, Performace = 440.962 GFlop/s
//LB = 3: Size = 0.5625, Time = 3.204 msec, Performace = 377.016 GFlop/s
//LB = 4: Size = 0.560303, Time = 2.514 msec, Performace = 478.616 GFlop/s
//LB = 3: Size = 0.560303, Time = 3.586 msec, Performace = 335.538 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_2_2(
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
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += oc0;

	//prepare for GM = IC * FH * FW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_fh_fw_ic(j0, fh0, fw0, ic0); 
	fh0 -= oph; fw0 -= opw;
	const int X0 = (fh0*IW + fw0)*IC + ic0;
	const int X1 = X0 + 1;

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	Ys[buf][tx][ty] = *(float2*)(deltaY + Y_k * OC);

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(fh0, fw0) ? *(float2*)(X + X0 + xoffset) : FLOAT_ZERO2);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = GK_slice >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float2 a = Ys[buf][ik][ty];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = *(float2*)(deltaY + Y_k * OC);

		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] //X_n *= IC;
		int X_k = (ok << LB) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(fh0, fw0) ? *(float2*)(X + X0 + xoffset) : FLOAT_ZERO2);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Xs[buf][ik][tx];
		float2 a = Ys[buf][ik][ty];
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		//load 2 elements from deltaY
		float2 a = *(float2*)(deltaY + (k + GK_start) * OC);

		//load 2 elements from X
		int X_k = k + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		float2 b = (LOAD_X(fh0, fw0) ? *(float2*)(X + X0 + xoffset) : FLOAT_ZERO2);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * Wstride + j0; //j = *fh * FW + fw)*IC + ic
	const int oc1 = oc0 + Wstride;

	*(float2*)(deltaW + oc0) = v0;
	*(float2*)(deltaW + oc1) = v1;
}

#endif


//---------------------------------------------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_KERNEL_8_1
#define DECONV3D_DW_GEMMSK_KERNEL_8_1

//[GM = 16 * 3 * 3]: LB = 4: Size = 0.5625, Time = 2.146 msec, Performace = 562.889 GFlop/s
//[GM =  8 * 3 * 3]: LB = 3: Size = 0.5625, Time = 2.324 msec, Performace = 519.776 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_8_1(
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
	//=====================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	get_fh_fw_ic(j0, fh0, fw0, ic0);
	fh0 -= oph; fw0 -= opw;
	X += (fh0*IW + fw0)*IC + ic0;//X += X0

	//load 8 elem from deltaY[N, OH, OW, OC]
	int Y_k = tx + GK_start;
	int yoffset = Y_k * OC;
	Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset);
	Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset + 4);

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
		int yoffset = Y_k * OC;
		Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset);
		Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset + 4);

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_KERNEL_4_1
#define DECONV3D_DW_GEMMSK_KERNEL_4_1

//[GM = 16 * 3 * 3]: LB = 4: Size = 0.5625, Time = 2.574 msec, Performace = 469.293 GFlop/s
//[GM =  8 * 3 * 3]: LB = 3: Size = 0.5625, Time = 3.372 msec, Performace = 358.232 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_4_1(
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
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	get_fh_fw_ic(j0, fh0, fw0, ic0); 
	fh0 -= oph; fw0 -= opw;
	const int X0 = (fh0*IW + fw0)*IC + ic0;

	//load 4 elem from deltaY[N, OH, OW, OC]
	int Y_k = tx + GK_start;
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

	//load 1 elem from X[N, IH, IW, IC]
	int X_k = ty + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
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
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

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
		b = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);

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


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_KERNEL_2_1
#define DECONV3D_DW_GEMMSK_KERNEL_2_1

//[GM = 16 * 3 * 3]: LB = 4: Size = 0.5625, Time = 3.562 msec, Performace = 339.124 GFlop/s
//[GM =  8 * 3 * 3]: LB = 3: Size = 0.5625, Time = 5.418 msec, Performace = 222.953 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_2_1(
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
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	const int X0 = (fh0*IW + fw0)*IC + ic0;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	Ys[buf][tx][ty] = *(float2*)(deltaY + Y_k * OC);

	//load 1 element  from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
	int X_k = ty + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1, OK = GK_slice >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Xs[buf][ik][tx];
			float2 a = Ys[buf][ik][ty];
			simdMM2(v0, b, a);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = *(float2*)(deltaY + Y_k * OC);

		//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
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


//---------------------------------------------------------------------
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*4), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_KERNEL_1_4
#define DECONV3D_DW_GEMMSK_KERNEL_1_4

//[OC = 16]: LB = 4: Size = 0.5625, Time = 3.396 msec, Performace = 355.701 GFlop/s
//[OC =  8]: LB = 3: Size = 0.5625, Time = 4.228 msec, Performace = 285.705 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_1_4(
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
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
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

	//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	Ys[buf][tx][ty] = deltaY[Y_k * OC];

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] //X_n *= IC;
	int X_k = ty + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	float4 xv; int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
	xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
	xv.z = (LOAD_X(fh2, fw2) ? X[X2 + xoffset] : 0);
	xv.w = (LOAD_X(fh3, fw3) ? X[X3 + xoffset] : 0);
	Xs[buf][ty][tx] = xv;
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

		//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = deltaY[Y_k * OC];

		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] //X_n *= IC;
		int X_k = (ok << LB) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		float4 xv; int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
		xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
		xv.z = (LOAD_X(fh2, fw2) ? X[X2 + xoffset] : 0);
		xv.w = (LOAD_X(fh3, fw3) ? X[X3 + xoffset] : 0);
		Xs[buf][ty][tx] = xv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b = Xs[buf][ik][tx];
		float  a = Ys[buf][ik][ty];
		simdMM4(v0, a, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		//load 1 element  from deltaY
		float a = deltaY[(k + GK_start) * OC];

		//load 2 elements from X
		int X_k = k + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
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


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_KERNEL_1_2
#define DECONV3D_DW_GEMMSK_KERNEL_1_2

//[OC = 16]: LB = 4: Size = 0.5625, Time = 4.472 msec, Performace = 270.116 GFlop/s
//[OC =  8]: LB = 3: Size = 0.5625, Time = 6.802 msec, Performace = 177.589 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_1_2(
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
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC * FH * FW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	get_fh_fw_ic(j1, fh1, fw1, ic1); fh1 -= oph; fw1 -= opw;
	const int X0 = (fh0*IW + fw0)*IC + ic0;
	const int X1 = (fh1*IW + fw1)*IC + ic1;

	//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	Ys[buf][tx][ty] = deltaY[Y_k * OC];

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] //X_n *= IC;
	int X_k = ty + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	float2 xv; int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
	xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
	Xs[buf][ty][tx] = xv; 
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1, OK = GK_slice >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float  a = Ys[buf][ik][ty];
			simdMM2(v0, a, b);
		}
		buf ^= 1;

		//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = deltaY[Y_k * OC];

		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] //X_n *= IC;
		int X_k = (ok << LB) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		float2 xv; int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		xv.x = (LOAD_X(fh0, fw0) ? X[X0 + xoffset] : 0);
		xv.y = (LOAD_X(fh1, fw1) ? X[X1 + xoffset] : 0);
		Xs[buf][ty][tx] = xv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Xs[buf][ik][tx];
		float  a = Ys[buf][ik][ty];
		simdMM2(v0, a, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
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


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_GEMMSK_KERNEL_1_1
#define DECONV3D_DW_GEMMSK_KERNEL_1_1

//LB = 4: Size = 0.560303, Time = 6.504 msec, Performace = 185 GFlop/s
//LB = 3: Size = 0.560303, Time = 10.156 msec, Performace = 118.476 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_1_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x; 

	bool buf = 0;
	__shared__ float Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float Xs[2][1 << LB][(1 << LB) + 1];

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
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	deltaY += oc0;

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	const int X0 = (fh0*IW + fw0)*IC + ic0;

	//load 1 element  from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx + GK_start;
	Ys[buf][tx][ty] = deltaY[Y_k * OC];

	//load 1 element  from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
	int OH_OW = OH * OW;
	int X_k = ty + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (lx0 ? X[X0 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float v = 0;
	for (int ok = 1, OK = GK_slice >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float b = Xs[buf][ik][tx];
			float a = Ys[buf][ik][ty];
			v += b * a;
		}
		buf ^= 1;

		//load 1 element  from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = deltaY[Y_k * OC];

		//load 1 element  from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
		int X_k = (ok << LB) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (lx0 ? X[X0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float b = Xs[buf][ik][tx];
		float a = Ys[buf][ik][ty];
		v += b * a;
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		//load 1 element from deltaY
		float a = deltaY[(k + GK_start) * OC];

		float b;//load 1 element from X
		int X_k = k + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b = (lx0 ? X[X0 + xoffset] : 0);

		v += b * a;
	}
	//when GK % STEP != 0--------------------------------------------

	//j = *fh * FW + fw)*IC + ic
	deltaW[oc0 * Wstride + j0] = v;
}

#endif

#endif
