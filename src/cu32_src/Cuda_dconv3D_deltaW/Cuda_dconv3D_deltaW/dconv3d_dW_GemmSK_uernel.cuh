#pragma once

#ifndef DECONV3D_DW_GEMMSK_UERNEL_H
#define DECONV3D_DW_GEMMSK_UERNEL_H

//Split K to improve parallism:
//	(1) FH * FW >= 2
//	(2) GN = OC:           GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW: GM%4 == 0, GM >= 8
//	(4) GK = N  * OH * OW: GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
//	(6) ph = oph, pw = opw
//	(7) IC % 4 == 0
//	(8) OW % 16 == 0 && GK_slice % 16 == 0
#ifndef DECONV3D_DW_GEMMSK_UERNEL_CALL
#define DECONV3D_DW_GEMMSK_UERNEL_CALL

//LB = log2(BLOCK_SIZE)
//GZ = gridDim.z

//=======[N is power of 2]==============================================
#define uGemmSK88_n2pow(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GM)\
	uernel_GemmSK_8_8_N2pow<LB, (1<<LB>>1), (1<<LB), ((1<<LB>>1) - 1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			  LN, IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//======[OH, OW is power of 2]==========================================
#define uGemmSK88_ohw2pow_LB4(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	uernel_GemmSK_8_8_OHW2pow_LB4<LB, (1<<LB>>1), (1<<LB), ((1<<LB>>1)-1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#define uGemmSK88_ohw2pow_LB3(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	uernel_GemmSK_8_8_OHW2pow_LB3<LB, (1<<LB>>1), (1<<LB), ((1<<LB>>1)-1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

#endif


//======[OH, OW is power of 2]==========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OW % BLOCK_SIZE == 0, IC % 4 == 0
//LB = 4: OW % 16 == 0, GK_slice % 16 == 0
#ifndef DECONV3D_DW_GEMMSK_UERNEL_8_8_OHW_2POW_LB4
#define DECONV3D_DW_GEMMSK_UERNEL_8_8_OHW_2POW_LB4

//synchronized: LB = 4:
//OH, OW = 32, N =  4: Size = 1.125, Time = 1.644 msec, Performace = 1469.54 GFlop/s
//OH, OW = 16, N = 16: Size = 1.125, Time = 1.628 msec, Performace = 1483.98 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void uernel_GemmSK_8_8_OHW2pow_LB4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

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
	X += (tfh0*IW + tfw0)*IC + tic0;//X[0, tfh0, tfw0, tic0]

	const int LOH_OW = LOH + LOW;
	const int OW_m1 = (1 << LOW) - 1;
	const int OH_OW_m1 = (1 << LOH_OW) - 1;
	const int IC_sw = IC * sw;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((ty & STEP_m1) << 1) + GK_start;
	uget_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow0, X_ow1);
	int xoffset = ((X_n*IH + X_oh)*IW + X_ow0)*IC;
	bool lx = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh);
	bool lx0 = lx & (tfw0 >= -X_ow0) && (tfw0 < IW - X_ow0);
	bool lx1 = lx & (tfw0 >= -X_ow1) && (tfw0 < IW - X_ow1);
	Xs[buf][(ty << 1)][tx] = (lx0 ? *(float4*)(X + xoffset) : F32_4_0);
	Xs[buf][(ty << 1) + 1][tx] = (lx1 ? *(float4*)(X + xoffset + IC_sw) : F32_4_0);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_k = ((tx & STEP_m1) << 1) + GK_start;
	int yoffset = Y_k * OC;
	Ys[buf][(tx << 1)][ty] = *(float4*)(deltaY + yoffset);
	Ys[buf][(tx << 1) + 1][ty] = *(float4*)(deltaY + yoffset + OC);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP2][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP2][tx];

			simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ((ty & STEP_m1) << 1) + GK_start;
		uget_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow0, X_ow1);
		int xoffset = ((X_n*IH + X_oh)*IW + X_ow0)*IC;
		bool lx = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh);
		bool lx0 = lx & (tfw0 >= -X_ow0) && (tfw0 < IW - X_ow0);
		bool lx1 = lx & (tfw0 >= -X_ow1) && (tfw0 < IW - X_ow1);
		Xs[buf][(ty << 1)][tx] = (lx0 ? *(float4*)(X + xoffset) : F32_4_0);
		Xs[buf][(ty << 1) + 1][tx] = (lx1 ? *(float4*)(X + xoffset + IC_sw) : F32_4_0);

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = (ok << LB) + ((tx & STEP_m1) << 1) + GK_start;
		int yoffset = Y_k * OC;
		Ys[buf][(tx << 1)][ty] = *(float4*)(deltaY + yoffset);
		Ys[buf][(tx << 1) + 1][ty] = *(float4*)(deltaY + yoffset + OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP2][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP2][tx];

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % BLOCK_SIZE == 0, IC % 4 == 0
//LB = 3: OW % 8 == 0, GK_slice % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_UERNEL_8_8_OHW_2POW_LB3
#define DECONV3D_DW_GEMMSK_UERNEL_8_8_OHW_2POW_LB3

//synchronized: LB = 3:
//OH, OW = 32, N =  4: Size = 1.125, Time = 1.726 msec, Performace = 1399.72 GFlop/s
//OH, OW = 16, N = 16: Size = 1.125, Time = 1.72  msec, Performace = 1404.6  GFlop/s
//OH, OW =  8, N = 64: Size = 1.125, Time = 1.716 msec, Performace = 1407.88 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void uernel_GemmSK_8_8_OHW2pow_LB3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

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
	X += (tfh0*IW + tfw0)*IC + tic0;//X[0, tfh0, tfw0, tic0]

	const int LOH_OW = LOH + LOW;
	const int OW_m1 = (1 << LOW) - 1;
	const int OH_OW_m1 = (1 << LOH_OW) - 1;
	const int IC_sw = IC * sw;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((ty & STEP_m1) << 1) + GK_start;
	uget_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow0, X_ow1);
	int xoffset = ((X_n*IH + X_oh)*IW + X_ow0)*IC;
	bool lx = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh);
	bool lx0 = lx & (tfw0 >= -X_ow0) && (tfw0 < IW - X_ow0);
	bool lx1 = lx & (tfw0 >= -X_ow1) && (tfw0 < IW - X_ow1);
	Xs[buf][(ty << 1)][tx] = (lx0 ? *(float4*)(X + xoffset) : F32_4_0);
	Xs[buf][(ty << 1) + 1][tx] = (lx1 ? *(float4*)(X + xoffset + IC_sw) : F32_4_0);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_k = ((tx & STEP_m1) << 1) + GK_start;
	int yoffset = Y_k * OC;
	Ys[buf][(tx << 1)][ty] = *(float4*)(deltaY + yoffset);
	Ys[buf][(tx << 1) + 1][ty] = *(float4*)(deltaY + yoffset + OC);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP2][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP2][tx];

			simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = (ok << LB) + ((tx & STEP_m1) << 1) + GK_start;
		int yoffset = Y_k * OC;
		Ys[buf][(tx << 1)][ty] = *(float4*)(deltaY + yoffset);
		Ys[buf][(tx << 1) + 1][ty] = *(float4*)(deltaY + yoffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ((ty & STEP_m1) << 1) + GK_start;
		uget_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow0, X_ow1);
		int xoffset = ((X_n*IH + X_oh)*IW + X_ow0)*IC;
		bool lx = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh);
		bool lx0 = lx & (tfw0 >= -X_ow0) && (tfw0 < IW - X_ow0);
		bool lx1 = lx & (tfw0 >= -X_ow1) && (tfw0 < IW - X_ow1);
		Xs[buf][(ty << 1)][tx] = (lx0 ? *(float4*)(X + xoffset) : F32_4_0);
		Xs[buf][(ty << 1) + 1][tx] = (lx1 ? *(float4*)(X + xoffset + IC_sw) : F32_4_0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP2][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP2][tx];

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
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


//=======[N is power of 2]==============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), N % BLOCK_SIZE == 0, IC % 4 == 0
//LB = 4: N % 16 == 0, GK_slice % 16 == 0
//LB = 3: N %  8 == 0, GK_slice %  8 == 0
#ifndef DECONV3D_DW_GEMMSK_UERNEL_8_8_N_2POW
#define DECONV3D_DW_GEMMSK_UERNEL_8_8_N_2POW
  
//synchronized:
//OH, OW = 16, N = 16: 
//LB = 4: Size = 1.125, Time = 1.692 msec, Performace = 1427.85 GFlop/s
//LB = 3: Size = 1.125, Time = 2.054 msec, Performace = 1176.2  GFlop/s
//OH, OW =  8, N = 64: 
//LB = 4: Size = 1.125, Time = 1.69  msec, Performace = 1429.54 GFlop/s
//LB = 3: Size = 1.125, Time = 1.762 msec, Performace = 1371.12 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void uernel_GemmSK_8_8_N2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

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
	X += (tfh0*IW + tfw0)*IC + tic0;//X[0, tf0, tfw0, tic0]

	const int OW_N = OW << LN, N_m1 = (1 << LN) - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = ((tx & STEP_m1) << 1) + GK_start;
	int oh = Y_k / (OW_N), ow = (Y_k - oh * OW_N) >> LN; 
	int yoffset = (((Y_k & N_m1)*OH + oh)*OW + ow)*OC; oh *= sh, ow *= sw;
	Ys[buf][(tx << 1)][ty] = *(float4*)(deltaY + yoffset);
	Ys[buf][(tx << 1) + 1][ty] = *(float4*)(deltaY + yoffset + OH * OW * OC);

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ((ty & STEP_m1) << 1) + GK_start;
	int xoffset = (((X_k & (N_m1))*IH + oh)*IW + ow)*IC;
	bool lx = LOAD_X2(tfh0, tfw0, oh, ow);
	Xs[buf][(ty << 1)][tx] = (lx ? *(float4*)(X + xoffset) : F32_4_0);
	Xs[buf][(ty << 1) + 1][tx] = (lx ? *(float4*)(X + xoffset + IH * IW * IC) : F32_4_0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP2][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP2][tx];

			simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		int k = GK_start + (ok << LB);
		int Y_k = k + ((tx & STEP_m1) << 1);
		int X_k = k + ((ty & STEP_m1) << 1);
		
		int oh = Y_k / (OW_N), ow = (Y_k - oh * OW_N) >> LN;
		int yoffset = (((Y_k & N_m1)*OH + oh)*OW + ow)*OC; oh *= sh, ow *= sw;
		int xoffset = (((X_k & N_m1)*IH + oh)*IW + ow)*IC;

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		bool lx = LOAD_X2(tfh0, tfw0, oh, ow);
		Xs[buf][(ty << 1)][tx] = (lx ? *(float4*)(X + xoffset) : F32_4_0);
		Xs[buf][(ty << 1) + 1][tx] = (lx ? *(float4*)(X + xoffset + IH * IW * IC) : F32_4_0);

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		Ys[buf][(tx << 1)][ty] = *(float4*)(deltaY + yoffset);
		Ys[buf][(tx << 1) + 1][ty] = *(float4*)(deltaY + yoffset + OH * OW * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP2][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP2][tx];

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
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

#endif
