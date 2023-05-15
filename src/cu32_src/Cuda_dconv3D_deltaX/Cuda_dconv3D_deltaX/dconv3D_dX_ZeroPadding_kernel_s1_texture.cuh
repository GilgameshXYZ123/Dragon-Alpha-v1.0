#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_S1_TEXTURE_H
#define DECONV3D_DX_ZERO_PADDING_KERNEL_S1_TEXTURE_H

//Unsparse Matrix Method:
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_S1_TEXTURE_CALL
#define DECONV3D_DX_ZERO_PADDING_KERNEL_S1_TEXTURE_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]==============================================
#define k88s1_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//======[FH, FW is power of 2]================================
#define k88s1_W2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, LFH, LFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,LFH,LFW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//======[PURE: direct conv]===================================
#define k48s1_pure_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_4_8_s1_pure_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

#define k44s1_pure_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_4_4_s1_pure_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//=====[Fixed Template Kernel]================================
#define f88s1x4_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_fernel_8_8_s1_x4_texture<LB, (1<<LB>>1), FH, FW, (FH*FW)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC), deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-1-pw),\
			ic_index,j_index)

#define f88s1_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_fernel_8_8_s1_texture<LB, (1<<LB>>1), FH, FW, (FH*FW)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC), deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-1-pw),\
			ic_index,j_index)

//======[FH = FW = 3]=========================================
#define k88s1W3x4_oc2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W3_x4_OC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,(IH*IW),IW, IC,LOC, (2-ph),(2-pw),\
			ic_index,j_index)

#define k88s1W3_oc2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W3_OC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,(IH*IW),IW, IC,LOC, (2-ph),(2-pw),\
			ic_index,j_index)

//======[FH = FW = 5]==============================================
#define k88s1W5x4_oc2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W5_x4_OC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,(IH*IW),IW, IC,LOC, (4-ph),(4-pw),\
			ic_index,j_index)

#define k88s1W5_oc2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W5_OC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,(IH*IW),IW, IC,LOC, (4-ph),(4-pw),\
			ic_index,j_index)

#endif


//======[Common]==============================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_TEXTURE

//LB = 4: Size = 1.125, Time = 1.876 msec, Performace = 1287.8 GFlop/s
//LB = 4: Size = 1, Time = 1.562 msec, Performace = 1374.83 GFlop/s
//LB = 3: Size = 1, Time = 1.878 msec, Performace = 1143.49 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_texture(
	cudaTextureObject_t  deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_oc, Wr_fh_fw; get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	int woffset = (W_oc*FH_FW + Wr_fh_fw)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_fh, Y_fw, Y_oc; get_dY_oc_fh_fw(Y_k, Y_oc, Y_fh, Y_fw);
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc, Wr_fh_fw; get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		int woffset = (W_oc*FH_FW + Wr_fh_fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_fh, Y_fw, Y_oc; get_dY_oc_fh_fw(Y_k, Y_oc, Y_fh, Y_fw);
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//======[FH, FW is power of 2]================================
//(BLOCK_X*8, BLOCK_Y*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W2POW_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W2POW_TEXTURE

//LB = 4: Size = 1, Time = 1.376 msec, Performace = 1560.67 GFlop/s
//LB = 3: Size = 1, Time = 1.532 msec, Performace = 1401.75 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int LFH, int LFW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * OC
	const int LFH_FW = LFH + LFW, FH_FW_m1 = (1 << LFH_FW) - 1;
	const int FW_m1 = (1 << LFW) - 1;
	const int GK = OC << LFH_FW;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_fh, Y_fw, Y_oc; get_dY_oc_fh_fw_W2pow(Y_k, Y_oc, Y_fh, Y_fw);
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_oc, Wr_fh_fw; get_W_oc_fh_fw_W2pow(W_k, W_oc, Wr_fh_fw);
	int woffset = ((W_oc << LFH_FW) + Wr_fh_fw)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_fh, Y_fw, Y_oc; get_dY_oc_fh_fw_W2pow(Y_k, Y_oc, Y_fh, Y_fw);
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc, Wr_fh_fw; get_W_oc_fh_fw_W2pow(W_k, W_oc, Wr_fh_fw);
		int woffset = ((W_oc << LFH_FW) + Wr_fh_fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0;  *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2;  *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4;  *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6;  *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8;  *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//======[PURE: direct conv]===================================
//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_4_8_S1_PURE_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_4_8_S1_PURE_TEXTURE

//LB = 4: Size = 1, Time = 1.678 msec, Performace = 1279.79 GFlop/s
//LB = 3: Size = 1, Time = 1.826 msec, Performace = 1176.06 GFlop/s
//LB = 4: Size = 1.125, Time = 1.832 msec, Performace = 1318.73 GFlop/s
//LB = 3: Size = 1.125, Time = 2.016 msec, Performace = 1198.37 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_4_8_s1_pure_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	const int tic0 = ((ty & 1) << 1) + ic0;
	const int Wstride = FH * FW * IC; W += Wstride - IC + tic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0);
	float4 v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0);
	float4 v7 = make_float4(0, 0, 0, 0);

	int Yoffset = 0;
	const int OOC = (OC << 1 >> LB), SY = (OW - FW)*OC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	for (int fh = 0; fh < FH; fh++, Yoffset += SY) {
		for (int fw = 0; fw < FW; fw++, Yoffset += OC, W -= IC)
		{
			//load 4 elements from deltaY
			float4 y; int Y_oc = Yoffset + tx - ((tx >= STEP) << LB >> 1);
			y.x = tex1Dfetch<float>(deltaY, Y0 + Y_oc);
			y.y = tex1Dfetch<float>(deltaY, Y1 + Y_oc);
			y.z = tex1Dfetch<float>(deltaY, Y2 + Y_oc);
			y.w = tex1Dfetch<float>(deltaY, Y3 + Y_oc);
			bool ly0 = LOAD_Y(tih0, tiw0, fh, fw); zero_float(y.x, ly0, y.x);
			bool ly1 = LOAD_Y(tih1, tiw1, fh, fw); zero_float(y.y, ly1, y.y);
			bool ly2 = LOAD_Y(tih2, tiw2, fh, fw); zero_float(y.z, ly2, y.z);
			bool ly3 = LOAD_Y(tih3, tiw3, fh, fw); zero_float(y.w, ly3, y.w);
			Ys[buf][tx][ty] = y;

			//load 2 elements from W
			int W_oc = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * Wstride);
			__syncthreads();

			for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll 
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
					float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

					//transposed compute core: (W * dY)^T
					simdMM4(v0, y0.x, w0);
					simdMM4(v1, y0.y, w0);
					simdMM4(v2, y0.z, w0);
					simdMM4(v3, y0.w, w0);
					simdMM4(v4, y1.x, w0);
					simdMM4(v5, y1.y, w0);
					simdMM4(v6, y1.z, w0);
					simdMM4(v7, y1.w, w0);
				}
				buf ^= 1;

				//load 4 elements from deltaY
				float4 y; int Y_oc = Yoffset + ((ooc - (tx >= STEP)) << LB >> 1) + tx;
				y.x = tex1Dfetch<float>(deltaY, Y0 + Y_oc);
				y.y = tex1Dfetch<float>(deltaY, Y1 + Y_oc);
				y.z = tex1Dfetch<float>(deltaY, Y2 + Y_oc);
				y.w = tex1Dfetch<float>(deltaY, Y3 + Y_oc);
				zero_float(y.x, ly0, y.x);
				zero_float(y.y, ly1, y.y);
				zero_float(y.z, ly2, y.z);
				zero_float(y.w, ly3, y.w);
				Ys[buf][tx][ty] = y;

				//load 2 elements from W
				int W_oc = ((ooc << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * Wstride);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
				float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

				//transposed compute core: (W * dY)^T
				simdMM4(v0, y0.x, w0);
				simdMM4(v1, y0.y, w0);
				simdMM4(v2, y0.z, w0);
				simdMM4(v3, y0.w, w0);
				simdMM4(v4, y1.x, w0);
				simdMM4(v5, y1.y, w0);
				simdMM4(v6, y1.z, w0);
				simdMM4(v7, y1.w, w0);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0;
	*(float4*)(deltaX + j1) = v1;
	*(float4*)(deltaX + j2) = v2;
	*(float4*)(deltaX + j3) = v3;
	*(float4*)(deltaX + j4) = v4;
	*(float4*)(deltaX + j5) = v5;
	*(float4*)(deltaX + j6) = v6;
	*(float4*)(deltaX + j7) = v7;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_4_4_S1_PURE_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_4_4_S1_PURE_TEXTURE

//LB = 4: Size = 1.125, Time = 2.198 msec, Performace = 1099.14  GFlop/s
//LB = 3: Size = 1.125, Time = 2.488 msec, Performace =  971.029 GFlop/s
//LB = 4: Size = 1, Time = 1.984 msec, Performace = 1082.4  GFlop/s
//LB = 3: Size = 1, Time = 2.226 msec, Performace =  964.727 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_4_4_s1_pure_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	const int tic0 = ((ty & 1) << 1) + ic0;
	const int Wstride = FH * FW * IC; W += Wstride - IC + tic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	const int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);

	int Yoffset = 0;
	const int OOC = (OC << 1 >> LB), SY = (OW - FW)*OC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	for (int fh = 0; fh < FH; fh++, Yoffset += SY) {
		for (int fw = 0; fw < FW; fw++, Yoffset += OC, W -= IC)
		{
			//load 2 elements from deltaY
			float2 y; int Y_oc = Yoffset + (tx >> 1);
			y.x = tex1Dfetch<float>(deltaY, Y0 + Y_oc);
			y.y = tex1Dfetch<float>(deltaY, Y1 + Y_oc);
			bool ly0 = LOAD_Y(tih0, tiw0, fh, fw); zero_float(y.x, ly0, y.x);
			bool ly1 = LOAD_Y(tih1, tiw1, fh, fw); zero_float(y.y, ly1, y.y);
			Ys[buf][Ys_x][Ys_y] = y;

			//load 2 elements from W
			int W_oc = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * Wstride);
			__syncthreads();

			for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll 
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
					float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

					//transposed compute core: (W * dY)^T
					simdMM4(v0, y.x, w);
					simdMM4(v1, y.y, w);
					simdMM4(v2, y.z, w);
					simdMM4(v3, y.w, w);
				}
				buf ^= 1;

				//load 2 elements from deltaY
				float2 y; int Y_oc = Yoffset + (((ooc << LB) + tx) >> 1);
				y.x = tex1Dfetch<float>(deltaY, Y0 + Y_oc);
				y.y = tex1Dfetch<float>(deltaY, Y1 + Y_oc);
				zero_float(y.x, ly0, y.x);
				zero_float(y.y, ly1, y.y);
				Ys[buf][Ys_x][Ys_y] = y;

				//load 2 elements from W
				int W_oc = ((ooc << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * Wstride);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
				float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

				//transposed compute core: (W * dY)^T
				simdMM4(v0, y.x, w);
				simdMM4(v1, y.y, w);
				simdMM4(v2, y.z, w);
				simdMM4(v3, y.w, w);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * IC + ic0; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;

	*(float4*)(deltaX + j0) = v0;
	*(float4*)(deltaX + j1) = v1;
	*(float4*)(deltaX + j2) = v2;
	*(float4*)(deltaX + j3) = v3;
}

#endif


//=====[Fixed Template Kernel]================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_FERNEL_8_8_S1_X4_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_FERNEL_8_8_S1_X4_TEXTURE

//[FH, FW] = [3, 3]
//LB = 4: Size = 1.125, Time = 1.552 msec, Performace = 1556.65 GFlop/s
//LB = 3: Size = 1.125, Time = 1.788 msec, Performace = 1351.19 GFlop/s
//[FH, FW] = [5, 5]
//LB = 4: Size = 1.5625, Time = 2.094 msec, Performace = 1602.41 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.396 msec, Performace = 1400.44 GFlop/s
template<int LB, int STEP, int FH, int FW, int FH_FW>
__global__ void zeroPadding_fernel_8_8_s1_x4_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	int Y1 = ((tn0*OH + tih0)*OW + tiw1)*OC;

	//prepare for GK = FH * FW * OC
	const int GK = FH_FW * OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_fh, Y_fw, Y_oc; get_dY_oc_fh_fw(Y_k, Y_oc, Y_fh, Y_fw);
	int yoffset = Y1 + ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	Ys[buf][tx][ty] = S1_SaveYs4x_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW, OC,
		tih0, tiw0, tiw1, tiw2, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int woffset = (((W_k / FH_FW) << 1)*FH_FW - W_k)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_fh, Y_fw, Y_oc; get_dY_oc_fh_fw(Y_k, Y_oc, Y_fh, Y_fw);
		int yoffset = Y1 + ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		Ys[buf][tx][ty] = S1_SaveYs4x_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW, OC,
			tih0, tiw0, tiw1, tiw2, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = (((W_k / FH_FW) << 1)*FH_FW - W_k)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0;  *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2;  *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4;  *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6;  *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8;  *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_FERNEL_8_8_S1_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_FERNEL_8_8_S1_TEXTURE

//[FH, FW] = [3, 3]
//LB = 4: Size = 1.125, Time = 1.574 msec, Performace = 1534.89 GFlop/s
//LB = 3: Size = 1.125, Time = 1.814 msec, Performace = 1331.82 GFlop/s
//[FH, FW] = [5, 5]
//LB = 4: Size = 1.5625, Time = 2.136 msec, Performace = 1570.9 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.416 msec, Performace = 1388.84 GFlop/s
template<int LB, int STEP, int FH, int FW, int FH_FW>
__global__ void zeroPadding_fernel_8_8_s1_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int GK = FH_FW * OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_fh, Y_fw, Y_oc; get_dY_oc_fh_fw(Y_k, Y_oc, Y_fh, Y_fw);
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int woffset = (((W_k / FH_FW) << 1)*FH_FW - W_k)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_fh, Y_fw, Y_oc; get_dY_oc_fh_fw(Y_k, Y_oc, Y_fh, Y_fw);
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = (((W_k / FH_FW) << 1)*FH_FW - W_k)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0;  *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2;  *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4;  *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6;  *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8;  *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//=====[FH = FW = 3]===========================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3_X4_OC_2POW_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3_X4_OC_2POW_TEXTUE

//LB = 4: Size = 0.5625, Time = 0.88 msec, Performace = 1372.68 GFlop/s
//LB = 3: Size = 0.5625, Time = 0.96 msec, Performace = 1258.29 GFlop/s
//LB = 4: Size = 1.125, Time = 1.536 msec, Performace = 1572.86 GFlop/s
//LB = 3: Size = 1.125, Time = 1.736 msec, Performace = 1391.66 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W3_x4_OC2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	int Y1 = ((tn0*OH + tih0)*OW + tiw1) << LOC;

	//prepare for GK = FH * FW * OC
	const int GK = 9 << LOC, LOC_m1 = (1 << LOC) - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int YIdx = Y_k >> LOC; char Yfhw = YIDX_W33[YIdx];
	int Y_fh = Yfhw >> 2, Y_fw = Yfhw & 3;
	int yoffset = Y1 + ((Y_fh*OW + Y_fw - YIdx) << LOC) + Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4x_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW, (1 << LOC),
		tih0, tiw0, tiw1, tiw2, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int WIdx = W_k >> LOC, W_oc = W_k & LOC_m1;
	int woffset = (W_oc * 9 + WIDX_W33[WIdx])*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int YIdx = Y_k >> LOC; char Yfhw = YIDX_W33[YIdx];
		int Y_fh = Yfhw >> 2, Y_fw = Yfhw & 3;
		int yoffset = Y1 + ((Y_fh*OW + Y_fw - YIdx) << LOC) + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4x_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW, (1 << LOC),
			tih0, tiw0, tiw1, tiw2, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int WIdx = W_k >> LOC, W_oc = W_k & LOC_m1;
		int woffset = (W_oc * 9 + WIDX_W33[WIdx])*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3_OC_2POW_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3_OC_2POW_TEXTURE

//LB = 4: Size = 1.125, Time = 1.552 msec, Performace = 1556.65 GFlop/s
//LB = 3: Size = 1.125, Time = 1.742 msec, Performace = 1386.86 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W3_OC2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0) << LOC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1) << LOC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2) << LOC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3) << LOC;

	//prepare for GK = FH * FW * OC
	const int GK = 9 << LOC, OC_m1 = (1 << LOC) - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int YIdx = Y_k >> LOC; char Yfhw = YIDX_W33[YIdx];
	int Y_fh = Yfhw >> 2, Y_fw = Yfhw & 3;
	int yoffset = ((Y_fh*OW + Y_fw - YIdx) << LOC) + Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int WIdx = W_k >> LOC, W_oc = W_k & OC_m1;
	int woffset = (W_oc * 9 + WIDX_W33[WIdx])*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int YIdx = Y_k >> LOC; char Yfhw = YIDX_W33[YIdx];
		int Y_fh = Yfhw >> 2, Y_fw = Yfhw & 3;
		int yoffset = ((Y_fh*OW + Y_fw - YIdx) << LOC) + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int WIdx = W_k >> LOC, W_oc = W_k & OC_m1;
		int woffset = (W_oc * 9 + WIDX_W33[WIdx])*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//=====[FH = FW = 5]===========================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W5_X4_OC2POW_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W5_X4_OC2POW_TEXTURE

//k88<4>: Size = 1.5625, Time = 2.416 msec, Performace = 1388.84 GFlop/s
//k88<3>: Size = 1.5625, Time = 2.  9 msec, Performace = 1157.05 GFlop/s
//LB = 4: Size = 1.5625, Time = 2.05  msec, Performace = 1636.8 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.326 msec, Performace = 1442.58 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W5_x4_OC2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 5
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	int Y1 = ((tn0*OH + tih0)*OW + tiw1) << LOC;

	//prepare for GK = FH * FW * OC
	const int GK = 25 << LOC, OC_m1 = (1 << LOC) - 1;

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int WIdx = W_k >> LOC, W_oc = W_k & OC_m1;
	int woffset = (W_oc * 25 + WIDX_W55[WIdx])*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	int YIdx = dY_k >> LOC; char Yfhw = YIDX_W55[YIdx];
	int dY_fh = Yfhw >> 3, dY_fw = Yfhw & 7;
	int yoffset = Y1 + ((dY_fh*OW + dY_fw - YIdx) << LOC) + dY_k;
	Ys[buf][tx][ty] = S1_SaveYs4x_tex(deltaY, dY_fh, dY_fw, yoffset, OH, OW, (1 << LOC),
		tih0, tiw0, tiw1, tiw2, tiw3);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int WIdx = W_k >> LOC, W_oc = W_k & OC_m1;
		int woffset = (W_oc * 25 + WIDX_W55[WIdx])*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int YIdx = dY_k >> LOC; char Yfhw = YIDX_W55[YIdx];
		int dY_fh = Yfhw >> 3, dY_fw = Yfhw & 7;
		int yoffset = Y1 + ((dY_fh*OW + dY_fw - YIdx) << LOC) + dY_k;
		Ys[buf][tx][ty] = S1_SaveYs4x_tex(deltaY, dY_fh, dY_fw, yoffset, OH, OW, (1 << LOC),
			tih0, tiw0, tiw1, tiw2, tiw3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W5_OC2POW_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W5_OC2POW_TEXTURE

//LB = 4: Size = 1.5625, Time = 2.086 msec, Performace = 1608.55 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.318 msec, Performace = 1447.56 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W5_OC2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 5
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0) << LOC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1) << LOC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2) << LOC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3) << LOC;

	//prepare for GK = FH * FW * OC
	const int GK = 25 << LOC, OC_m1 = (1 << LOC) - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	int YIdx = dY_k >> LOC; char Yfhw = YIDX_W55[YIdx];
	int dY_fh = Yfhw >> 3, dY_fw = Yfhw & 7;
	int yoffset = ((dY_fh*OW + dY_fw - YIdx) << LOC) + dY_k;
	Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, dY_fh, dY_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int WIdx = W_k >> LOC, W_oc = W_k & OC_m1;
	int woffset = (W_oc * 25 + WIDX_W55[WIdx])*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
			simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
			simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
			simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
			simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
			simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int YIdx = dY_k >> LOC; char Yfhw = YIDX_W55[YIdx];
		int dY_fh = Yfhw >> 3, dY_fw = Yfhw & 7;
		int yoffset = ((dY_fh*OW + dY_fw - YIdx) << LOC) + dY_k;
		Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, dY_fh, dY_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int WIdx = W_k >> LOC, W_oc = W_k & OC_m1;
		int woffset = (W_oc * 25 + WIDX_W55[WIdx])*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif

#endif