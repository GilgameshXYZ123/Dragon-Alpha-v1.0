#pragma once

#ifndef DECONV3D_DW_GEMM_KERNEL_TEXTURE_H
#define DECONV3D_DW_GEMM_KERNEL_TEXTURE_H


#ifndef DECONV3D_DW_GEMM_KERNEL_TEXTURE_CALL
#define DECONV3D_DW_GEMM_KERNEL_TEXTURE_CALL

//LB = log2(BLOCK_SIZE)
#define kGemm88_tex(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_8_8_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kGemm88_ohw2pow_tex(stream, LB, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_8_8_OHW2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//=============================================================================
#define kGemm44_tex(stream, LB, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_Gemm_4_4_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2)
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_8_8_TEXTURE
#define DECONV3D_DW_GEMM_KERNEL_8_8_TEXTURE

//LB = 4: Size = 1, Time = 1.85  msec, Performace = 1160.8   GFlop/s
//LB = 3: Size = 1, Time = 2.248 msec, Performace =  955.286 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_8_8_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	const int toc0 = oc0 + ((tx >= STEP) << 2);

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	get_fh_fw_ic(tj1, tfh1, tfw1, tic1);
	get_fh_fw_ic(tj2, tfh2, tfw2, tic2);
	get_fh_fw_ic(tj3, tfh3, tfw3, tic3);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	tfh1 = tfh1 - oph, tfw1 = tfw1 - opw;
	tfh2 = tfh2 - oph, tfw2 = tfw2 - opw;
	tfh3 = tfh3 - oph, tfw3 = tfw3 - opw;
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1;
	const int Xoffset2 = (tfh2*IW + tfw2)*IC + tic2;
	const int Xoffset3 = (tfh3*IW + tfw3)*IC + tic3;

	//prepare for GK = N * OH * OW
	const int OH_OW = OH * OW, GK = N * OH_OW;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + toc0]);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
	bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
	bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	zero_float(Xs[buf][ty][tx].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
	zero_float(Xs[buf][ty][tx].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
	zero_float(Xs[buf][ty][tx].z, lx2, tex1Dfetch<float>(X, Xoffset2 + xoffset));
	zero_float(Xs[buf][ty][tx].w, lx3, tex1Dfetch<float>(X, Xoffset3 + xoffset));
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
			float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];
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

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + toc0]);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
		bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
		bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		zero_float(Xs[buf][ty][tx].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
		zero_float(Xs[buf][ty][tx].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
		zero_float(Xs[buf][ty][tx].z, lx2, tex1Dfetch<float>(X, Xoffset2 + xoffset));
		zero_float(Xs[buf][ty][tx].w, lx3, tex1Dfetch<float>(X, Xoffset3 + xoffset));
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];
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

	const int FH_FW_IC = FH * FW_IC;
	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), OH, OW is power of 2
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_8_8_OHW2POW_TEXTURE
#define DECONV3D_DW_GEMM_KERNEL_8_8_OHW2POW_TEXTURE

//LB = 4: Size = 1, Time = 1.664 msec, Performace = 1290.55 GFlop/s
//LB = 3: Size = 1, Time = 1.888 msec, Performace = 1137.44 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_8_8_OHW2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	const int toc0 = oc0 + ((tx >= STEP) << 2);

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	get_fh_fw_ic(tj1, tfh1, tfw1, tic1);
	get_fh_fw_ic(tj2, tfh2, tfw2, tic2);
	get_fh_fw_ic(tj3, tfh3, tfw3, tic3);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	tfh1 = tfh1 - oph, tfw1 = tfw1 - opw;
	tfh2 = tfh2 - oph, tfw2 = tfw2 - opw;
	tfh3 = tfh3 - oph, tfw3 = tfw3 - opw;
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1;
	const int Xoffset2 = (tfh2*IW + tfw2)*IC + tic2;
	const int Xoffset3 = (tfh3*IW + tfw3)*IC + tic3;

	//prepare for GK = N * OH * OW
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK = N << LOH_OW;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + toc0]);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
	bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
	bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	zero_float(Xs[buf][ty][tx].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
	zero_float(Xs[buf][ty][tx].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
	zero_float(Xs[buf][ty][tx].z, lx2, tex1Dfetch<float>(X, Xoffset2 + xoffset));
	zero_float(Xs[buf][ty][tx].w, lx3, tex1Dfetch<float>(X, Xoffset3 + xoffset));
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
			float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];
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

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + toc0]);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
		bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
		bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		zero_float(Xs[buf][ty][tx].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
		zero_float(Xs[buf][ty][tx].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
		zero_float(Xs[buf][ty][tx].z, lx2, tex1Dfetch<float>(X, Xoffset2 + xoffset));
		zero_float(Xs[buf][ty][tx].w, lx3, tex1Dfetch<float>(X, Xoffset3 + xoffset));
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];
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

	const int FH_FW_IC = FH * FW_IC;
	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2)
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMM_KERNEL_4_4_TEXTURE
#define DECONV3D_DW_GEMM_KERNEL_4_4_TEXTURE

//LB = 4: Size = 1, Time = 2.606 msec, Performace = 824.054 GFlop/s
//LB = 3: Size = 1, Time = 3.342 msec, Performace = 630.291 GFlop/s
template<int LB, int STEP>
__global__ void kernel_Gemm_4_4_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int toc0 = ((tx & 1) << 1) + oc0;

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	get_fh_fw_ic(tj1, tfh1, tfw1, tic1);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	tfh1 = tfh1 - oph, tfw1 = tfw1 - opw;
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1;

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
	bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
	Xs[buf][Xs_y][Xs_x].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
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
			float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);

			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k* OC + toc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		Xs[buf][Xs_y][Xs_x].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);

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

#endif