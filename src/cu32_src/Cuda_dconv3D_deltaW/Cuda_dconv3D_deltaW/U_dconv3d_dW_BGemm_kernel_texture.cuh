#pragma once

#ifndef DECONV3D_DW_BGEMM_KERNEL_TEXTURE_H
#define DECONV3D_DW_BGEMM_KERNEL_TEXTURE_H

//oph = ph, opw = pw
#ifndef DECONV3D_DW_BGEMM_KERNEL_TEXTURE_CALL
#define DECONV3D_DW_BGEMM_KERNEL_TEXTURE_CALL

//LB = log2(BLOCK_SIZE)
//GZ = gridDim.z

#define UkBGemm88_tex(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	Ukernel_BGemm_8_8_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define UkBGemm88_ohw2pow_tex(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	Ukernel_BGemm_8_8_OHW2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define UkBGemm88_ohw_oic2pow_tex(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, N, LIC, LOC, sh, sw, ph, pw, GN, GM)\
	Ukernel_BGemm_8_8_OHW_OIC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 N, LIC, LOC, sh, sw, ph, pw, oc_index, j_index)

//=========================================================================================
#define UkBGemm84_tex(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	Ukernel_BGemm_8_4_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define UkBGemm84_ohw2pow_tex(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	Ukernel_BGemm_8_4_OHW2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//=========================================================================================
#define UkBGemm48_tex(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	Ukernel_BGemm_4_8_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define UkBGemm48_ohw2pow_tex(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	Ukernel_BGemm_4_8_OHW2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

//=========================================================================================
#define UkBGemm44_tex(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	Ukernel_BGemm_4_4_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define UkBGemm44_ohw2pow_tex(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	Ukernel_BGemm_4_4_OHW2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define UkBGemm44_ohw_oic2pow_tex(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW, N, LIC, LOC, sh, sw, ph, pw, GN, GM)\
	Ukernel_BGemm_4_4_OHW_OIC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, LOH, LOW, deltaW, deltaW_buf, FH, FW,\
			 N, LIC, LOC, sh, sw, ph, pw, oc_index, j_index)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2)
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_8_8_TEXTURE
#define DECONV3D_DW_BGEMM_KERNEL_8_8_TEXTURE

//for GZ = N >> 3
//LB = 4: Size = 1, Time = 1.722 msec, Performace = 1247.09 GFlop/s
//LB = 3: Size = 1, Time = 2.062 msec, Performace = 1041.46 GFlop/s
template<int LB, int STEP>
__global__ void Ukernel_BGemm_8_8_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	int XNoffset = N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	const int X0 = (tfh0*IW + tfw0)*IC + tic0 + XNoffset;
	const int X1 = X0 + 1;
	const int X2 = X0 + 2;
	const int X3 = X0 + 3;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	zero_float(Xs[buf][ty][tx].x, lx0, tex1Dfetch<float>(X, X0 + xoffset));
	zero_float(Xs[buf][ty][tx].y, lx0, tex1Dfetch<float>(X, X1 + xoffset));
	zero_float(Xs[buf][ty][tx].z, lx0, tex1Dfetch<float>(X, X2 + xoffset));
	zero_float(Xs[buf][ty][tx].w, lx0, tex1Dfetch<float>(X, X3 + xoffset));
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
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

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
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		zero_float(Xs[buf][ty][tx].x, lx0, tex1Dfetch<float>(X, X0 + xoffset));
		zero_float(Xs[buf][ty][tx].y, lx0, tex1Dfetch<float>(X, X1 + xoffset));
		zero_float(Xs[buf][ty][tx].z, lx0, tex1Dfetch<float>(X, X2 + xoffset));
		zero_float(Xs[buf][ty][tx].w, lx0, tex1Dfetch<float>(X, X3 + xoffset));
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  *(float4*)(dst + oc0 + 4) = v1;
	*(float4*)(dst + oc1) = v2;  *(float4*)(dst + oc1 + 4) = v3;
	*(float4*)(dst + oc2) = v4;  *(float4*)(dst + oc2 + 4) = v5;
	*(float4*)(dst + oc3) = v6;  *(float4*)(dst + oc3 + 4) = v7;
	*(float4*)(dst + oc4) = v8;  *(float4*)(dst + oc4 + 4) = v9;
	*(float4*)(dst + oc5) = v10; *(float4*)(dst + oc5 + 4) = v11;
	*(float4*)(dst + oc6) = v12; *(float4*)(dst + oc6 + 4) = v13;
	*(float4*)(dst + oc7) = v14; *(float4*)(dst + oc7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2). OH, OW is power of 2
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_8_8_OHW2POW_TEXTURE
#define DECONV3D_DW_BGEMM_KERNEL_8_8_OHW2POW_TEXTURE

//LB = 4: Size = 1, Time = 1.584 msec, Performace = 1355.73 GFlop/s
//LB = 3: Size = 1, Time = 1.792 msec, Performace = 1198.37 GFlop/s
template<int LB, int STEP>
__global__ void Ukernel_BGemm_8_8_OHW2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK_slice = (N_end - N_start) << LOH_OW;
	int XNoffset = N_start * IH*IW*IC; //X[N_start,...]
	deltaY += (N_start << LOH_OW) *OC; //deltaY[N_start,...]
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	const int toc0 = oc0 + ((tx >= STEP) << 2);//(oc4 - oc0)*(tx&1) + oc0

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
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0 + XNoffset;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1 + XNoffset;
	const int Xoffset2 = (tfh2*IW + tfw2)*IC + tic2 + XNoffset;
	const int Xoffset3 = (tfh3*IW + tfw3)*IC + tic3 + XNoffset;

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

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];

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
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];

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
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  *(float4*)(dst + oc0 + 4) = v1;
	*(float4*)(dst + oc1) = v2;  *(float4*)(dst + oc1 + 4) = v3;
	*(float4*)(dst + oc2) = v4;  *(float4*)(dst + oc2 + 4) = v5;
	*(float4*)(dst + oc3) = v6;  *(float4*)(dst + oc3 + 4) = v7;
	*(float4*)(dst + oc4) = v8;  *(float4*)(dst + oc4 + 4) = v9;
	*(float4*)(dst + oc5) = v10; *(float4*)(dst + oc5 + 4) = v11;
	*(float4*)(dst + oc6) = v12; *(float4*)(dst + oc6 + 4) = v13;
	*(float4*)(dst + oc7) = v14; *(float4*)(dst + oc7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), OH, OW, OC, IC is power of 2
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_8_8_OHW_OIC_2POW_TEXTURE
#define DECONV3D_DW_BGEMM_KERNEL_8_8_OHW_OIC_2POW_TEXTURE

//LB = 4: Size = 1, Time = 1.5   msec, Performace = 1431.66 GFlop/s
//LB = 3: Size = 1, Time = 1.748 msec, Performace = 1228.54 GFlop/s
template<int LB, int STEP>
__global__ void Ukernel_BGemm_8_8_OHW_OIC2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int LIC, int LOC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK_slice = (N_end - N_start) << LOH_OW;
	int XNoffset = (N_start * IH*IW) << LIC; //X[N_start,...]
	deltaY += (N_start << LOH_OW) << LOC; //deltaY[N_start,...]
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	const int toc0 = oc0 + ((tx >= STEP) << 2);//(oc4 - oc0)*(tx&1) + oc0

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int FW_IC = FW << LIC, IC_m1 = (1 << LIC) - 1;
	get_fh_fw_ic_IC2pow(tj0, tfh0, tfw0, tic0);
	get_fh_fw_ic_IC2pow(tj1, tfh1, tfw1, tic1);
	get_fh_fw_ic_IC2pow(tj2, tfh2, tfw2, tic2);
	get_fh_fw_ic_IC2pow(tj3, tfh3, tfw3, tic3);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	tfh1 = tfh1 - oph, tfw1 = tfw1 - opw;
	tfh2 = tfh2 - oph, tfw2 = tfw2 - opw;
	tfh3 = tfh3 - oph, tfw3 = tfw3 - opw;
	const int Xoffset0 = ((tfh0*IW + tfw0) << LIC) + tic0 + XNoffset;
	const int Xoffset1 = ((tfh1*IW + tfw1) << LIC) + tic1 + XNoffset;
	const int Xoffset2 = ((tfh2*IW + tfw2) << LIC) + tic2 + XNoffset;
	const int Xoffset3 = ((tfh3*IW + tfw3) << LIC) + tic3 + XNoffset;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	dYs[buf][tx][ty] = *(float4*)(&deltaY[(dY_k << LOC) + toc0]);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
	bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
	bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow) << LIC;
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

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
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
		dYs[buf][tx][ty] = *(float4*)(&deltaY[(dY_k << LOC) + toc0]);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
		bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
		bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow) << LIC;
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
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + ((bz - 1) << LOC) *FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  *(float4*)(dst + oc0 + 4) = v1;
	*(float4*)(dst + oc1) = v2;  *(float4*)(dst + oc1 + 4) = v3;
	*(float4*)(dst + oc2) = v4;  *(float4*)(dst + oc2 + 4) = v5;
	*(float4*)(dst + oc3) = v6;  *(float4*)(dst + oc3 + 4) = v7;
	*(float4*)(dst + oc4) = v8;  *(float4*)(dst + oc4 + 4) = v9;
	*(float4*)(dst + oc5) = v10; *(float4*)(dst + oc5 + 4) = v11;
	*(float4*)(dst + oc6) = v12; *(float4*)(dst + oc6 + 4) = v13;
	*(float4*)(dst + oc7) = v14; *(float4*)(dst + oc7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2)
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_8_4_TEXTURE
#define DECONV3D_DW_BGEMM_KERNEL_8_4_TEXTURE

//for GZ = N >> 3
//LB = 4: Size = 1, Time = 1.956 msec, Performace = 1097.9 GFlop/s
//LB = 3: Size = 1, Time = 2.512 msec, Performace =  854.89 GFlop/s
template<int LB, int STEP>
__global__ void Ukernel_BGemm_8_4_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	int XNoffset = N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	const int toc0 = oc0 + ((tx >= STEP) << 2);

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	get_fh_fw_ic(tj1, tfh1, tfw1, tic1);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	tfh1 = tfh1 - oph, tfw1 = tfw1 - opw;
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0 + XNoffset;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1 + XNoffset;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + toc0]);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	zero_float(Xs[buf][Xs_y][Xs_x].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
	zero_float(Xs[buf][Xs_y][Xs_x].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
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
			float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];
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
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + toc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		zero_float(Xs[buf][Xs_y][Xs_x].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
		zero_float(Xs[buf][Xs_y][Xs_x].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];
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
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;
	*(float4*)(dst + oc1) = v2;
	*(float4*)(dst + oc2) = v4;
	*(float4*)(dst + oc3) = v6;
	*(float4*)(dst + oc4) = v8;
	*(float4*)(dst + oc5) = v10;
	*(float4*)(dst + oc6) = v12;
	*(float4*)(dst + oc7) = v14;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2), OH, OW is power of 2
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_8_4_OHW_2POW_TEXTURE
#define DECONV3D_DW_BGEMM_KERNEL_8_4_OHW_2POW_TEXTURE

//for GZ = N >> 3
//LB = 4: Size = 1, Time = 1.77  msec, Performace = 1213.27 GFlop/s
//LB = 3: Size = 1, Time = 2.042 msec, Performace = 1051.66 GFlop/s
template<int LB, int STEP>
__global__ void Ukernel_BGemm_8_4_OHW2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK_slice = (N_end - N_start) << LOH_OW;
	int XNoffset = N_start * IH*IW*IC; //X[N_start,...]
	deltaY += (N_start << LOH_OW) *OC; //deltaY[N_start,...]
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	const int toc0 = oc0 + ((tx >= STEP) << 2);

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	get_fh_fw_ic(tj1, tfh1, tfw1, tic1);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	tfh1 = tfh1 - oph, tfw1 = tfw1 - opw;
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0 + XNoffset;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1 + XNoffset;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + toc0]);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	zero_float(Xs[buf][Xs_y][Xs_x].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
	zero_float(Xs[buf][Xs_y][Xs_x].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
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
			float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];
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
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + toc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		zero_float(Xs[buf][Xs_y][Xs_x].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
		zero_float(Xs[buf][Xs_y][Xs_x].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];
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
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;
	*(float4*)(dst + oc1) = v2;
	*(float4*)(dst + oc2) = v4;
	*(float4*)(dst + oc3) = v6;
	*(float4*)(dst + oc4) = v8;
	*(float4*)(dst + oc5) = v10;
	*(float4*)(dst + oc6) = v12;
	*(float4*)(dst + oc7) = v14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2)
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_4_8_TEXTURE
#define DECONV3D_DW_BGEMM_KERNEL_4_8_TEXTURE

//for GZ = N >> 3
//LB = 4: Size = 1, Time = 2.308 msec, Performace = 930.452 GFlop/s
//LB = 3: Size = 1, Time = 2.656 msec, Performace = 808.541 GFlop/s
template<int LB, int STEP>
__global__ void Ukernel_BGemm_4_8_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4  Xs[2][1 << LB][(1 << LB) + 1];//followed k88

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	int XNoffset = N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int toc0 = ((tx & 1) << 1) + oc0;

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
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0 + XNoffset;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1 + XNoffset;
	const int Xoffset2 = (tfh2*IW + tfw2)*IC + tic2 + XNoffset;
	const int Xoffset3 = (tfh3*IW + tfw3)*IC + tic3 + XNoffset;

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

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&dYs[buf][ik][ty << 1]);
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		}
		buf ^= 1;

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

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&dYs[buf][ik][ty << 1]);
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
	}

	const int FH_FW_IC = FH * FW_IC;
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  *(float4*)(dst + oc0 + 4) = v1;
	*(float4*)(dst + oc1) = v2;  *(float4*)(dst + oc1 + 4) = v3;
	*(float4*)(dst + oc2) = v4;  *(float4*)(dst + oc2 + 4) = v5;
	*(float4*)(dst + oc3) = v6;  *(float4*)(dst + oc3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), OH, OW is power of 2
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_4_8_OHW_2POW_TEXTURE
#define DECONV3D_DW_BGEMM_KERNEL_4_8_OHW_2POW_TEXTURE

//for GZ = N >> 3
//LB = 4: Size = 1, Time = 1.9   msec, Performace = 1130.25 GFlop/s
//LB = 3: Size = 1, Time = 2.172 msec, Performace =  988.713 GFlop/s
template<int LB, int STEP>
__global__ void Ukernel_BGemm_4_8_OHW2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4  Xs[2][1 << LB][(1 << LB) + 1];//followed k88

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK_slice = (N_end - N_start) << LOH_OW;
	int XNoffset = N_start * IH*IW*IC; //X[N_start,...]
	deltaY += (N_start << LOH_OW) *OC; //deltaY[N_start,...]
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int toc0 = ((tx & 1) << 1) + oc0;

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
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0 + XNoffset;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1 + XNoffset;
	const int Xoffset2 = (tfh2*IW + tfw2)*IC + tic2 + XNoffset;
	const int Xoffset3 = (tfh3*IW + tfw3)*IC + tic3 + XNoffset;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
	bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
	bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
	Xs[buf][ty][tx].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
	Xs[buf][ty][tx].z = lx2 * tex1Dfetch<float>(X, Xoffset2 + xoffset);
	Xs[buf][ty][tx].w = lx3 * tex1Dfetch<float>(X, Xoffset3 + xoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = *(float4*)(&dYs[buf][ik][ty << 1]);

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
		bool lx2 = (tfh2 >= -X_oh) && (tfh2 < IH - X_oh) && (tfw2 >= -X_ow) && (tfw2 < IW - X_ow);
		bool lx3 = (tfh3 >= -X_oh) && (tfh3 < IH - X_oh) && (tfw3 >= -X_ow) && (tfw3 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		Xs[buf][ty][tx].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
		Xs[buf][ty][tx].z = lx2 * tex1Dfetch<float>(X, Xoffset2 + xoffset);
		Xs[buf][ty][tx].w = lx3 * tex1Dfetch<float>(X, Xoffset3 + xoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = *(float4*)(&dYs[buf][ik][ty << 1]);

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
	}

	const int FH_FW_IC = FH * FW_IC;
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  *(float4*)(dst + oc0 + 4) = v1;
	*(float4*)(dst + oc1) = v2;  *(float4*)(dst + oc1 + 4) = v3;
	*(float4*)(dst + oc2) = v4;  *(float4*)(dst + oc2 + 4) = v5;
	*(float4*)(dst + oc3) = v6;  *(float4*)(dst + oc3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2)
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_4_4_TEXTURE
#define DECONV3D_DW_BGEMM_KERNEL_4_4_TEXTURE

//for GZ = N >> 3
//LB = 4: Size = 1, Time = 2.524 msec, Performace = 850.825 GFlop/s
//LB = 3: Size = 1, Time = 3.398 msec, Performace = 631.984 GFlop/s
template<int LB, int STEP>
__global__ void Ukernel_BGemm_4_4_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	int XNoffset = N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

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
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0 + XNoffset;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1 + XNoffset;

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
	zero_float(Xs[buf][Xs_y][Xs_x].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
	zero_float(Xs[buf][Xs_y][Xs_x].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
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
		zero_float(Xs[buf][Xs_y][Xs_x].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
		zero_float(Xs[buf][Xs_y][Xs_x].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
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
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0; //j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;
	*(float4*)(dst + oc1) = v1;
	*(float4*)(dst + oc2) = v2;
	*(float4*)(dst + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2), OH, OW is power of 2
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_4_4_OHW_2POW_TEXTURE
#define DECONV3D_DW_BGEMM_KERNEL_4_4_OHW_2POW_TEXTURE

//LB = 4: Size = 1, Time = 2.178 msec, Performace = 985.989 GFlop/s
//LB = 3: Size = 1, Time = 2.65  msec, Performace = 810.371 GFlop/s
template<int LB, int STEP>
__global__ void Ukernel_BGemm_4_4_OHW2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK_slice = (N_end - N_start) << LOH_OW;
	int XNoffset = N_start * IH*IW*IC; //X[N_start,...]
	deltaY += (N_start << LOH_OW)*OC; //deltaY[N_start,...]
	//======================================================================

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
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0 + XNoffset;
	const int Xoffset1 = (tfh1*IW + tfw1)*IC + tic1 + XNoffset;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	zero_float(Xs[buf][Xs_y][Xs_x].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
	zero_float(Xs[buf][Xs_y][Xs_x].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
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
		dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		zero_float(Xs[buf][Xs_y][Xs_x].x, lx0, tex1Dfetch<float>(X, Xoffset0 + xoffset));
		zero_float(Xs[buf][Xs_y][Xs_x].y, lx1, tex1Dfetch<float>(X, Xoffset1 + xoffset));
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
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0; //j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;
	*(float4*)(dst + oc1) = v1;
	*(float4*)(dst + oc2) = v2;
	*(float4*)(dst + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2),OH, OW, OC, IC is power of 2
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_4_4_OHW_OIC_2POW_TEXTURE
#define DECONV3D_DW_BGEMM_KERNEL_4_4_OHW_OIC_2POW_TEXTURE

//LB = 4: Size = 1, Time = 2.144 msec, Performace = 1001.62 GFlop/s
//LB = 3: Size = 1, Time = 2.57  msec, Performace =  835.597 GFlop/s
template<int LB, int STEP>
__global__ void Ukernel_BGemm_4_4_OHW_OIC2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ deltaY, int LOH, int LOW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int LIC, int LOC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OW_m1 = (1 << LOW) - 1;
	const int LOH_OW = LOH + LOW, OH_OW_m1 = (1 << LOH_OW) - 1;
	const int GK_slice = (N_end - N_start) << LOH_OW;
	int XNoffset = (N_start * IH*IW) << LIC; //X[N_start,...]
	deltaY += (N_start << LOH_OW) << LOC; //deltaY[N_start,...]
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int toc0 = ((tx & 1) << 1) + oc0;

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int FW_IC = FW << LIC, IC_m1 = (1 << LIC) - 1;
	get_fh_fw_ic_IC2pow(tj0, tfh0, tfw0, tic0);
	get_fh_fw_ic_IC2pow(tj1, tfh1, tfw1, tic1);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	tfh1 = tfh1 - oph, tfw1 = tfw1 - opw;
	const int Xoffset0 = ((tfh0*IW + tfw0) << LIC) + tic0 + XNoffset;
	const int Xoffset1 = ((tfh1*IW + tfw1) << LIC) + tic1 + XNoffset;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[(dY_k << LOC) + toc0]);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow) << LIC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
	Xs[buf][Xs_y][Xs_x].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
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
		dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[(dY_k << LOC) + toc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow_OHW2pow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		bool lx1 = (tfh1 >= -X_oh) && (tfh1 < IH - X_oh) && (tfw1 >= -X_ow) && (tfw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow) << LIC;
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
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + ((bz - 1) << LOC)*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0; //j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;
	*(float4*)(dst + oc1) = v1;
	*(float4*)(dst + oc2) = v2;
	*(float4*)(dst + oc3) = v3;
}

#endif

#endif