#pragma once

#ifndef CONV_3D_GEMMR_A_KERNEL_TEXTURE_H
#define CONV_3D_GEMMR_A_KERNEL_TEXTURE_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
//N % 8 == 0
#ifndef CONV_3D_GEMMR_A_KERNEL_TEXTURE_CALL
#define CONV_3D_GEMMR_A_KERNEL_TEXTURE_CALL

//======[FW, IC is power of 2]================================
#define conv3dGemm_k88RA_fw_ic2pow_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, LFW, Y, OH, OW, N, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8RA_FW_IC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,LFW, Y,OH,OW, N,LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//======[FH = FW = 3]=========================================
#define conv3dGemm_k88RA_W3_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8RA_W3_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			oc_index,j_index) 

#define conv3dGemm_k88RA_W3_ic2pow_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8RA_W3_IC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,OH,OW, N,LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)  

//======[FH = FW = 5]=========================================
#define conv3dGemm_k88RA_W5_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8RA_W5_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			oc_index,j_index) 

#define conv3dGemm_k88RA_W5_ic2pow_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8RA_W5_IC2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,OH,OW, N,LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)  

#endif


//======[FW, IC is power of 2]================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8RA_FW_IC2POW_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8RA_FW_IC2POW_TEXTURE

//LB = 4: Size = 1, Time = 1.38845 msec, Performace = 1546.67 GFlop/s
//LB = 3: Size = 1, Time = 1.61206 msec, Performace = 1332.14 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8RA_FW_IC2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW, int FH, int LFW,
	float* __restrict__ Y, int OH, int OW,
	int N, int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int OW_N = OW * N;
	get_oh_ow_n(j0, oh0, ow0, n0);
	int Y0 = ((n0*OH + oh0)*OW + ow0)*OC + oc0;
	int Ystride = OH * OW * OC;
	int toh = oh0 * sh - ph, tow = ow0 * sw - pw;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X1 = (((tn0 + 1)*IH + toh)*IW + tow) << LIC;
	const int Xstride = (IH * IW) << LIC;

	//prepare for GK = FH * FW * IC
	const int LFW_IC = LFW + LIC, LFW_IC_m1 = (1 << LFW_IC) - 1;
	const int GK = FH << LFW_IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
	int xoffset = X1 + (X_fh << LIC)*IW + X_k;
	float4 x; bool lx = LOAD_X(toh, tow, X_fh, X_fw);
	x.x = tex1Dfetch<float>(X, xoffset - Xstride);
	x.y = tex1Dfetch<float>(X, xoffset);
	x.z = tex1Dfetch<float>(X, xoffset + Xstride);
	x.w = tex1Dfetch<float>(X, xoffset + (Xstride << 1));
	zero_float4(x, lx); Xs[buf][tx][ty] = x;

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
		int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
		int xoffset = X1 + (X_fh << LIC)*IW + X_k;
		float4 x; bool lx = LOAD_X(toh, tow, X_fh, X_fw);
		x.x = tex1Dfetch<float>(X, xoffset - Xstride);
		x.y = tex1Dfetch<float>(X, xoffset);
		x.z = tex1Dfetch<float>(X, xoffset + Xstride);
		x.w = tex1Dfetch<float>(X, xoffset + (Xstride << 1));
		zero_float4(x, lx); Xs[buf][tx][ty] = x;

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

	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//======[FH = FW = 3]=========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8RA_W3_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8RA_W3_TEXTURE

//LB = 4: Size = 1.125, Time = 1.66402 msec, Performace = 1451.86 GFlop/s
//LB = 3: Size = 1.125, Time = 1.94797 msec, Performace = 1240.23 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8RA_W3_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int OW_N = OW * N;
	get_oh_ow_n(j0, oh0, ow0, n0);
	int Y0 = ((n0*OH + oh0)*OW + ow0)*OC + oc0;
	int Ystride = OH * OW * OC;
	int toh = oh0 * sh - ph, tow = ow0 * sw - pw;
	int tn1 = n0 + ((tx >= STEP) << 2) + 1;
	int X1 = ((tn1*IH + toh)*IW + tow)*IC;
	const int Xstride = IH * IW * IC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = X1 + (X_fh * IW + X_fw - Idx)*IC + X_k;
	float4 x; bool lx = LOAD_X(toh, tow, X_fh, X_fw);
	x.x = tex1Dfetch<float>(X, xoffset - Xstride);
	x.y = tex1Dfetch<float>(X, xoffset);
	x.z = tex1Dfetch<float>(X, xoffset + Xstride);
	x.w = tex1Dfetch<float>(X, xoffset + (Xstride << 1));
	zero_float4(x, lx); Xs[buf][tx][ty] = x;

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
		int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = X1 + (X_fh * IW + X_fw - Idx)*IC + X_k;
		float4 x; bool lx = LOAD_X(toh, tow, X_fh, X_fw);
		x.x = tex1Dfetch<float>(X, xoffset - Xstride);
		x.y = tex1Dfetch<float>(X, xoffset);
		x.z = tex1Dfetch<float>(X, xoffset + Xstride);
		x.w = tex1Dfetch<float>(X, xoffset + (Xstride << 1));
		zero_float4(x, lx); Xs[buf][tx][ty] = x;

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

	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8RA_W3_IC2POW_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8RA_W3_IC2POW_TEXTURE

//LB = 4: Size = 1.125, Time = 1.59607 msec, Performace = 1513.67 GFlop/s
//LB = 3: Size = 1.125, Time = 1.83213 msec, Performace = 1318.64 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8RA_W3_IC2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH, int OW,
	int N, int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int OW_N = OW * N;
	get_oh_ow_n(j0, oh0, ow0, n0);
	int Y0 = ((n0*OH + oh0)*OW + ow0)*OC + oc0;
	int Ystride = OH * OW * OC;
	int toh = oh0 * sh - ph, tow = ow0 * sw - pw;
	int tn1 = n0 + ((tx >= STEP) << 2) + 1;
	int X1 = ((tn1*IH + toh)*IW + tow) << LIC;
	const int Xstride = (IH * IW) << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = X1 + ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	float4 x; bool lx = LOAD_X(toh, tow, X_fh, X_fw);
	x.x = tex1Dfetch<float>(X, xoffset - Xstride);
	x.y = tex1Dfetch<float>(X, xoffset);
	x.z = tex1Dfetch<float>(X, xoffset + Xstride);
	x.w = tex1Dfetch<float>(X, xoffset + (Xstride << 1));
	zero_float4(x, lx); Xs[buf][tx][ty] = x;

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
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = X1 + ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		float4 x; bool lx = LOAD_X(toh, tow, X_fh, X_fw);
		x.x = tex1Dfetch<float>(X, xoffset - Xstride);
		x.y = tex1Dfetch<float>(X, xoffset);
		x.z = tex1Dfetch<float>(X, xoffset + Xstride);
		x.w = tex1Dfetch<float>(X, xoffset + (Xstride << 1));
		zero_float4(x, lx); Xs[buf][tx][ty] = x;

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

	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//======[FH = FW = 5]=========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8RA_W5_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8RA_W5_TEXTURE

//LB = 4: Size = 1.5625, Time = 2.25885 msec, Performace = 1485.46 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.6397  msec, Performace = 1271.14 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8RA_W5_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int OW_N = OW * N;
	get_oh_ow_n(j0, oh0, ow0, n0);
	int Y0 = ((n0*OH + oh0)*OW + ow0)*OC + oc0;
	int Ystride = OH * OW * OC;
	int toh = oh0 * sh - ph, tow = ow0 * sw - pw;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X1 = (((tn0 + 1)*IH + toh)*IW + tow)*IC;
	const int Xstride = IH * IW * IC;

	//prepare for GK = FH * FW * IC
	const int GK = 25 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = X1 + (X_fh * IW + X_fw - Idx)*IC + X_k;
	float4 x; bool lx = LOAD_X(toh, tow, X_fh, X_fw);
	x.x = tex1Dfetch<float>(X, xoffset - Xstride);
	x.y = tex1Dfetch<float>(X, xoffset);
	x.z = tex1Dfetch<float>(X, xoffset + Xstride);
	x.w = tex1Dfetch<float>(X, xoffset + (Xstride << 1));
	zero_float4(x, lx); Xs[buf][tx][ty] = x;

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
		int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = X1 + (X_fh * IW + X_fw - Idx)*IC + X_k;
		float4 x; bool lx = LOAD_X(toh, tow, X_fh, X_fw);
		x.x = tex1Dfetch<float>(X, xoffset - Xstride);
		x.y = tex1Dfetch<float>(X, xoffset);
		x.z = tex1Dfetch<float>(X, xoffset + Xstride);
		x.w = tex1Dfetch<float>(X, xoffset + (Xstride << 1));
		zero_float4(x, lx); Xs[buf][tx][ty] = x;

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

	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef CONV_3D_GEMM_KERNEL_8_8RA_W5_IC2POW_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8RA_W5_IC2POW_TEXTURE

//LB = 4: Size = 1.5625, Time = 2.16223 msec, Performace = 1551.84 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.4621  msec, Performace = 1362.84 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_8_8RA_W5_IC2pow_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	float* __restrict__ Y, int OH, int OW,
	int N, int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int OW_N = OW * N;
	get_oh_ow_n(j0, oh0, ow0, n0);
	int Y0 = ((n0*OH + oh0)*OW + ow0)*OC + oc0;
	int Ystride = OH * OW * OC;
	int toh = oh0 * sh - ph, tow = ow0 * sw - pw;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X1 = (((tn0 + 1)*IH + toh)*IW + tow) << LIC;
	const int Xstride = (IH * IW) << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 25 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = X1 + ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	float4 x; bool lx = LOAD_X(toh, tow, X_fh, X_fw);
	x.x = tex1Dfetch<float>(X, xoffset - Xstride);
	x.y = tex1Dfetch<float>(X, xoffset);
	x.z = tex1Dfetch<float>(X, xoffset + Xstride);
	x.w = tex1Dfetch<float>(X, xoffset + (Xstride << 1));
	zero_float4(x, lx); Xs[buf][tx][ty] = x;

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
		int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = X1 + ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		float4 x; bool lx = LOAD_X(toh, tow, X_fh, X_fw);
		x.x = tex1Dfetch<float>(X, xoffset - Xstride);
		x.y = tex1Dfetch<float>(X, xoffset);
		x.z = tex1Dfetch<float>(X, xoffset + Xstride);
		x.w = tex1Dfetch<float>(X, xoffset + (Xstride << 1));
		zero_float4(x, lx); Xs[buf][tx][ty] = x;

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

	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif

#endif