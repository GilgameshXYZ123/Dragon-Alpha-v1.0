#pragma once

#ifndef DECONV_3D_DELTAW_BGEMM_W1
#define DECONV_3D_DELTAW_BGEMM_W1

#define __dconv3D_deltaW_BGemm_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, N,IC,OC)\
	deconv3d_deltaW_BGemm_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, N,IC,OC,\
		GET_GN(OC), IC, GET_GK(N, IH, IW), 0, 0)

#define dconv3d_dW_BGemm_W1_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_BGemm_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, N,IC,OC,\
			GNr, GM, GK, next_oc_index, j_index);\
		deconv3d_deltaW_BGemm_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, N,IC,OC,\
            GN, GMr, GK, oc_index, next_j_index);\
		deconv3d_deltaW_BGemm_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, N,IC,OC,\
            GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		deconv3d_deltaW_BGemm_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, N,IC,OC,\
			 GNr, GM, GK, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_BGemm_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, N,IC,OC,\
			 GN, GMr, GK, oc_index, next_j_index);}}

//GN%4 == 0, GN >= 4
//GM%4 == 0, GM >= 4
//GK%4 == 0, As: GN >= 16, We have: GK >= 16
void deconv3d_deltaW_BGemm_W1(jlong *streams, int &index, int length, int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int N, int IC, int OC,
	int GN, int GM, int GK,
	int oc_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 127) && (GM > 127) && !(GK & 7)) {//GK % 8 == 0
		kBGemm88W1(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
		dconv3d_dW_BGemm_W1_Branch(127, 127); return;
	}
	if ((GN > 63) && (GM > 63)) {
		kBGemm88W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
		dconv3d_dW_BGemm_W1_Branch(63, 63); return;
	}
	if ((GN > 31) && (GM > 31)) {
		kBGemm44W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
		dconv3d_dW_BGemm_W1_Branch(31, 31); return;
	}
	if ((GN > 15) && (GM > 31)) {
		kBGemm24W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
		dconv3d_dW_BGemm_W1_Branch(15, 31); return;
	}
	if ((GN > 31) && (GM > 15)) {
		kBGemm42W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
		dconv3d_dW_BGemm_W1_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 15)) {
		kBGemm22W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
		dconv3d_dW_BGemm_W1_Branch(15, 15); return;
	}
	if ((GN > 7) && (GM > 15)) {
		kBGemm12W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
		dconv3d_dW_BGemm_W1_Branch(7, 15); return;
	}
	if ((GN > 15) && (GM > 7)) {
		kBGemm21W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
		dconv3d_dW_BGemm_W1_Branch(15, 7); return;
	}
	if ((GN > 7) && (GM > 7)) {
		kBGemm11W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
		dconv3d_dW_BGemm_W1_Branch(7, 7); return;
	}

	if (GN > 7) {
		kBGemm21W1(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
		dconv3d_dW_BGemm_W1_Branch(7, 3); return;
	}
	if (GM > 7) {
		kBGemm12W1(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
		dconv3d_dW_BGemm_W1_Branch(3, 7); return;
	}
	kBGemm11W1(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM);
}

#endif


#ifndef DECONV3D_DW_BGEMM_KERNEL_W1_H
#define DECONV3D_DW_BGEMM_KERNEL_W1_H

//oph = ph, opw = pw
//pretest==============================================================
//for [OH, OW] = 64, [FH, FW] = 1, N = 4
//k88<4>: Size = 0.25, Time = 2.692 msec, Performace = 199.432 GFlop/s
//k88<3>: Size = 0.25, Time = 2.7 msec  , Performace = 198.841 GFlop/s
//k44<4>: Size = 0.25, Time = 1.838 msec, Performace = 292.095 GFlop/s
//k44<3>: Size = 0.25, Time = 2.914 msec, Performace = 184.238 GFlop/s
//[FH, FW] = 2:
//k88<3>: Size = 1, Time = 3.388 msec, Performace = 633.85 GFlop/s
//[FH, FW] = [1, 2]: 
//k88<4>: Size = 0.5, Time = 2.706 msec, Performace = 396.8 GFlop/s
//k88<3>: Size = 0.5, Time = 2.752 msec, Performace = 390.168 GFlop/s
//k44<4>: Size = 0.5, Time = 2.142 msec, Performace = 501.28 GFlop/s
//[FH, FW] = [1, 4]
//k88<4>: Size = 1, Time = 2.702 msec, Performace = 794.776 GFlop/s
//k44<4>: Size = 1, Time = 3.26  msec, Performace = 658.737 GFlop/s
//FH*FW>=4, use k88 first, else use k44 first
//pretest==============================================================
//we have:
//	(1) FH * FW >= 2
//	(2) GN = OC:                GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW = IC: GM%4 == 0, GM >= 4
//	(4) GK = N  * OH * OW     : GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)

#ifndef DECONV3D_DW_BGEMM_KERNEL_W1_CALL
#define DECONV3D_DW_BGEMM_KERNEL_W1_CALL

#define kBGemm88W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM)\
	kernel_BGemm_8_8_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, oc_index, j_index)

#define kBGemm44W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM)\
	kernel_BGemm_4_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, oc_index, j_index)

#define kBGemm42W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM)\
	kernel_BGemm_4_2_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, oc_index, j_index)

#define kBGemm24W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM)\
	kernel_BGemm_2_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, oc_index, j_index)

#define kBGemm22W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM)\
	kernel_BGemm_2_2_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, oc_index, j_index)

#define kBGemm21W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM)\
	kernel_BGemm_2_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, oc_index, j_index)

#define kBGemm12W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM)\
	kernel_BGemm_1_2_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf,N, IC, OC, oc_index, j_index)

#define kBGemm11W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, GN, GM)\
	kernel_BGemm_1_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, N, IC, OC, oc_index, j_index)

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_8_8_W1
#define DECONV3D_DW_BGEMM_KERNEL_8_8_W1

//LB = 4: Size = 0.25, Time = 0.46 msec, Performace = 1167.11 GFlop/s
//LB = 3: Size = 0.25, Time = 0.51 msec, Performace = 1052.69 GFlop/s
//LB = 3: Size = 0.125, Time = 0.322 msec, Performace = 833.65 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_8_8_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	const int toc0 = ((tx >= STEP) << 2) + oc0;

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	const int tic0 = ((ty >= STEP) << 2) + ic0;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = IH * IW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + toc0]);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1);
	Xs[buf][ty][tx] = *(float4*)(&X[X_k*IC + tic0]);
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

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];
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
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + toc0]);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Xs[buf][ty][tx] = *(float4*)(&X[X_k*IC + tic0]);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = dYs[buf][ik][ty], a1 = dYs[buf][ik + STEP][ty];
		float4 b0 =  Xs[buf][ik][tx], b1 =  Xs[buf][ik + STEP][tx];

		simdMM4( v0, a0.x, b0); simdMM4( v1, a0.x, b1);
		simdMM4( v2, a0.y, b0); simdMM4( v3, a0.y, b1);
		simdMM4( v4, a0.z, b0); simdMM4( v5, a0.z, b1);
		simdMM4( v6, a0.w, b0); simdMM4( v7, a0.w, b1);
		simdMM4( v8, a1.x, b0); simdMM4( v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC, oc2 = oc1 + IC, oc3 = oc2 + IC;
	int oc4 = oc3 + IC, oc5 = oc4 + IC, oc6 = oc5 + IC, oc7 = oc6 + IC;

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_4_4_W1
#define DECONV3D_DW_BGEMM_KERNEL_4_4_W1

//LB = 4: Size = 0.5   , Time = 1.3   msec, Performace = 825.955 GFlop/s
//LB = 4: Size = 0.25  , Time = 0.674 msec, Performace = 796.544 GFlop/s
//LB = 4: Size = 0.0625, Time = 0.214 msec, Performace = 627.186 GFlop/s
//LB = 3: Size = 0.0625, Time = 0.254 msec, Performace = 528.416 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_4_4_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int toc0 = ((tx & 1) << 1) + oc0;

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tic0 = ((ty & 1) << 1) + ic0;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = IH * IW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(&X[X_k*IC + tic0]);
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
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(&X[X_k*IC + tic0]);
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

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC, oc2 = oc1 + IC, oc3 = oc2 + IC;

	*(float4*)(dst + oc0) = v0;
	*(float4*)(dst + oc1) = v1;
	*(float4*)(dst + oc2) = v2;
	*(float4*)(dst + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_4_2_W1
#define DECONV3D_DW_BGEMM_KERNEL_4_2_W1

//LB = 4: Size = 0.0625, Time = 0.238 msec, Performace = 563.94 GFlop/s
//LB = 3: Size = 0.0625, Time = 0.332 msec, Performace = 404.27 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_4_2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float   Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int toc0 = ((tx & 1) << 1) + oc0;

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	const int tic0 = (ty & 1) + ic0;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = IH * IW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

	//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = X[X_k*IC + tic0];
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
			float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);
			float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);

			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
			simdMM2(v2, a.z, b);
			simdMM2(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

		//load 1 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = X[X_k*IC + tic0];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);
		float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
		simdMM2(v2, a.z, b);
		simdMM2(v3, a.w, b);
	}

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC, oc2 = oc1 + IC, oc3 = oc2 + IC;

	*(float2*)(dst + oc0) = v0;
	*(float2*)(dst + oc1) = v1;
	*(float2*)(dst + oc2) = v2;
	*(float2*)(dst + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_2_4_W1
#define DECONV3D_DW_BGEMM_KERNEL_2_4_W1

//LB = 4: Size = 0.0625, Time = 0.282 msec, Performace = 475.949 GFlop/s
//LB = 3: Size = 0.0625, Time = 0.354 msec, Performace = 379.146 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_2_4_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, 
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int toc0 = (tx & 1) + oc0;

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tic0 = ((ty & 1) << 1) + ic0;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = IH * IW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = deltaY[dY_k*OC + toc0];

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(&X[X_k*IC + tic0]);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = *(float2*)(&dYs[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = deltaY[dY_k*OC + toc0];

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		X_k = ((ok << LB) + ty) >> 1;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(&X[X_k*IC + tic0]);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = *(float2*)(&dYs[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
	}

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * IC + ic0; 
	int oc1 = oc0 + IC;

	*(float4*)(dst + oc0) = v0;
	*(float4*)(dst + oc1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
#ifndef DECONV3D_DW_BGEMM_KERNEL_2_2_W1
#define DECONV3D_DW_BGEMM_KERNEL_2_2_W1

//LB = 4: Size = 0.053215, Time = 0.248 msec, Performace = 460.8 GFlop/s
//LB = 3: Size = 0.469238, Time = 3.226 msec, Performace = 312.363 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_2_2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, 
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float2  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + j_index;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = (bz * N_slice);
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = IH * IW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = tx;
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = ty;
		Xs[buf][ty][tx] = *(float2*)(&X[X_k*IC + ic0]);
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = dYs[buf][ik][ty];
			float2 b = Xs[buf][ik][tx];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty;
		Xs[buf][ty][tx] = *(float2*)(&X[X_k*IC + ic0]);
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = dYs[buf][ik][ty];
			float2 b = Xs[buf][ik][tx];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++) {
		float2 a = *(float2*)(&deltaY[k*OC + oc0]);//load 2 elements from deltaY
		float2 b = *(float2*)(&X[k*IC + ic0]);//load 2 elements from X
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP != 0--------------------------------------------

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * IC + ic0; 
	int oc1 = oc0 + IC;

	*(float2*)(dst + oc0) = v0;
	*(float2*)(dst + oc1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1)
#ifndef DECONV3D_DW_BGEMM_KERNEL_2_1_W1
#define DECONV3D_DW_BGEMM_KERNEL_2_1_W1

//LB = 4: Size = 0.053215, Time = 0.36 msec, Performace = 317.44 GFlop/s
//LB = 3: Size = 0.053215, Time = 0.46 msec, Performace = 248.431 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_2_1_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float   Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;

	//prepare for GM = IC
	int ic0 = ((blockIdx.x << LB) + tx) + j_index;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = (bz * N_slice);
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = IH * IW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = tx;
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ty;
		Xs[buf][ty][tx] = X[X_k*IC + ic0];
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = dYs[buf][ik][ty];
			float  b =  Xs[buf][ik][tx];
			simdMM2(v, b, a);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty;
		Xs[buf][ty][tx] = X[X_k*IC + ic0];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = dYs[buf][ik][ty];
			float  b = Xs[buf][ik][tx];
			simdMM2(v, b, a);
		}

	//when GK_slice%STEP != 0 -------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++) {
		float2 a = *(float2*)(&deltaY[k*OC + oc0]);//load 2 elements from deltaY
		float  b = X[k*IC + ic0];//load 1 element from X
		simdMM2(v, b, a);
	}
	//when GK_slice%STEP != 0 -------------------------------------------

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * IC + ic0; 
	int oc1 = oc0 + IC;

	dst[oc0] = v.x;
	dst[oc1] = v.y;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef DECONV3D_DW_BGEMM_KERNEL_1_2_W1
#define DECONV3D_DW_BGEMM_KERNEL_1_2_W1

//LB = 4: Size = 0.053215, Time = 0.484 msec, Performace = 236.112 GFlop/s
//LB = 3: Size = 0.053215, Time = 0.574 msec, Performace = 199.091 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_1_2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, 
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float2  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + j_index;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = (bz * N_slice);
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = IH * IW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = ty;
		Xs[buf][ty][tx] = *(float2*)(&X[X_k*IC + ic0]);
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float  a = dYs[buf][ik][ty];
			simdMM2(v, a, b);
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty;
		Xs[buf][ty][tx] = *(float2*)(&X[X_k*IC + ic0]);
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b =  Xs[buf][ik][tx];
			float  a = dYs[buf][ik][ty];
			simdMM2(v, a, b);
		}

	//when GK_slice % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++) {
		float  a = deltaY[k*OC + oc0];//load 1 element from deltaY
		float2 b = *(float2*)(&X[k*IC + ic0]);//load 2 elements from X
		simdMM2(v, a, b);
	}
	//when GK_slice % STEP != 0--------------------------------------------

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	*(float2*)(&dst[oc0*IC + ic0]) = v;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
#ifndef DECONV3D_DW_BGEMM_KERNEL_1_1_W1
#define DECONV3D_DW_BGEMM_KERNEL_1_1_W1

//LB = 4: Size = 0.053215, Time = 0.626 msec, Performace = 182.553 GFlop/s
//LB = 3: Size = 0.053215, Time = 0.718 msec, Performace = 159.162 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_1_1_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int N, int IC, int OC,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = IC
	int ic0 = ((blockIdx.x << LB) + tx) + j_index;
	
	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = (bz * N_slice);
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = IH * IW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = ty;
		Xs[buf][ty][tx] = X[X_k*IC + ic0];
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float v = 0;
	for (int ok = 1; ok < OK; ok++)
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

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty;
		Xs[buf][ty][tx] = X[X_k*IC + ic0];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float b =  Xs[buf][ik][tx];
			float a = dYs[buf][ik][ty];
			v += a * b;
		}

	//when GK_slice % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++) {
		float a = deltaY[k*OC + oc0];//load 1 element from deltaY
		float b = X[k*IC + ic0];//load 1 element from X
		v += a * b;
	}
	//when GK_slice % STEP != 0--------------------------------------------

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	dst[oc0*IC + ic0] = v;
}

#endif

#endif