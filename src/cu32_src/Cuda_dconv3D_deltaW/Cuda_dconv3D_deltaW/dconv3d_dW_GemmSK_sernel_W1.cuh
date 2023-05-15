#pragma once

#ifndef DECONV3D_DW_GEMMSK_SERNEL_W1_H
#define DECONV3D_DW_GEMMSK_SERNEL_W1_H

//Split K to improve parallism:
//	(1) FH = FW = 1
//	(2) GN = OC:           GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW: GM%4 == 0, GM >= 4
//	(4) GK = N  * OH * OW: GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
//	(6) ph = oph = 0, pw = opw = 0
//	(7) sh = sw = 1
#ifndef DECONV3D_DW_GEMMSK_SERNEL_W1_CALL
#define DECONV3D_DW_GEMMSK_SERNEL_W1_CALL

//LB = log2(BLOCK_SIZE)
//GZ = gridDim.z

//======[Small GM]===========================================
#define sGemmSK_8x2_1_W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	sernel_GemmSK_8x2_1_W1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM<<1>>LB, GN>>LB>>3, GZ), dim3(1<<LB>>1, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define sGemmSK_4x2_1_W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	sernel_GemmSK_4x2_1_W1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM<<1>>LB, GN>>LB>>2, GZ), dim3(1<<LB>>1, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define sGemmSK_2x2_1_W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	sernel_GemmSK_2x2_1_W1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM<<1>>LB, GN>>LB>>1, GZ), dim3(1<<LB>>1, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

//======[Small GN]===========================================
#define sGemmSK_1_4x2_W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	sernel_GemmSK_1_4x2_W1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN<<1>>LB, GZ), dim3(1<<LB, 1<<LB>>1), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define sGemmSK_1_2x2_W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	sernel_GemmSK_1_2x2_W1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN<<1>>LB, GZ), dim3(1<<LB, 1<<LB>>1), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#endif


//======[Small GM]===========================================
//(Y: BLOCK_SIZE*6, X: BLOCK_SIZE*0.5)
#ifndef DECONV3D_DW_GEMMSK_SERNEL_8X2_1_W1
#define DECONV3D_DW_GEMMSK_SERNEL_8X2_1_W1

//[IC =  8]: LB = 3: Size = 0.5, Time = 3.36 msec, Performace = 319.566 GFlop/s
//[IC =  4]: LB = 3: Size = 0.5, Time = 4    msec, Performace = 268.435 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmSK_8x2_1_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int IC, int OC,
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
	deltaW_buf += (bz - 1) * OC * IC;//Wstrid = FH * FW * IC = IC
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC
	int ic0 = ((blockIdx.x << LB >> 1) + tx) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 8 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = tx + GK_start;
		int yoffset1 = Y_k * OC;
		Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset1);
		Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset1 + 4);

		int yoffset2 = yoffset1 + (OC << LB >> 1);
		Ys[buf][tx + STEP2][(ty << 1)] = *(float4*)(deltaY + yoffset2);
		Ys[buf][tx + STEP2][(ty << 1) + 1] = *(float4*)(deltaY + yoffset2 + 4);

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float   b = Xs[buf][ik][tx];
			float4 a0 = Ys[buf][ik][(ty << 1)];
			float4 a1 = Ys[buf][ik][(ty << 1) + 1];

			simdMM4(v0, b, a0);
			simdMM4(v1, b, a1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		int yoffset1 = Y_k * OC;
		Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset1);
		Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset1 + 4);

		int yoffset2 = yoffset1 + (OC << LB >> 1);
		Ys[buf][tx + STEP2][(ty << 1)] = *(float4*)(deltaY + yoffset2);
		Ys[buf][tx + STEP2][(ty << 1) + 1] = *(float4*)(deltaY + yoffset2 + 4);

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float   b = Xs[buf][ik][tx];
			float4 a0 = Ys[buf][ik][(ty << 1)];
			float4 a1 = Ys[buf][ik][(ty << 1) + 1];

			simdMM4(v0, b, a0);
			simdMM4(v1, b, a1);
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice & (STEP - 1)); k < GK_slice; k++)
	{
		//load (1, 2) elements from (X, deltaY)
		float  b = X[(k + GK_start)*IC];
		int yoffset = (k + GK_start)*OC;
		float4 a0 = *(float4*)(deltaY + yoffset);
		float4 a1 = *(float4*)(deltaY + yoffset + 4);

		simdMM4(v0, b, a0);
		simdMM4(v1, b, a1);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * IC + ic0;
	const int oc1 = oc0 + IC, oc2 = oc1 + IC;
	const int oc3 = oc2 + IC, oc4 = oc3 + IC;
	const int oc5 = oc4 + IC, oc6 = oc5 + IC, oc7 = oc6 + IC;

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*0.5)
#ifndef DECONV3D_DW_GEMMSK_SERNEL_4X2_1_W1
#define DECONV3D_DW_GEMMSK_SERNEL_4X2_1_W1

//[IC =  8]: LB = 3: Size = 0.5, Time = 3.62 msec, Performace = 296.614 GFlop/s
//[IC =  4]: LB = 3: Size = 0.5, Time = 3.72 msec, Performace = 288.64  GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmSK_4x2_1_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int IC, int OC,
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
	deltaW_buf += (bz - 1) * OC * IC;//Wstrid = FH * FW * IC = IC
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC
	int ic0 = ((blockIdx.x << LB >> 1) + tx) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = tx + GK_start;
		int yoffset1 = Y_k * OC, yoffset2 = yoffset1 + (OC << LB >> 1);
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset1);
		Ys[buf][tx + STEP2][ty] = *(float4*)(deltaY + yoffset2);

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Xs[buf][ik][tx];
			float4 a = Ys[buf][ik][ty];
			simdMM4(v0, b, a);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		int yoffset1 = Y_k * OC, yoffset2 = yoffset1 + (OC << LB >> 1);
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset1);
		Ys[buf][tx + STEP2][ty] = *(float4*)(deltaY + yoffset2);

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Xs[buf][ik][tx];
			float4 a = Ys[buf][ik][ty];
			simdMM4(v0, b, a);
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice & (STEP - 1)); k < GK_slice; k++) {
		//load (1, 2) elements from (X, deltaY)
		float  b = X[(k + GK_start)*IC];
		float4 a = *(float4*)(deltaY + (k + GK_start)*OC);

		simdMM4(v0, b, a);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * IC + ic0;
	const int oc1 = oc0 + IC;
	const int oc2 = oc1 + IC;
	const int oc3 = oc2 + IC;

	deltaW[oc0] = v0.x;
	deltaW[oc1] = v0.y;
	deltaW[oc2] = v0.z;
	deltaW[oc3] = v0.w;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*0.5)
#ifndef DECONV3D_DW_GEMMSK_SERNEL_2X2_1_W1
#define DECONV3D_DW_GEMMSK_SERNEL_2X2_1_W1

//[IC =  8]: LB = 3: Size = 0.5, Time = 4.46 msec, Performace = 240.749 GFlop/s
//[IC =  4]: LB = 3: Size = 0.5, Time = 4.62 msec, Performace = 232.412 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmSK_2x2_1_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int IC, int OC,
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
	deltaW_buf += (bz - 1) * OC * IC;//Wstrid = FH * FW * IC = IC
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC
	int ic0 = ((blockIdx.x << LB >> 1) + tx) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = tx + GK_start;
		int yoffset1 = Y_k * OC, yoffset2 = yoffset1 + (OC << LB >> 1);
		Ys[buf][tx][ty] = *(float2*)(deltaY + yoffset1);
		Ys[buf][tx + STEP2][ty] = *(float2*)(deltaY + yoffset2);

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
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
		int yoffset1 = Y_k * OC, yoffset2 = yoffset1 + (OC << LB >> 1);
		Ys[buf][tx][ty] = *(float2*)(deltaY + yoffset1);
		Ys[buf][tx + STEP2][ty] = *(float2*)(deltaY + yoffset2);

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Xs[buf][ik][tx];
			float2 a = Ys[buf][ik][ty];
			simdMM2(v0, b, a);
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++) {
		//load (1, 2) elements from (X, deltaY)
		float b = X[(k + GK_start)*IC];
		float2 a = *(float2*)(deltaY + (k + GK_start)*OC);

		simdMM2(v0, b, a);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * IC + ic0;
	const int oc1 = oc0 + IC;

	deltaW[oc0] = v0.x;
	deltaW[oc1] = v0.y;
}

#endif


//======[Small GN]===========================================
//(Y: BLOCK_SIZE*0.5, X: BLOCK_SIZE*8)
#ifndef DECONV3D_DW_GEMMSK_SERNEL_1_4X2_W1
#define DECONV3D_DW_GEMMSK_SERNEL_1_4X2_W1

//[OC = 8]: LB = 4: Size = 0.5, Time = 3.382 msec, Performace = 317.487 GFlop/s
//[OC = 4]: LB = 3: Size = 0.5, Time = 3.562 msec, Performace = 301.444 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmSK_1_4x2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int IC, int OC,
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
	deltaW_buf += (bz - 1) * OC * IC;//Wstrid = FH * FW * IC = IC
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB >> 1) + ty) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		int xoffset1 = X_k * IC, xoffset2 = xoffset1 + (IC << LB >> 1);
		Xs[buf][ty][tx] = *(float4*)(X + xoffset1);
		Xs[buf][ty + STEP2][tx] = *(float4*)(X + xoffset2);

		//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = tx + GK_start;
		Ys[buf][tx][ty] = deltaY[Y_k * OC];
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = Xs[buf][ik][tx];
			float  a = Ys[buf][ik][ty];
			simdMM4(v0, a, b);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty + GK_start;
		int xoffset1 = X_k * IC, xoffset2 = xoffset1 + (IC << LB >> 1);
		Xs[buf][ty][tx] = *(float4*)(X + xoffset1);
		Xs[buf][ty + STEP2][tx] = *(float4*)(X + xoffset2);

		//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = deltaY[Y_k * OC];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = Xs[buf][ik][tx];
			float  a = Ys[buf][ik][ty];
			simdMM4(v0, a, b);
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice & (STEP - 1)); k < GK_slice; k++) {
		//load (2, 2) elements from (X, deltaY)
		float4 b = *(float4*)(X + (k + GK_start)*IC);
		float  a = deltaY[(k + GK_start)*OC];

		simdMM2(v0, a, b);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * IC + ic0;
	*(float4*)(deltaW + oc0) = v0;
}

#endif


//(Y: BLOCK_SIZE*0.5, X: BLOCK_SIZE*2)
#ifndef DECONV3D_DW_GEMMSK_SERNEL_1_2X2_W1
#define DECONV3D_DW_GEMMSK_SERNEL_1_2X2_W1

//[OC = 8]: LB = 4: Size = 0.5, Time = 4.654 msec, Performace = 230.714 GFlop/s
//[OC = 4]: LB = 3: Size = 0.5, Time = 4.69  msec, Performace = 228.943 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_GemmSK_1_2x2_W1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int IC, int OC,
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
	deltaW_buf += (bz - 1) * OC * IC;//Wstrid = FH * FW * IC = IC
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB >> 1) + ty) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		int xoffset1 = X_k * IC, xoffset2 = xoffset1 + (IC << LB >> 1);
		Xs[buf][ty][tx] = *(float2*)(X + xoffset1);
		Xs[buf][ty + STEP2][tx] = *(float2*)(X + xoffset2);

		//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = tx + GK_start;
		Ys[buf][tx][ty] = deltaY[Y_k * OC];
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float  a = Ys[buf][ik][ty];
			simdMM2(v0, a, b);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty + GK_start;
		int xoffset1 = X_k * IC, xoffset2 = xoffset1 + (IC << LB >> 1);
		Xs[buf][ty][tx] = *(float2*)(X + xoffset1);
		Xs[buf][ty + STEP2][tx] = *(float2*)(X + xoffset2);

		//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = deltaY[Y_k * OC];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float  a = Ys[buf][ik][ty];
			simdMM2(v0, a, b);
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++) {
		//load (2, 2) elements from (X, deltaY)
		float2 b = *(float2*)(X + (k + GK_start)*IC);
		float  a = deltaY[(k + GK_start)*OC];

		simdMM2(v0, a, b);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * IC + ic0;
	*(float2*)(deltaW + oc0) = v0;
}

#endif

#endif