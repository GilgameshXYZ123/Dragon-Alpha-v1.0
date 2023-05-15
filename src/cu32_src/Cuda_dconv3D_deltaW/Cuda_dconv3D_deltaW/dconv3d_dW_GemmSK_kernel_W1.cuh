#pragma once

#ifndef DECONV3D_DW_GEMMSK_KERNEL_W1_H
#define DECONV3D_DW_GEMMSK_KERNEL_W1_H

//Split K to improve parallism:
//	(1) FH = FW = 1
//	(2) GN = OC:           GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW: GM%4 == 0, GM >= 4
//	(4) GK = N  * OH * OW: GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
//	(6) ph = oph = 0, pw = opw = 0
//	(7) sh = sw = 1
#ifndef DECONV3D_DW_GEMMSK_KERNEL_W1_CALL
#define DECONV3D_DW_GEMMSK_KERNEL_W1_CALL

//LB = log2(BLOCK_SIZE)
//GZ = gridDim.z

//======[Common]==============================================
#define kGemmSK88W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_8_8_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define kGemmSK84W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_8_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define kGemmSK48W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_4_8_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define kGemmSK44W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_4_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define kGemmSK82W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_8_2_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define kGemmSK28W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_2_8_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define kGemmSK42W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_4_2_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define kGemmSK24W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_2_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

//======[Small]===============================================
#define kGemmSK22W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_2_2_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

//------------------------------------------------------------
#define kGemmSK81W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_8_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define kGemmSK41W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_4_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define kGemmSK21W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_2_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

//------------------------------------------------------------
#define kGemmSK14W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_1_4_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define kGemmSK12W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_1_2_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#define kGemmSK11W1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM)\
	kernel_GemmSK_1_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GK, GK_slice, oc_index, j_index)

#endif


//======[Common]==============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_8_8_W1
#define DECONV3D_DW_GEMMSK_KERNEL_8_8_W1

//LB = 4: Size = 0.5, Time = 0.838 msec, Performace = 1281.31 GFlop/s
//LB = 3: Size = 0.5, Time = 0.838 msec, Performace = 1281.31 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_8_8_W1(
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
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += ((tx >= STEP) << 2) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	X += ((ty >= STEP) << 2) + ic0;//X[0, 0, 0, tic0]

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	Xs[buf][ty][tx] = *(float4*)(X + X_k * IC);

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
	
		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		Xs[buf][ty][tx] = *(float4*)(X + X_k * IC);

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
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

	oc0 = oc0 * IC + ic0;
	const int oc1 = oc0 + IC, oc2 = oc1 + IC, oc3 = oc2 + IC;
	const int oc4 = oc3 + IC, oc5 = oc4 + IC, oc6 = oc5 + IC, oc7 = oc6 + IC;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_8_4_W1
#define DECONV3D_DW_GEMMSK_KERNEL_8_4_W1

//LB = 4: Size = 0.5, Time = 0.924 msec, Performace = 1162.06 GFlop/s
//LB = 3: Size = 0.5, Time = 1.004 msec, Performace = 1069.46 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_8_4_W1(
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
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

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
	deltaY += ((tx >= STEP) << 2) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	X += ((ty & 1) << 1) + ic0;//X[0, 0, 0, tic0]

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(X + X_k * IC);

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
		Xs[buf][Xs_y][Xs_x] = *(float2*)(X + X_k * IC);

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

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC, oc2 = oc1 + IC, oc3 = oc2 + IC;
	int oc4 = oc3 + IC, oc5 = oc4 + IC, oc6 = oc5 + IC, oc7 = oc6 + IC;

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_4_8_W1
#define DECONV3D_DW_GEMMSK_KERNEL_4_8_W1

//LB = 4: Size = 0.5, Time = 1.062 msec, Performace = 1011.06 GFlop/s
//LB = 3: Size = 0.5, Time = 1.068 msec, Performace = 1005.38 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_4_8_W1(
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
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

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
	deltaY += ((tx & 1) << 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	X += ((ty >= STEP) << 2) + ic0;//X[0, 0, 0, tic0]

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	Xs[buf][ty][tx] = *(float4*)(X + X_k * IC);

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
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

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		Xs[buf][ty][tx] = *(float4*)(X + X_k * IC);

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

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC, oc2 = oc1 + IC, oc3 = oc2 + IC;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
	*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
	*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_4_4_W1
#define DECONV3D_DW_GEMMSK_KERNEL_4_4_W1

//LB = 4: Size = 0.5, Time = 1.178 msec, Performace = 911.496 GFlop/s
//LB = 3: Size = 0.5, Time = 1.262 msec, Performace = 850.825 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_4_4_W1(
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
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

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
	deltaY += ((tx & 1) << 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	X += ((ty & 1) << 1) + ic0;//X[0, 0, 0, tic0]

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(X + X_k * IC);
	
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

		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(X + X_k * IC);

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
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

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC, oc2 = oc1 + IC, oc3 = oc2 + IC;

	*(float4*)(deltaW + oc0) = v0;
	*(float4*)(deltaW + oc1) = v1;
	*(float4*)(deltaW + oc2) = v2;
	*(float4*)(deltaW + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_8_2_W1
#define DECONV3D_DW_GEMMSK_KERNEL_8_2_W1

//LB = 4: Size = 0.5, Time = 1.232 msec, Performace = 871.544 GFlop/s
//LB = 3: Size = 0.5, Time = 1.272 msec, Performace = 844.137 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_8_2_W1(
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
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

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
	deltaY += ((tx >= STEP) << 2) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	X += (ty & 1) + ic0;//X[0, 0, 0, tic0]

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = X[X_k * IC];

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

		//load 1 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		Xs[buf][Xs_y][Xs_x] = X[X_k * IC];

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

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC, oc2 = oc1 + IC, oc3 = oc2 + IC;
	int oc4 = oc3 + IC, oc5 = oc4 + IC, oc6 = oc5 + IC, oc7 = oc6 + IC;

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_2_8_W1
#define DECONV3D_DW_GEMMSK_KERNEL_2_8_W1

//LB = 4: Size = 0.5, Time = 1.536 msec, Performace = 699.051 GFlop/s
//LB = 3: Size = 0.5, Time = 1.59  msec, Performace = 675.309 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_2_8_W1(
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
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//follow k88

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
	deltaY += (tx & 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	X += ((ty >= STEP) << 2) + ic0;//X[0, 0, 0, tic0]

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	Xs[buf][ty][tx] = *(float4*)(X + X_k * IC);

	//load 1 element  from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1) + GK_start;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = deltaY[Y_k * OC];
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
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

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		Xs[buf][ty][tx] = *(float4*)(X + X_k * IC);

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
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

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC;

	*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
	*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_4_2_W1
#define DECONV3D_DW_GEMMSK_KERNEL_4_2_W1

//LB = 4: Size = 0.5, Time = 1.466 msec, Performace = 732.43 GFlop/s
//LB = 3: Size = 0.5, Time = 1.956 msec, Performace = 548.948 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_4_2_W1(
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
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];

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
	deltaY += ((tx & 1) << 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	X += (ty & 1) + ic0;//X[0, 0, 0, tic0]

	//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = X[X_k * IC];

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
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
			simdMM2(v2, a.z, b);
			simdMM2(v3, a.w, b);
		}
		buf ^= 1;


		//load 1 element  from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		Xs[buf][Xs_y][Xs_x] = X[X_k * IC];

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (((ok << LB) + tx) >> 1) + GK_start;
		Ys[buf][Ys_x][Ys_y] = *(float2*)(deltaY + Y_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
		simdMM2(v2, a.z, b);
		simdMM2(v3, a.w, b);
	}

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC, oc2 = oc1 + IC, oc3 = oc2 + IC;

	*(float2*)(deltaW + oc0) = v0;
	*(float2*)(deltaW + oc1) = v1;
	*(float2*)(deltaW + oc2) = v2;
	*(float2*)(deltaW + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_GEMMSK_KERNEL_2_4_W1
#define DECONV3D_DW_GEMMSK_KERNEL_2_4_W1

//LB = 4: Size = 0.5, Time = 1.752 msec, Performace = 612.866 GFlop/s
//LB = 3: Size = 0.5, Time = 2.02  msec, Performace = 531.555 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_2_4_W1(
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
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

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
	deltaY += (tx & 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	X += ((ty & 1) << 1) + ic0;//X[0, 0, 0, tic0]

	//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = (ty >> 1) + GK_start;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = *(float2*)(X + X_k * IC);

	//load 1 element from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
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

		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = (((ok << LB) + ty) >> 1) + GK_start;
		Xs[buf][Xs_y][Xs_x] = *(float2*)(X + X_k * IC);

		//load 1 element  from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
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

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC;

	*(float4*)(deltaW + oc0) = v0;
	*(float4*)(deltaW + oc1) = v1;
}

#endif


//======[Small]===============================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
#ifndef DECONV3D_DW_GEMMSK_KERNEL_2_2_W1
#define DECONV3D_DW_GEMMSK_KERNEL_2_2_W1

//LB = 4: Size = 0.5, Time = 1.92  msec, Performace = 559.24  GFlop/s
//LB = 3: Size = 0.5, Time = 2.344 msec, Performace = 458.081 GFlop/s
//LB = 4: Size = 0.42572, Time = 1.666 msec, Performace = 548.756 GFlop/s
//LB = 3: Size = 0.42572, Time = 2.004 msec, Performace = 456.201 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_2_2_W1(
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
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		Xs[buf][ty][tx] = *(float2*)(X + X_k * IC);

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = tx + GK_start;
		Ys[buf][tx][ty] = *(float2*)(deltaY + Y_k * OC);
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float2 a = Ys[buf][ik][ty];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}
		buf ^= 1;

		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty + GK_start;
		Xs[buf][ty][tx] = *(float2*)(X + X_k * IC);
		
		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = *(float2*)(deltaY + Y_k * OC);
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float2 a = Ys[buf][ik][ty];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++) {
		//load (2, 2) elements from (X, deltaY)
		float2 b = *(float2*)(X + (k + GK_start)*IC);
		float2 a = *(float2*)(deltaY + (k + GK_start)*OC);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * IC + ic0;
	int oc1 = oc0 + IC;

	*(float2*)(deltaW + oc0) = v0;
	*(float2*)(deltaW + oc1) = v1;
}

#endif


//------------------------------------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*1)
#ifndef DECONV3D_DW_GEMMSK_KERNEL_8_1_W1
#define DECONV3D_DW_GEMMSK_KERNEL_8_1_W1

//[IC = 16]: LB = 4: Size = 0.5, Time = 2.06 msec, Performace = 521.234 GFlop/s
//[IC =  8]: LB = 3: Size = 0.5, Time = 2.04 msec, Performace = 526.344 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_8_1_W1(
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
	int ic0 = ((blockIdx.x << LB) + tx) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];

		//load 8 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = tx + GK_start;
		int yoffset = Y_k * OC;
		Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset);
		Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset + 4);
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

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		int yoffset = Y_k * OC;
		Ys[buf][tx][(ty << 1)] = *(float4*)(deltaY + yoffset);
		Ys[buf][tx][(ty << 1) + 1] = *(float4*)(deltaY + yoffset + 4);
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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1)
#ifndef DECONV3D_DW_GEMMSK_KERNEL_4_1_W1
#define DECONV3D_DW_GEMMSK_KERNEL_4_1_W1

//[IC = 16]: LB = 4: Size = 0.5, Time = 2.34 msec, Performace = 458.864 GFlop/s
//[IC =  8]: LB = 3: Size = 0.5, Time = 2.76 msec, Performace = 389.037 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_4_1_W1(
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
	int ic0 = ((blockIdx.x << LB) + tx) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = tx + GK_start;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
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

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
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


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1)
#ifndef DECONV3D_DW_GEMMSK_KERNEL_2_1_W1
#define DECONV3D_DW_GEMMSK_KERNEL_2_1_W1

//[IC = 16]: LB = 4: Size = 0.5, Time = 3.14 msec, Performace = 341.956 GFlop/s
//[IC =  8]: LB = 3: Size = 0.5, Time = 3.84 msec, Performace = 279.62 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_2_1_W1(
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
	int ic0 = ((blockIdx.x << LB) + tx) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = tx + GK_start;
		Ys[buf][tx][ty] = *(float2*)(deltaY + Y_k * OC);
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

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = *(float2*)(deltaY + Y_k * OC);
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
	int oc1 = oc0 + IC;

	deltaW[oc0] = v0.x;
	deltaW[oc1] = v0.y;
}

#endif


//------------------------------------------------------------
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*4)
#ifndef DECONV3D_DW_GEMMSK_KERNEL_1_4_W1
#define DECONV3D_DW_GEMMSK_KERNEL_1_4_W1

//[OC = 16]: LB = 4: Size = 0.5, Time = 3.68 msec, Performace = 291.778 GFlop/s
//[OC =  8]: LB = 3: Size = 0.5, Time = 3.9  msec, Performace = 275.318 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_1_4_W1(
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
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		Xs[buf][ty][tx] = *(float4*)(X + X_k * IC);

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
		Xs[buf][ty][tx] = *(float4*)(X + X_k * IC);

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


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef DECONV3D_DW_GEMMSK_KERNEL_1_2_W1
#define DECONV3D_DW_GEMMSK_KERNEL_1_2_W1

//[OC = 16]: LB = 4: Size = 0.5, Time = 4.26  msec, Performace = 252.052 GFlop/s
//[OC =  8]: LB = 3: Size = 0.5, Time = 4.132 msec, Performace = 259.86 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_1_2_W1(
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
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 2 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		Xs[buf][ty][tx] = *(float2*)(X + X_k * IC);

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
		Xs[buf][ty][tx] = *(float2*)(X + X_k * IC);

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


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1)
#ifndef DECONV3D_DW_GEMMSK_KERNEL_1_1_W1
#define DECONV3D_DW_GEMMSK_KERNEL_1_1_W1

//LB = 4: Size = 0.42572, Time = 4.11  msec, Performace = 222.44  GFlop/s
//LB = 3: Size = 0.42572, Time = 4.766 msec, Performace = 191.823 GFlop/s
template<int LB, int STEP>
__global__ void kernel_GemmSK_1_1_W1(
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
	__shared__ float Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float Xs[2][1 << LB][(1 << LB) + 1];

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
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	deltaY += oc0;//deltaY[0, 0, 0, oc0]

	//prepare for GM = IC
	int ic0 = ((blockIdx.x << LB) + tx) + j_index;
	X += ic0;//X[0, 0, 0, ic0]

	const int OK = (GK_slice >> LB);
	if (OK) {
		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];

		//load 1 element from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = tx + GK_start;
		Ys[buf][tx][ty] = deltaY[Y_k * OC];
		__syncthreads();
	}

	//compute area-----------------------------------------------------
	float v = 0;
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float b = Xs[buf][ik][tx];
			float a = Ys[buf][ik][ty];
			v += a * b;
		}
		buf ^= 1;

		//load 1 element from X[N, IH, IW, IC]: Xe[IC, IH, IW, N] 
		int X_k = (ok << LB) + ty + GK_start;
		Xs[buf][ty][tx] = X[X_k * IC];

		//load 1 element from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = (ok << LB) + tx + GK_start;
		Ys[buf][tx][ty] = deltaY[Y_k * OC];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float b = Xs[buf][ik][tx];
			float a = Ys[buf][ik][ty];
			v += a * b;
		}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++) {
		//load (2, 2) elements from (X, deltaY)
		float b = X[(k + GK_start)*IC];
		float a = deltaY[(k + GK_start)*OC];

		v += a * b;
	}
	//when GK % STEP != 0--------------------------------------------

	deltaW[oc0 * IC + ic0] = v;
}

#endif

#endif